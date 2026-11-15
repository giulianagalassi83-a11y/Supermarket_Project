"""Bi-encoder with an incremental float16 cache keyed by (text_hash, model_id@revision).

Only texts missing from the cache are sent to the model, so a weekly re-scrape encodes just the
new or changed products. Changing the model or its revision ignores every old row on purpose.
"""

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.pipeline import config


def _hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _prefix(model_id, kind):
    """e5 models need 'query: ' / 'passage: '; bge-m3 and others take raw text."""
    return f"{kind}: " if "e5" in str(model_id).lower() else ""


def folder_revision(path):
    """Short content hash of a local model folder's weights (stands in for a Hub commit hash)."""
    weights = sorted(Path(path).glob("*.safetensors"))
    digest = hashlib.sha1()
    for w in weights:
        digest.update(w.read_bytes())
    return digest.hexdigest()[:12]


class Encoder:
    def __init__(self, model_id, revision, cache_path=config.SIM_CACHE_PATH, model=None, prefix_id=None):
        self.model_id, self.revision, self.cache_path, self.model = model_id, revision, Path(cache_path), model
        self.prefix_id = prefix_id or model_id  # a fine-tuned folder keeps its base model's prefixes
        self._vectors = self._read()
        self._new = {}

    @property
    def key(self):
        return f"{self.model_id}@{self.revision}"

    def _read(self):
        if not self.cache_path.exists():
            return {}
        cache = pd.read_parquet(self.cache_path, filters=[("model_key", "==", self.key)])
        return {h: np.asarray(v, dtype=np.float32) for h, v in zip(cache["text_hash"], cache["vector"])}

    def _model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_id, revision=self.revision)  # Hub models only
        return self.model

    def encode(self, texts, kind: Literal["query", "passage"] = "passage"):
        """L2-normalized float32 rows, in the order of `texts`; only cache misses reach the model."""
        prefixed = [_prefix(self.prefix_id, kind) + t for t in texts]
        hashes = [_hash(t) for t in prefixed]
        missing = list(dict.fromkeys(t for t, h in zip(prefixed, hashes) if h not in self._vectors))
        if missing:
            vectors = self._model().encode(missing, batch_size=128, show_progress_bar=len(missing) > 1000,
                                           convert_to_numpy=True, normalize_embeddings=True)
            for t, v in zip(missing, vectors):
                v = np.asarray(v, dtype=np.float32)
                v = v / np.linalg.norm(v)
                # round-trip through float16 now so fresh and cached vectors are identical
                v = v.astype(np.float16).astype(np.float32)
                self._vectors[_hash(t)] = self._new[_hash(t)] = v
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows = np.vstack([self._vectors[h] for h in hashes])
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def save_cache(self):
        """Append the vectors computed since the last save; returns the number of rows written."""
        if not self._new:
            return 0
        rows = pd.DataFrame({
            "text_hash": list(self._new), "model_key": self.key,
            "vector": [v.astype(np.float16) for v in self._new.values()],
        })
        if self.cache_path.exists():
            rows = pd.concat([pd.read_parquet(self.cache_path), rows], ignore_index=True)
            rows["vector"] = [np.asarray(v, dtype=np.float16) for v in rows["vector"]]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = rows.drop_duplicates(["text_hash", "model_key"], keep="last")
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            "text_hash": pa.array(rows["text_hash"].tolist()),
            "model_key": pa.array(rows["model_key"].tolist()),
            "vector": pa.array([pa.array(v, type=pa.float16()) for v in rows["vector"]], type=pa.list_(pa.float16())),
        })
        pq.write_table(table, self.cache_path)
        written, self._new = len(self._new), {}
        return written


def load_encoder(which: Literal["base", "finetuned"] = "finetuned", cache_path=config.SIM_CACHE_PATH):
    """The configured encoder, or None when sentence-transformers or the fine-tuned folder is missing."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    if which == "finetuned":
        path = Path(config.SIM["finetuned_path"])
        if not path.exists():
            return None
        return Encoder(path.name, folder_revision(path), cache_path, prefix_id=config.SIM["encoder_id"],
                       model=_load_local(path))
    return Encoder(config.SIM["encoder_id"], config.SIM["encoder_revision"], cache_path)


def _load_local(path):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(path))

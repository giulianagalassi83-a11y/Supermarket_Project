"""Cross-encoder re-ranker for the top-k candidate pairs only; scores are memoized in-process."""

import numpy as np

from src.pipeline import config

CHUNK, PAUSE_S = 2048, 1.0


class Reranker:
    def __init__(self, model_id=config.SIM["reranker_id"], revision=config.SIM["reranker_revision"], model=None):
        self.model_id, self.revision, self.model = model_id, revision, model
        self._memo = {}

    def _model(self):
        if self.model is None:
            import torch
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_id, revision=self.revision, max_length=256,
                activation_fn=torch.nn.Identity(),  # raw logits; the sigmoid is applied below
                model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {},
            )
        return self.model

    def score(self, pairs):
        """Probability-like score in (0, 1) for each (text_a, text_b), in input order."""
        pairs = [tuple(p) for p in pairs]
        missing = list(dict.fromkeys(p for p in pairs if p not in self._memo))
        if missing:
            import time

            from src.similarity.train import HOT_C, cool_down, gpu_temp

            for start in range(0, len(missing), CHUNK):  # chunked with pauses: laptop GPU guard (see train.py)
                chunk = missing[start:start + CHUNK]
                logits = np.asarray(self._model().predict(chunk, batch_size=64, show_progress_bar=False),
                                    dtype=np.float64)
                for p, logit in zip(chunk, logits):
                    self._memo[p] = 1 / (1 + np.exp(-logit))
                if start + CHUNK < len(missing):
                    time.sleep(PAUSE_S)
                    if (t := gpu_temp()) is not None and t >= HOT_C:
                        cool_down()
        return np.array([self._memo[p] for p in pairs], dtype=np.float32)


def load_reranker():
    """The configured re-ranker, or None when sentence-transformers is missing."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    return Reranker()

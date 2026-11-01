"""Step 4: assign each product a type from the taxonomy (zero-shot, by text similarity).

Products and type descriptions become vectors (sentence embeddings, cached in a parquet file;
TF-IDF on character n-grams if embeddings aren't available). A product is compared only with the
types of the search terms that returned it. See the decision rule in `apply_rule`.
"""

import hashlib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pipeline import config
from src.pipeline.text import strip_accents
from src.pipeline.units import canonical_unit

TAXONOMY_COLUMNS = ["type_id", "search_term", "label", "description", "is_negative", "unit"]


# ---------------------------------------------------------------- taxonomy and texts

def load_taxonomy(path):
    """Read product_types.csv and check it; raises ValueError naming the bad row."""
    tax = pd.read_csv(path, encoding="utf-8-sig", dtype={"type_id": str})
    missing = set(TAXONOMY_COLUMNS) - set(tax.columns)
    if missing:
        raise ValueError(f"taxonomy is missing columns: {sorted(missing)}")
    tax = tax[TAXONOMY_COLUMNS].copy()
    tax["is_negative"] = tax["is_negative"].fillna(0).astype(int)
    tax["unit"] = tax["unit"].map(canonical_unit)
    duplicated = tax["type_id"][tax["type_id"].duplicated()]
    if len(duplicated):
        raise ValueError(f"duplicated type_id: {list(duplicated)}")
    unknown = tax[~tax["search_term"].isin(config.SEARCH_TERMS)]
    if len(unknown):
        raise ValueError(f"unknown search_term in rows: {unknown[['type_id', 'search_term']].values.tolist()}")
    no_unit = tax[(tax["is_negative"] == 0) & tax["unit"].isna()]
    if len(no_unit):
        raise ValueError(f"real types need a unit: {list(no_unit['type_id'])}")
    return tax


def types_for_db(taxonomy):
    """The taxonomy plus the 'no_relevante' type every rejected product points to."""
    extra = pd.DataFrame([{
        "type_id": config.NO_RELEVANTE, "search_term": None, "label": "No relevante",
        "description": "Resultado de búsqueda que no es el producto buscado", "is_negative": 1, "unit": None,
    }])
    return pd.concat([taxonomy, extra], ignore_index=True)


def product_text(name_clean, chain_category):
    """The text embedded for a product: clean name, plus the chain's category when it has one."""
    if chain_category is None or pd.isna(chain_category) or not str(chain_category).strip():
        return name_clean
    return f"{name_clean}. Categoría: {chain_category}"


# ---------------------------------------------------------------- vectorizers

def _hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_model(model_name):
    """The sentence-transformers model, or None if the package isn't installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(model_name)


def _read_cache(cache_path, model_name):
    if not cache_path.exists():
        return {}
    cache = pd.read_parquet(cache_path)
    cache = cache[cache["model_name"] == model_name]
    return {h: np.asarray(v, dtype=np.float32) for h, v in zip(cache["text_hash"], cache["vector"])}


def _write_cache(cache_path, model_name, new_vectors):
    rows = pd.DataFrame({
        "text_hash": list(new_vectors), "model_name": model_name,
        "vector": [v.astype(np.float32).tolist() for v in new_vectors.values()],
    })
    if cache_path.exists():
        rows = pd.concat([pd.read_parquet(cache_path), rows], ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.drop_duplicates(["text_hash", "model_name"], keep="last").to_parquet(cache_path, index=False)


class EmbeddingVectorizer:
    method = "embedding"

    def __init__(self, model_name, vectors, model=None):
        self.model_name, self.vectors, self.model = model_name, vectors, model

    def encode(self, texts):
        misses = [t for t in dict.fromkeys(texts) if _hash(t) not in self.vectors]
        if misses:
            if self.model is None:
                self.model = _load_model(self.model_name)
            if self.model is None:
                raise RuntimeError("Text not in the embedding cache and sentence-transformers is not installed")
            for t, v in zip(misses, self.model.encode(misses, batch_size=64, show_progress_bar=False)):
                self.vectors[_hash(t)] = np.asarray(v, dtype=np.float32)
        rows = np.vstack([self.vectors[_hash(t)] for t in texts])
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class TfidfFallback:
    method = "tfidf"
    model_name = config.TFIDF_NAME

    def __init__(self, texts, missing=0):
        self.missing = missing
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(texts)

    def encode(self, texts):
        return self.tfidf.transform(texts)  # sparse, rows already L2-normalized


def get_vectorizer(texts_needed, cache_path=config.CACHE_PATH, model_name=config.MODEL_NAME):
    """Embeddings if every text is cached or the model can be loaded; otherwise TF-IDF for the whole run."""
    vectors = _read_cache(cache_path, model_name)
    misses = [t for t in dict.fromkeys(texts_needed) if _hash(t) not in vectors]
    if not misses:
        return EmbeddingVectorizer(model_name, vectors)
    model = _load_model(model_name)
    if model is None:
        return TfidfFallback(list(texts_needed), missing=len(misses))
    computed = model.encode(misses, batch_size=64, show_progress_bar=True)
    new_vectors = {_hash(t): np.asarray(v, dtype=np.float32) for t, v in zip(misses, computed)}
    _write_cache(cache_path, model_name, new_vectors)
    vectors.update(new_vectors)
    return EmbeddingVectorizer(model_name, vectors, model)


def similarity(a, b):
    """Cosine similarity of row-normalized matrices (dense or sparse) → dense array."""
    s = a @ b.T
    return s.toarray() if sparse.issparse(s) else np.asarray(s)


# ---------------------------------------------------------------- classification

def candidate_scores(products, hits, taxonomy, vectorizer):
    """Long table: one row per (product, candidate type) with the similarity score.

    Candidates are the types (real and negative) of the search terms that returned the product.
    """
    texts = [product_text(n, c) for n, c in zip(products["name_clean"], products["chain_category"])]
    scores = similarity(vectorizer.encode(texts), vectorizer.encode(list(taxonomy["description"])))
    cands = hits.merge(taxonomy[["type_id", "search_term", "is_negative"]], on="search_term")
    cands = cands.drop_duplicates(["product_key", "type_id"])
    row = pd.Series(range(len(products)), index=products["product_key"].values)
    col = pd.Series(range(len(taxonomy)), index=taxonomy["type_id"].values)
    cands = cands[cands["product_key"].isin(row.index)]
    cands["score"] = scores[row[cands["product_key"]].values, col[cands["type_id"]].values]
    return cands[["product_key", "type_id", "is_negative", "score"]].reset_index(drop=True)


def _excluded(category):
    text = "" if pd.isna(category) else str(category).lower()
    return any(word in text for word in config.EXCLUDED_CATEGORIES)


def apply_rule(cands, products, taxonomy, thresholds, vectorizer):
    """Decision rule (spec §6.4):
    1. chain category on the exclusion list → no_relevante (category_exclusion)
    2. closest candidate is a negative anchor, or best real type scores below tau → no_relevante
    3. otherwise the closest real type; low_confidence when it beats the runner-up by less than delta
    """
    tau, delta = thresholds["tau"], thresholds["delta"]
    real = cands[cands["is_negative"] == 0].sort_values(["product_key", "score"], ascending=[True, False])
    rank = real.groupby("product_key").cumcount()
    first = real[rank == 0].set_index("product_key")
    second = real[rank == 1].set_index("product_key")
    best_negative = cands[cands["is_negative"] == 1].groupby("product_key")["score"].max()

    out = products[["product_key", "chain_category"]].set_index("product_key")
    out["type_id"] = first["type_id"]
    out["score"] = first["score"]
    out["runner_up_type_id"] = second["type_id"]
    out["margin"] = out["score"] - second["score"].reindex(out.index).fillna(0)
    out["low_confidence"] = out["margin"] < delta
    out["method"] = vectorizer.method
    out["model_name"] = vectorizer.model_name

    rejected = (best_negative.reindex(out.index) > out["score"]) | (out["score"] < tau) | out["score"].isna()
    out.loc[rejected, "runner_up_type_id"] = out.loc[rejected, "type_id"]  # best real type, kept for review
    out.loc[rejected, ["type_id", "margin", "low_confidence"]] = [config.NO_RELEVANTE, np.nan, False]

    no_hits = ~out.index.isin(cands["product_key"])
    out.loc[no_hits, ["type_id", "score", "runner_up_type_id", "margin", "low_confidence", "method", "model_name"]] = \
        [None, np.nan, None, np.nan, False, "unclassified", None]

    excluded = out["chain_category"].map(_excluded)
    out.loc[excluded, ["type_id", "score", "runner_up_type_id", "margin", "low_confidence", "method", "model_name"]] = \
        [config.NO_RELEVANTE, np.nan, None, np.nan, False, "category_exclusion", None]

    out["low_confidence"] = out["low_confidence"].astype(bool)
    return out.drop(columns="chain_category").reset_index()


def decide(products, hits, taxonomy, vectorizer, thresholds):
    """Classify every product: candidate scores + decision rule."""
    cands = candidate_scores(products, hits, taxonomy, vectorizer)
    return apply_rule(cands, products, taxonomy, thresholds, vectorizer)


# ---------------------------------------------------------------- similar-product search

def find_similar(query_or_product_key, k=5, chain=None, type_id=None, *, products, classification, vectorizer):
    """The k products most similar to a product (by key) or to free text, optionally per chain / type."""
    pool = products.merge(classification[["product_key", "type_id"]], on="product_key", how="left")
    if query_or_product_key in set(pool["product_key"]):
        q = pool.loc[pool["product_key"] == query_or_product_key].iloc[0]
        query_text = product_text(q["name_clean"], q["chain_category"])
        pool = pool[pool["product_key"] != query_or_product_key]
    else:
        query_text = strip_accents(query_or_product_key)
    if chain is not None:
        pool = pool[pool["chain_id"] == chain]
    if type_id is not None:
        pool = pool[pool["type_id"] == type_id]
    if pool.empty:
        return pool.assign(similarity=[])[["product_key", "name", "chain_id", "type_id", "similarity"]]
    texts = [product_text(n, c) for n, c in zip(pool["name_clean"], pool["chain_category"])]
    pool = pool.assign(similarity=similarity(vectorizer.encode(texts), vectorizer.encode([query_text]))[:, 0])
    return pool.nlargest(k, "similarity")[["product_key", "name", "chain_id", "type_id", "similarity"]]

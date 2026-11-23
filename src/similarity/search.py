"""Product search for the dashboard: embeddings + TF-IDF, with a TF-IDF-only fallback.

Product vectors come from the embedding cache only (written by `python -m src.similarity.build`);
only the query is encoded. Products missing from the cache are scored by TF-IDF alone. Without
sentence-transformers or the fine-tuned model (encoder None), search runs in TF-IDF mode.
"""

import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.similarity.data import match_text

COLUMNS = ["product_key", "name", "chain_id", "type_id", "score"]
W_COSINE, W_TFIDF, TYPE_BONUS = 0.7, 0.3, 0.1


def _fold(text):
    """Lowercase, accents removed, spaces squeezed."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


class Searcher:
    def __init__(self, products, encoder, taxonomy):
        """products: product_key, name, chain_id, type_id (+ brand); taxonomy: type_id, label."""
        self.products = products.reset_index(drop=True)
        brand = self.products["brand"] if "brand" in self.products else pd.Series("", index=self.products.index)
        self.texts = [match_text(n, b) for n, b in zip(self.products["name"], brand)]
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.matrix = self.tfidf.fit_transform([_fold(t) for t in self.texts])
        labels = taxonomy.set_index("type_id")["label"].map(_fold)
        self.type_label = self.products["type_id"].map(labels).fillna("").to_numpy()
        self.encoder, self.vectors, self.has_vector = encoder, None, np.zeros(len(self.products), dtype=bool)
        if encoder is not None:
            mask, vectors = encoder.cached(self.texts, "query")
            self.has_vector = np.asarray(mask, dtype=bool)
            if self.has_vector.any():
                self.vectors = vectors

    @property
    def mode(self):
        return "hybrid" if self.encoder is not None else "tfidf"

    def search(self, query, k=20, chain=None):
        q = _fold(query or "")
        if not q:
            return pd.DataFrame(columns=COLUMNS)
        tfidf = (self.matrix @ self.tfidf.transform([q]).T).toarray().ravel()
        score = tfidf.copy()
        if self.vectors is not None:
            cosine = self.vectors @ self.encoder.encode([query.strip()], "query")[0]
            score[self.has_vector] = W_COSINE * cosine + W_TFIDF * tfidf[self.has_vector]
        bonus = np.array([bool(lab) and (lab in q or (len(q) >= 4 and q in lab)) for lab in self.type_label])
        score = score + TYPE_BONUS * bonus
        out = self.products.assign(score=score)
        if chain is not None:
            out = out[out["chain_id"] == chain]
        return out.sort_values("score", ascending=False, kind="stable").head(k)[COLUMNS].reset_index(drop=True)

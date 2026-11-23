"""In-memory cosine index over normalized vectors (numpy is enough at ~18k rows), filterable by chain and type."""

import numpy as np
import pandas as pd


class VectorIndex:
    def __init__(self, keys, vectors, meta: pd.DataFrame):
        self.keys = np.asarray(keys, dtype=object)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        meta = meta.reindex(self.keys)
        self.chain = meta["chain_id"].to_numpy(dtype=object)
        self.type = meta["type_id"].to_numpy(dtype=object)

    def topk(self, query_vecs, k, chain=None, type_ids=None):
        """(keys, scores), each (n_queries, k). A query with a type_id only sees products of that type;
        rows with fewer than k candidates are padded with '' / -inf."""
        scores = np.asarray(query_vecs, dtype=np.float32) @ self.vectors.T
        if chain is not None:
            scores[:, self.chain != chain] = -np.inf
        if type_ids is not None:
            for i, t in enumerate(type_ids):
                if t is not None and not (isinstance(t, float) and np.isnan(t)):
                    scores[i, self.type != t] = -np.inf
        k = min(k, scores.shape[1])
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(part, np.argsort(-np.take_along_axis(scores, part, 1), axis=1, kind="stable"), 1)
        top_scores = np.take_along_axis(scores, order, 1)
        top_keys = np.where(np.isfinite(top_scores), self.keys[order], "")
        return top_keys, top_scores

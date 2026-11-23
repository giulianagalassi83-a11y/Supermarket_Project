"""Cross-chain matching: candidates → attribute veto → re-rank features → logistic decision → threshold → grouping."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.pipeline import config
from src.similarity import attributes
from src.similarity.index import VectorIndex

FEATURES = ["cosine", "ce_score", "brand_match", "size_ratio", "size_known", "variant_agree", "pack_match"]


def candidates(queries, targets, encoder, k=config.SIM["top_k"], use_type=True):
    """Top-k targets per query by bi-encoder cosine: long table key_a, key_b, cosine, rank.

    Frames need product_key, mtext, chain_id, type_id. With use_type, a typed query only sees its type.
    """
    index = VectorIndex(targets["product_key"], encoder.encode(list(targets["mtext"]), "query"),
                        targets.set_index("product_key"))
    types = None
    if use_type:
        types = [t if isinstance(t, str) and t != config.NO_RELEVANTE else None for t in queries["type_id"]]
    keys, scores = index.topk(encoder.encode(list(queries["mtext"]), "query"), k, type_ids=types)
    n, kk = keys.shape
    out = pd.DataFrame({
        "key_a": np.repeat(queries["product_key"].to_numpy(), kk),
        "key_b": keys.ravel(), "cosine": scores.ravel(), "rank": np.tile(np.arange(1, kk + 1), n),
    })
    return out[out["key_b"] != ""].reset_index(drop=True)


def apply_veto(cands, attrs, level):
    """Drop pairs with a known attribute conflict at this level; vetoed reasons go to out.attrs['vetoed']."""
    reasons = [attributes.conflict(attrs[a], attrs[b], level) for a, b in zip(cands["key_a"], cands["key_b"])]
    keep = np.array([r is None for r in reasons], dtype=bool)
    out = cands[keep].reset_index(drop=True)
    out.attrs["vetoed"] = dict(Counter(r for r in reasons if r))
    return out


def add_features(cands, texts, attrs, reranker=None):
    """Add ce_score (0 without a re-ranker) and the attribute agreement features."""
    out = cands.copy()
    if reranker is not None and len(out):
        out["ce_score"] = reranker.score(list(zip(out["key_a"].map(texts), out["key_b"].map(texts))))
    else:
        out["ce_score"] = 0.0
    feats = pd.DataFrame([attributes.features(attrs[a], attrs[b]) for a, b in zip(out["key_a"], out["key_b"])],
                         index=out.index, columns=FEATURES[2:], dtype=float)
    return pd.concat([out, feats], axis=1)


def fit_decider(feats, y, features=FEATURES):
    """Logistic decision model over `features` (all FEATURES by default); remembers them in .features_."""
    model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000).fit(feats[features], y)
    model.features_ = list(features)
    return model


def predict(model, feats):
    """Match probability of each candidate row under a fitted decider."""
    return model.predict_proba(feats[model.features_])[:, 1] if len(feats) else np.zeros(0)


def choose_threshold(probs, y, target, min_accepted=20):
    """Lowest threshold whose precision ≥ target with at least min_accepted accepted; 1.0 (accept nothing) if none."""
    probs, y = np.asarray(probs, dtype=float), np.asarray(y, dtype=int)
    best = 1.0
    for t in np.unique(probs)[::-1]:
        accepted = probs >= t
        if accepted.sum() >= min_accepted and y[accepted].mean() >= target:
            best = float(t)
    return best


def best_per_query(cands_with_prob, threshold):
    """The highest-probability candidate per key_a, kept only if prob ≥ threshold."""
    top = cands_with_prob.sort_values("prob", ascending=False, kind="stable").drop_duplicates("key_a")
    return top[top["prob"] >= threshold].reset_index(drop=True)


def mutual_best(a_to_b, b_to_a):
    """Pairs that are each other's best match (stops A→B→C chaining)."""
    back = set(zip(b_to_a["key_b"], b_to_a["key_a"]))
    keep = np.array([(a, b) in back for a, b in zip(a_to_b["key_a"], a_to_b["key_b"])], dtype=bool)  # [] would select columns
    return a_to_b[keep].reset_index(drop=True)


def group(matches, chain_of):
    """Canonical products: components of the match graph, built strongest edge first, never two products of one
    chain in a component (an edge that would do so is dropped). Columns: canonical_id, product_key, chain_id."""
    parent, members = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key in pd.unique(pd.concat([matches["key_a"], matches["key_b"]])):
        parent[key], members[key] = key, {chain_of(key)}
    for a, b in matches.sort_values("prob", ascending=False, kind="stable")[["key_a", "key_b"]].itertuples(index=False):
        ra, rb = find(a), find(b)
        if ra == rb or members[ra] & members[rb]:
            continue
        parent[rb] = ra
        members[ra] |= members.pop(rb)
    roots = {k: find(k) for k in parent}
    ids = {r: f"c{i:05d}" for i, r in enumerate(sorted(set(roots.values())))}
    return pd.DataFrame({
        "canonical_id": [ids[roots[k]] for k in parent],
        "product_key": list(parent), "chain_id": [chain_of(k) for k in parent],
    })

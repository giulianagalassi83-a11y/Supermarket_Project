"""Evaluation for the similarity engine: retrieval metrics, grouped folds, baselines and result tables."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from src.pipeline import config


def retrieval_metrics(ranked_keys, gold):
    """top-1, top-5 and MRR of each row's gold key in its ranking (beyond the ranking → reciprocal rank 0)."""
    ranked_keys = np.asarray(ranked_keys, dtype=object)
    hits = ranked_keys == np.asarray(gold, dtype=object)[:, None]
    found = hits.any(axis=1)
    rank = np.where(found, hits.argmax(axis=1) + 1, np.inf)
    return {
        "top1": float(np.mean(rank <= 1)),
        "top5": float(np.mean(rank <= 5)),
        "mrr": float(np.mean(np.where(found, 1 / rank, 0.0))),
    }


def grouped_folds(pairs, n=config.SIM["n_folds"]):
    """Deterministic GroupKFold on the BM product's search_term: a category is never in train and test."""
    return list(GroupKFold(n_splits=n).split(pairs, groups=pairs["search_term"]))


def tfidf_rank(query_texts, target_texts, target_keys, k):
    """Baseline ranking: char_wb 3–5 TF-IDF cosine (as the pipeline's fallback)."""
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(list(query_texts) + list(target_texts))
    scores = (tfidf.transform(query_texts) @ tfidf.transform(target_texts).T).toarray()
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return np.asarray(target_keys, dtype=object)[order]


def write_result(table, row):
    """Append one row to data/similarity/results/<table>.csv."""
    path = config.SIM_RESULTS_DIR / f"{table}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def prf(accepted, gold):
    """Precision = correct accepted / accepted; recall = correct accepted / all queries. Both Series by query."""
    accepted = accepted.reindex(gold.index)
    made = accepted.notna()
    correct = made & (accepted == gold)
    precision = float(correct.sum() / made.sum()) if made.any() else float("nan")
    recall = float(correct.sum() / len(gold)) if len(gold) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if made.any() and (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accepted": int(made.sum())}


# ---------------------------------------------------------------- matching cross-validation (spec §8, Task 8)

VARIANTS = ["tfidf", "base", "finetuned", "finetuned+attrs", "finetuned+attrs+rerank", "finetuned+attrs+rerank_ft"]
_ATTR_FEATURES = ["brand_match", "size_ratio", "size_known", "variant_agree", "pack_match"]
_VARIANT_FEATURES = {
    "tfidf": ["cosine"], "base": ["cosine"], "finetuned": ["cosine"],
    "finetuned+attrs": ["cosine"] + _ATTR_FEATURES,
    "finetuned+attrs+rerank": ["cosine", "ce_score"] + _ATTR_FEATURES,
    "finetuned+attrs+rerank_ft": ["cosine", "ce_score"] + _ATTR_FEATURES,  # per-fold fine-tuned re-ranker
}
CE_SETTINGS = "e2-n7"  # tag of the `train --cv-reranker` run whose held-out scores are evaluated
FOLD_SETTINGS = (3, 3)  # (epochs, per_pair) of the `train --cv` run whose cached embeddings are evaluated


class _NoModel:
    def encode(self, texts, **kw):
        raise RuntimeError(f"{len(texts)} texts missing from the CV cache: run python -m src.similarity.train --cv")


def _oof_candidates(variant, products, pairs, folds, k):
    """Top-k Consum candidates for every BM query, each scored by a model that never trained on that query."""
    from src.similarity.encoder import Encoder
    from src.similarity.index import VectorIndex
    from src.similarity.train import CV_CACHE_PATH, fold_model_key

    text = products.set_index("product_key")["mtext"]
    consum = products[products["chain_id"] == "consum"].reset_index(drop=True)
    frames = []
    for i, (_, test) in enumerate(folds):
        q = pairs.loc[test, "key_bm"].to_numpy()
        if variant == "tfidf":
            tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(list(text[q]) + list(consum["mtext"]))
            sims = (tf.transform(text[q]) @ tf.transform(consum["mtext"]).T).toarray()
            order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
            keys, cos = consum["product_key"].to_numpy()[order], np.take_along_axis(sims, order, 1)
        else:
            if variant == "base":
                enc = Encoder(config.SIM["encoder_id"], config.SIM["encoder_revision"])
            else:
                model_id, revision = fold_model_key(i, *FOLD_SETTINGS)
                enc = Encoder(model_id, revision, CV_CACHE_PATH, model=_NoModel(), prefix_id=config.SIM["encoder_id"])
            index = VectorIndex(consum["product_key"], enc.encode(list(consum["mtext"]), "query"),
                                consum.set_index("product_key"))
            keys, cos = index.topk(enc.encode(list(text[q]), "query"), k)
        frames.append(pd.DataFrame({"key_a": np.repeat(q, keys.shape[1]), "key_b": keys.ravel(),
                                    "cosine": cos.ravel(), "fold": i}))
    cands = pd.concat(frames, ignore_index=True)
    cands["y"] = (cands["key_b"] == cands["key_a"].map(pairs.set_index("key_bm")["key_consum"])).astype(int)
    return cands


def _top5(scored, queries):
    """(n_queries, 5) array of each query's best keys by prob, padded with ''."""
    lists = (scored.sort_values(["key_a", "prob"], ascending=[True, False])
             .groupby("key_a")["key_b"].apply(lambda s: list(s)[:5]))
    return np.array([(lists.get(q, []) + [""] * 5)[:5] for q in queries], dtype=object)


def run_matching_cv(products, pairs, variant, k=config.SIM["top_k"], reranker=None):
    """Held-out metrics per outer fold: decider fit on the other folds, threshold chosen by an inner CV on them."""
    from src.similarity import attributes
    from src.similarity.match import add_features, apply_veto, best_per_query, choose_threshold, fit_decider, predict

    folds = grouped_folds(pairs)
    cands = _oof_candidates("finetuned" if variant.startswith("finetuned") else variant, products, pairs, folds, k)
    by_key = products.set_index("product_key")
    keys = pd.unique(pd.concat([cands["key_a"], cands["key_b"]]))
    attrs = {key: attributes.extract(by_key.at[key, "name"], by_key.at[key, "brand"], by_key.at[key, "chain_id"])
             for key in keys}
    if "attrs" in variant:
        cands = apply_veto(cands, attrs, "same")
    cands = add_features(cands, by_key["mtext"].to_dict(), attrs, reranker if variant.endswith("rerank") else None)
    if variant.endswith("rerank_ft"):
        from src.similarity.train import CE_CV_SCORES_PATH

        ce = pd.read_parquet(CE_CV_SCORES_PATH)
        ce = ce[ce["tag"] == CE_SETTINGS].set_index(["key_a", "key_b"])["ce_score"]
        idx = pd.MultiIndex.from_arrays([cands["key_a"], cands["key_b"]])
        if not idx.isin(ce.index).all():
            raise RuntimeError("held-out re-ranker scores missing: run python -m src.similarity.train --cv-reranker")
        cands["ce_score"] = ce.reindex(idx).to_numpy(dtype=float)
    features = _VARIANT_FEATURES[variant]
    gold = pairs.set_index("key_bm")["key_consum"]
    rows = []
    for i, (_, test_idx) in enumerate(folds):
        train = cands[cands["fold"] != i]
        oof = np.zeros(len(train))
        for j in sorted(train["fold"].unique()):  # inner CV → out-of-fold probabilities for the threshold
            inner = (train["fold"] == j).to_numpy()
            oof[inner] = predict(fit_decider(train[~inner], train.loc[~inner, "y"], features), train[inner])
        best = train.assign(prob=oof).sort_values("prob", ascending=False, kind="stable").drop_duplicates("key_a")
        threshold = choose_threshold(best["prob"], best["y"], config.SIM["precision_target"])
        model = fit_decider(train, train["y"], features)
        queries = pairs.loc[test_idx, "key_bm"].to_numpy()
        test = cands[cands["fold"] == i].assign(prob=lambda d: predict(model, d))
        accepted = best_per_query(test, threshold).set_index("key_a")["key_b"].reindex(queries)
        rows.append({"variant": variant, "fold": i, "threshold": round(threshold, 4),
                     **retrieval_metrics(_top5(test, queries), gold[queries].to_numpy(dtype=object)),
                     **prf(accepted, gold[queries])})
    table = pd.DataFrame(rows)
    mean = {"variant": variant, "fold": "mean", **table.drop(columns=["variant", "fold"]).mean().round(4).to_dict()}
    return pd.concat([table, pd.DataFrame([mean])], ignore_index=True)


if __name__ == "__main__":
    import sys

    from src.similarity.data import barcode_pairs, load_products

    if sys.argv[1:2] == ["matching"]:
        from src.similarity.rerank import load_reranker

        products = load_products()
        pairs = barcode_pairs(products)
        reranker = load_reranker()
        only = sys.argv[2:]  # optional variant names: run only these and replace their rows in ablation.csv
        out = pd.concat([run_matching_cv(products, pairs, v, reranker=reranker) for v in (only or VARIANTS)],
                        ignore_index=True)
        path = config.SIM_RESULTS_DIR / "ablation.csv"
        if only and path.exists():
            old = pd.read_csv(path)
            out = pd.concat([old[~old["variant"].isin(only)], out], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        print(out[out["fold"] == "mean"].to_string(index=False))

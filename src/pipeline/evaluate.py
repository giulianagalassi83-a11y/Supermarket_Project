"""Evaluate the classifier against hand labels and tune its thresholds (tau, delta).

Labels live in data/labels/hand_labels.csv: product_key, type_id, prefilled_type_id, confirmed.
Half of the labels tune the thresholds, the other half measure accuracy (spec §6.7).
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.pipeline import config
from src.pipeline.classify import apply_rule, candidate_scores

LABEL_COLUMNS = ["product_key", "type_id", "prefilled_type_id", "confirmed"]
MIN_LABELS = 10


def load_labels(path=config.LABELS_PATH):
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_csv(path, encoding="utf-8")


def sample_for_labelling(products, hits, classification, labelled, per_term=5, seed=config.SEED):
    """`per_term` unlabelled products per search term, spread across chains, with the classifier's guess."""
    pool = (
        hits.drop_duplicates("product_key")  # a product is labelled once, under its first term
        .merge(products[["product_key", "chain_id", "name", "chain_category"]], on="product_key")
        .merge(classification[["product_key", "type_id"]].rename(columns={"type_id": "prefilled_type_id"}),
               on="product_key", how="left")
    )
    pool = pool[~pool["product_key"].isin(labelled)]
    pool = pool.sample(frac=1, random_state=seed)
    pool["chain_turn"] = pool.groupby(["search_term", "chain_id"]).cumcount()  # round-robin over chains
    pool = pool.sort_values("chain_turn", kind="stable")
    picked = pool.groupby("search_term", sort=False).head(per_term)
    return picked.drop(columns="chain_turn").reset_index(drop=True)


def validate_labels(labels, products, taxonomy):
    """Confirmed labels whose product and type still exist; plus a list of the problems found."""
    labels = labels[labels["confirmed"] == 1]
    known_types = set(taxonomy.loc[taxonomy["is_negative"] == 0, "type_id"]) | {config.NO_RELEVANTE}
    problems = []
    missing = ~labels["product_key"].isin(products["product_key"])
    problems += [f"product no longer exists: {k}" for k in labels.loc[missing, "product_key"]]
    unknown = ~labels["type_id"].isin(known_types)
    problems += [f"unknown type_id '{t}' for {k}" for k, t in labels.loc[unknown & ~missing, ["product_key", "type_id"]].values]
    return labels[~missing & ~unknown].reset_index(drop=True), problems


def split(labels, seed=config.SEED):
    """Deterministic 50/50 split: (tune, test)."""
    order = np.random.default_rng(seed).permutation(len(labels))
    half = len(labels) // 2
    return labels.iloc[order[:half]].reset_index(drop=True), labels.iloc[order[half:]].reset_index(drop=True)


def macro_f1(pred, truth):
    return precision_recall_fscore_support(truth, pred, average="macro", zero_division=0)[2]


def tune_tau(predict, truth, grid=np.arange(0.05, 0.80, 0.01)):
    """The tau whose predictions (predict(tau)) give the best macro-F1 on `truth`."""
    scores = [macro_f1(predict(tau), truth) for tau in grid]
    return round(float(grid[int(np.argmax(scores))]), 2)


def tune_delta(margins, correct, grid=np.arange(0.0, 0.15, 0.005), min_flagged=3):
    """The delta whose 'low confidence' flag best separates wrong from right predictions."""
    best, best_gap = 0.0, -1.0
    for delta in grid:
        flagged = margins < delta
        if flagged.sum() < min_flagged or (~flagged).sum() == 0:
            continue
        gap = (1 - correct[flagged].mean()) - (1 - correct[~flagged].mean())
        if gap > best_gap:
            best, best_gap = float(delta), gap
    return round(best, 3)


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def report(pred, truth):
    pred, truth = pd.Series(pred).reset_index(drop=True), pd.Series(truth).reset_index(drop=True)
    correct = int((pred == truth).sum())
    p, r, f, _ = precision_recall_fscore_support(truth, pred, average="macro", zero_division=0)
    rel_pred, rel_true = pred != config.NO_RELEVANTE, truth != config.NO_RELEVANTE
    return {
        "n": len(truth),
        "accuracy": correct / len(truth) if len(truth) else math.nan,
        "accuracy_ci95": wilson(correct, len(truth)),
        "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "relevant_precision": (rel_pred & rel_true).sum() / rel_pred.sum() if rel_pred.sum() else math.nan,
        "relevant_recall": (rel_pred & rel_true).sum() / rel_true.sum() if rel_true.sum() else math.nan,
        "confusion": pd.crosstab(truth.rename("true"), pred.rename("predicted")),
    }


def evaluate(prepared, labels):
    """Tune tau/delta on half the labels and report on the other half, for one vectorizer.

    `prepared` is the dict returned by build.prepare().
    """
    products, taxonomy, vectorizer = prepared["products"], prepared["taxonomy"], prepared["vectorizer"]
    labels, problems = validate_labels(labels, products, taxonomy)
    if len(labels) < MIN_LABELS:
        raise ValueError(f"only {len(labels)} usable labels (need {MIN_LABELS}): run the labelling cell first")
    tune_set, test_set = split(labels)
    cands = candidate_scores(products, prepared["hits"], taxonomy, vectorizer)

    def classify_with(tau, delta=0.0):
        return apply_rule(cands, products, taxonomy, {"tau": tau, "delta": delta}, vectorizer).set_index("product_key")

    def predict_on(subset):
        return lambda tau: classify_with(tau).loc[subset["product_key"], "type_id"].fillna("unclassified").values

    tau = tune_tau(predict_on(tune_set), tune_set["type_id"].values)
    tuned = classify_with(tau)
    on_tune = tuned.loc[tune_set["product_key"]]
    delta = tune_delta(on_tune["margin"].fillna(np.inf).values,
                       (on_tune["type_id"].values == tune_set["type_id"].values))
    pred_test = tuned.loc[test_set["product_key"], "type_id"].fillna("unclassified")
    return {"method": vectorizer.method, "tau": tau, "delta": delta, "problems": problems,
            "n_tune": len(tune_set), "report": report(pred_test, test_set["type_id"])}

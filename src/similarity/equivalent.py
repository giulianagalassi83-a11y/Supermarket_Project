"""Equivalent-level decider: is B a fair substitute for A (same or equivalent), brand allowed to differ?

The main decider is fit on BM↔Consum barcode pairs, so it only knows 'same product'; cross-brand equivalents
(own brand vs national brand) rarely reach its threshold. This one is fit on the Gemini silver pairs
(data/similarity/silver_candidates.csv + llm_pair_labels.csv): target = label in {same, equivalent}.
Its threshold is the precision rule P ≥ config.SIM['equiv_precision_target'] (.85) on 5-fold out-of-fold probabilities.

    python -m src.similarity.equivalent   # → results/equivalent_decider.csv (OOF vs Gemini, and vs Felipe's audit)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipeline import config
from src.similarity.match import add_features, choose_threshold, fit_decider, predict

FEATURES = ["cosine", "brand_match", "size_ratio", "size_known", "variant_agree", "pack_match"]
RESULTS_PATH = config.SIM_RESULTS_DIR / "equivalent_decider.csv"


def target(labels):
    """1 when a pair is a fair substitute (same or equivalent)."""
    return pd.Series(labels).isin(["same", "equivalent"]).astype(int).to_numpy()


def training_rows(attrs):
    """Silver pairs with a Gemini label and the attribute features; pairs whose products left the catalogue dropped."""
    silver = pd.read_csv(config.SIM_DIR / "silver_candidates.csv")
    llm = pd.read_csv(config.SIM_DIR / "llm_pair_labels.csv")[["key_a", "key_b", "label"]].dropna()
    d = silver.merge(llm, on=["key_a", "key_b"])
    d = d[d["key_a"].isin(attrs.keys()) & d["key_b"].isin(attrs.keys())].reset_index(drop=True)
    d = add_features(d, {}, attrs)
    d["y"] = target(d["label"])
    return d


def oof_probs(d, seed=config.SEED):
    oof = np.zeros(len(d))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(d, d["y"]):
        oof[test] = predict(fit_decider(d.iloc[train], d["y"].iloc[train], FEATURES), d.iloc[test])
    return oof


def fit(attrs):
    """(model fit on all silver rows, threshold from out-of-fold probabilities)."""
    d = training_rows(attrs)
    threshold = choose_threshold(oof_probs(d), d["y"], config.SIM["equiv_precision_target"])
    return fit_decider(d, d["y"], FEATURES), threshold


def _pr(probs, y, t):
    acc = probs >= t
    return {"accepted": int(acc.sum()), "precision": round(float(y[acc].mean()), 3) if acc.any() else None,
            "recall": round(float(y[acc].sum() / max(y.sum(), 1)), 3)}


def main():
    from src.similarity import attributes
    from src.similarity.data import load_products
    from src.similarity.silver import heldout_decider
    from src.similarity.data import barcode_pairs

    products = load_products()
    attrs = attributes.extract_all(products)
    d = training_rows(attrs)
    d["oof"] = oof_probs(d)
    t = choose_threshold(d["oof"], d["y"], config.SIM["equiv_precision_target"])
    _, same_t, _ = heldout_decider(products, barcode_pairs(products), attrs)

    rows = [{"truth": "gemini", "decider": "equivalent (OOF)", "threshold": round(t, 3), "n": len(d),
             **_pr(d["oof"].to_numpy(), d["y"].to_numpy(), t)},
            {"truth": "gemini", "decider": "same-product", "threshold": round(same_t, 3), "n": len(d),
             **_pr(d["prob"].to_numpy(), d["y"].to_numpy(), same_t)}]
    audit_path = config.SIM_DIR / "audit" / "audit_felipe.xlsx"
    if audit_path.exists():
        key = pd.read_csv(config.SIM_DIR / "audit" / "audit_key.csv")
        lab = pd.read_excel(audit_path)[["audit_id", "label"]].dropna()
        a = key.merge(lab, on="audit_id").merge(d[["key_a", "key_b", "oof"]], on=["key_a", "key_b"])
        y = target(a["label"])
        rows += [{"truth": "felipe audit", "decider": "equivalent (OOF)", "threshold": round(t, 3), "n": len(a),
                  **_pr(a["oof"].to_numpy(), y, t)},
                 {"truth": "felipe audit", "decider": "same-product", "threshold": round(same_t, 3), "n": len(a),
                  **_pr(a["prob"].to_numpy(), y, same_t)}]
    out = pd.DataFrame(rows)
    config.SIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_PATH, index=False)
    print(out.to_string(index=False))
    model = fit_decider(d, d["y"], FEATURES)
    print("coefficients:", dict(zip(FEATURES, np.round(model.coef_[0], 2))))


if __name__ == "__main__":
    main()

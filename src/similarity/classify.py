"""Type classification on the engine encoder (spec §6.1).

The existing decision rule (`pipeline.classify.apply_rule`) runs unchanged; only the vectors change.
τ/δ are tuned on the ~300 Gemini-labelled products and tested on Felipe's 97 hand labels (never tuned on).

    python -m src.similarity.classify   # compare TF-IDF / base / fine-tuned → results/classification.csv
"""

import sqlite3

import numpy as np
import pandas as pd

from src.pipeline import config
from src.pipeline.classify import TfidfFallback, apply_rule, candidate_scores, decide, load_taxonomy, product_text
from src.pipeline.evaluate import load_labels, macro_f1, report, tune_delta, tune_tau, validate_labels

RESULTS_PATH = config.SIM_RESULTS_DIR / "classification.csv"
CONFUSION_PATH = config.SIM_RESULTS_DIR / "confusion.png"
TAU_GRID = np.round(np.arange(0.05, 0.90, 0.01), 2)


class EngineVectorizer:
    """Adapter so the pipeline's decision rule runs on the engine encoder."""
    method = "engine"

    def __init__(self, encoder):
        self.encoder = encoder
        self.model_name = encoder.key

    def encode(self, texts):
        return self.encoder.encode(list(texts), "query")


def classify(products, hits, taxonomy, encoder, thresholds):
    """Same columns as pipeline.classify.decide. No re-ranker: dropped from the engine (Felipe, 15 Nov)."""
    return decide(products, hits, taxonomy, EngineVectorizer(encoder), thresholds)


def load_all_products(db_path=config.DB_PATH):
    """Every product in the DB (current and historical-only, the pipeline classifies both) + all search hits."""
    with sqlite3.connect(db_path) as con:
        products = pd.read_sql_query("SELECT * FROM products", con, dtype={"ean": str})
        hits = pd.read_sql_query("SELECT DISTINCT product_key, search_term FROM search_hits", con)
    products["text"] = [product_text(n, c) for n, c in zip(products["name_clean"], products["chain_category"])]
    return products, hits


def tune_thresholds(cands, products, taxonomy, vectorizer, labels):
    """τ maximising macro-F1 on `labels` (product_key, type_id), then δ separating wrong from right."""
    truth = labels.set_index("product_key")["type_id"]
    keys = list(truth.index)

    def predict(tau, delta=0.0):
        out = apply_rule(cands, products, taxonomy, {"tau": tau, "delta": delta}, vectorizer).set_index("product_key")
        return out["type_id"].reindex(keys).fillna("unclassified")

    tau = tune_tau(predict, truth.to_numpy(), TAU_GRID)
    out = apply_rule(cands, products, taxonomy, {"tau": tau, "delta": 0.0}, vectorizer).set_index("product_key").reindex(keys)
    margins = out["margin"].fillna(np.inf).to_numpy()
    correct = (out["type_id"].fillna("unclassified").to_numpy() == truth.to_numpy())
    return {"tau": tau, "delta": tune_delta(margins, correct)}


def _evaluate(name, vectorizer, products, hits, taxonomy, llm, hand):
    keys = pd.unique(pd.concat([llm["product_key"], hand["product_key"]]))
    sub = products[products["product_key"].isin(keys)].reset_index(drop=True)
    cands = candidate_scores(sub, hits, taxonomy, vectorizer)
    th = tune_thresholds(cands, sub, taxonomy, vectorizer, llm)
    pred = apply_rule(cands, sub, taxonomy, th, vectorizer).set_index("product_key")["type_id"]
    tune_pred = pred.reindex(llm["product_key"]).fillna("unclassified")
    tune_f1 = macro_f1(tune_pred, llm["type_id"])
    tune_acc = float((tune_pred.to_numpy() == llm["type_id"].to_numpy()).mean())
    rep = report(pred.reindex(hand["product_key"]).fillna("unclassified").to_numpy(), hand["type_id"].to_numpy())
    row = {"vectorizer": name, "model": vectorizer.model_name, "tau": th["tau"], "delta": th["delta"],
           "tune_n": len(llm), "tune_accuracy": round(tune_acc, 3), "tune_macro_f1": round(tune_f1, 3), "test_n": rep["n"],
           "test_accuracy": round(rep["accuracy"], 3),
           "test_accuracy_ci95": f"{rep['accuracy_ci95'][0]:.2f}–{rep['accuracy_ci95'][1]:.2f}",
           "test_macro_f1": round(rep["macro_f1"], 3),
           "relevant_precision": round(rep["relevant_precision"], 3), "relevant_recall": round(rep["relevant_recall"], 3)}
    return row, rep["confusion"]


def _save_confusion(confusion, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(0.35 * confusion.shape[1] + 4, 0.35 * confusion.shape[0] + 3))
    ax.imshow(confusion.to_numpy(), cmap="Blues")
    ax.set_xticks(range(confusion.shape[1]), confusion.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(confusion.shape[0]), confusion.index, fontsize=7)
    ax.set_xlabel("predicted"), ax.set_ylabel("hand label"), ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=130)
    plt.close(fig)


def main():
    import gc

    from src.similarity.encoder import load_encoder

    products, hits = load_all_products()
    taxonomy = load_taxonomy(config.TAXONOMY_PATH)
    hand, problems = validate_labels(load_labels(), products, taxonomy)
    for p in problems:
        print("hand label skipped:", p)
    llm = pd.read_csv(config.SIM_DIR / "llm_type_labels.csv", dtype=str).rename(columns={"label": "type_id"})
    llm = llm[llm["type_id"].notna()]
    llm, problems = validate_labels(llm.assign(confirmed=1), products, taxonomy)
    llm = llm[~llm["product_key"].isin(hand["product_key"])]  # never tune on a test product
    print(f"tune on {len(llm)} Gemini labels, test on {len(hand)} hand labels", flush=True)

    rows, confusions = [], {}
    tfidf = TfidfFallback(list(products["text"]) + list(taxonomy["description"]))
    row, confusions["tfidf"] = _evaluate("tfidf", tfidf, products, hits, taxonomy, llm, hand)
    rows.append(row)
    for which in ["base", "finetuned"]:
        enc = load_encoder(which)
        row, confusions[which] = _evaluate(which, EngineVectorizer(enc), products, hits, taxonomy, llm, hand)
        rows.append(row)
        enc.save_cache()
        del enc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    table = pd.DataFrame(rows)
    config.SIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)
    print(table.to_string(index=False))
    engine = table[table["vectorizer"] != "tfidf"]
    pick = engine.loc[engine["tune_macro_f1"].idxmax()]  # chosen on the tuning labels, never on the test set
    _save_confusion(confusions[pick["vectorizer"]], f"{pick['vectorizer']} — hand labels (n={pick['test_n']})")
    print(f"\npick (best tune macro-F1): {pick['vectorizer']} τ={pick['tau']} δ={pick['delta']}"
          f" → to use it: config.SIM['classifier'] and config.SIM['classifier_thresholds'], then python -m src.similarity.build")


if __name__ == "__main__":
    main()

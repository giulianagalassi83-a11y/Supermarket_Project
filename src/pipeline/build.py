"""Run the whole pipeline: raw CSVs -> data/supermarkets.db + data/products_normalized.csv.

Usage (from the project root):  python -m src.pipeline.build
"""

import sys

import numpy as np
import pandas as pd

from src.pipeline import adapters, config
from src.pipeline.classify import (
    TfidfFallback, decide, get_vectorizer, load_taxonomy, product_text, types_for_db,
)
from src.pipeline.db import export_csv, write_db
from src.pipeline.standardize import standardize

OUTLIER_MIN_GROUP = 5  # don't judge outliers in tiny groups
OUTLIER_IQR = 3


def _with_type(obs, classification, taxonomy):
    real = taxonomy[taxonomy["is_negative"] == 0][["type_id", "unit"]].rename(columns={"unit": "type_unit"})
    return obs.merge(classification[["product_key", "type_id"]], on="product_key", how="left") \
              .merge(real, on="type_id", how="left")


def add_outlier_flags(obs, classification, taxonomy):
    """flag_outlier: log €/unit outside Q1 − 3·IQR … Q3 + 3·IQR within the same type and unit."""
    df = _with_type(obs, classification, taxonomy)
    log_ppu = np.log(df["price_per_unit"].where(df["price_per_unit"] > 0))
    groups = log_ppu.groupby([df["type_id"], df["unit"]])
    q1, q3, size = groups.transform("quantile", 0.25), groups.transform("quantile", 0.75), groups.transform("count")
    iqr = q3 - q1
    outside = (log_ppu < q1 - OUTLIER_IQR * iqr) | (log_ppu > q3 + OUTLIER_IQR * iqr)
    out = obs.copy()
    out["flag_outlier"] = (outside & (size >= OUTLIER_MIN_GROUP) & df["type_unit"].notna()).values
    return out


def add_unit_mismatch(obs, classification, taxonomy):
    """flag_unit_mismatch: the €/unit is in a different unit than its type's (e.g. €/litro for a 'por lavado' type)."""
    df = _with_type(obs, classification, taxonomy)
    out = obs.copy()
    out["flag_unit_mismatch"] = (df["type_unit"].notna() & df["unit"].notna() & (df["unit"] != df["type_unit"])).values
    return out


def _chain_stats(chain_id, raw, products, observations, classification):
    keys = products.loc[products["chain_id"] == chain_id, "product_key"]
    obs = observations[observations["product_key"].isin(keys)]
    cls = classification[classification["product_key"].isin(keys)]
    classified = cls[cls["method"] != "unclassified"]
    pct = lambda s: round(100 * float(s.mean()), 1) if len(s) else 0.0
    return {
        "rows": int((raw["chain_id"] == chain_id).sum()),
        "products": int(len(keys)),
        "pct_ppu": pct(obs["price_per_unit"].notna()),
        "pct_no_relevante": pct(classified["type_id"] == config.NO_RELEVANTE),
        "pct_low_conf": pct(classified["low_confidence"]),
        "error": None,
    }


def prepare(raw_dir=config.RAW_DIR, taxonomy_path=config.TAXONOMY_PATH, cache_path=config.CACHE_PATH,
            force_tfidf=False):
    """Steps 1–3 plus the vectorizer: everything classification needs (also used by evaluate.py)."""
    report = {"chains": {}, "warnings": []}

    # 1. adapters: a chain that fails is skipped, the others still build
    frames = []
    for chain_id in config.CHAINS:
        try:
            frames.append(adapters.load_chain(chain_id, raw_dir))
        except Exception as e:  # noqa: BLE001 — any broken raw file must not stop the build
            report["chains"][chain_id] = {"rows": 0, "products": 0, "error": f"{type(e).__name__}: {e}"}
    raw = pd.concat(frames, ignore_index=True)

    # 2–3. standardize + first flags
    tables = standardize(raw)
    taxonomy = load_taxonomy(taxonomy_path)
    products = tables["products"]
    texts = [product_text(n, c) for n, c in zip(products["name_clean"], products["chain_category"])]
    texts += list(taxonomy["description"])
    if force_tfidf:
        vectorizer = TfidfFallback(texts)
    else:
        vectorizer = get_vectorizer(texts, cache_path, config.MODEL_NAME)
    report["method"], report["model_name"] = vectorizer.method, vectorizer.model_name
    if vectorizer.method == "tfidf" and not force_tfidf:
        report["warnings"].insert(0, (
            f"WARNING: embeddings unavailable for {vectorizer.missing} texts -> whole run used TF-IDF "
            "(install requirements-embeddings.txt and rebuild)"
        ))
    return {**tables, "raw": raw, "taxonomy": taxonomy, "vectorizer": vectorizer, "report": report}


def build(raw_dir=config.RAW_DIR, db_path=config.DB_PATH, csv_path=config.CSV_PATH,
          taxonomy_path=config.TAXONOMY_PATH, cache_path=config.CACHE_PATH):
    p = prepare(raw_dir, taxonomy_path, cache_path)
    raw, products, observations, hits = p["raw"], p["products"], p["observations"], p["hits"]
    taxonomy, vectorizer, report = p["taxonomy"], p["vectorizer"], p["report"]

    # 4. classify
    classification = decide(products, hits, taxonomy, vectorizer, config.THRESHOLDS[vectorizer.method])
    observations = add_outlier_flags(observations, classification, taxonomy)
    observations = add_unit_mismatch(observations, classification, taxonomy)

    # 5. store
    write_db(db_path, {
        "products": products, "observations": observations, "hits": hits,
        "product_types": types_for_db(taxonomy), "classification": classification,
    })
    export_csv(db_path, csv_path)

    for chain_id in config.CHAINS:
        if chain_id not in report["chains"]:
            report["chains"][chain_id] = _chain_stats(chain_id, raw, products, observations, classification)
    return report


def format_report(report):
    lines = list(report["warnings"])
    lines.append(f"Classifier: {report['method']} ({report['model_name']})")
    lines.append(f"{'chain':<10} {'rows':>6} {'products':>9} {'%€/unit':>8} {'%no_rel':>8} {'%low_conf':>9}  error")
    for chain_id, s in report["chains"].items():
        if s["error"]:
            lines.append(f"{chain_id:<10} {'-':>6} {'-':>9} {'-':>8} {'-':>8} {'-':>9}  {s['error']}")
        else:
            lines.append(f"{chain_id:<10} {s['rows']:>6} {s['products']:>9} {s['pct_ppu']:>8} "
                         f"{s['pct_no_relevante']:>8} {s['pct_low_conf']:>9}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # the report has accents; redirected output on Windows is cp1252
    print(format_report(build()))
    print(f"\nWrote {config.DB_PATH.relative_to(config.ROOT)} and {config.CSV_PATH.relative_to(config.ROOT)}")

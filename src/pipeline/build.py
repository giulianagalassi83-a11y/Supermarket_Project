
"""Build the current supermarket catalog and normalized price history.

Inputs:
    data/raw/*.csv
    data/history/<chain>/YYYY-MM-DD_raw.csv

Outputs:
    data/supermarkets.db
    data/products_normalized.csv
    data/price_history_normalized.csv

Usage:
    python -m src.pipeline.build
"""

import re
import sys

import numpy as np
import pandas as pd

from src.pipeline import adapters, config
from src.pipeline.adapters import apis, scraped
from src.pipeline.classify import (
    TfidfFallback,
    decide,
    get_vectorizer,
    load_taxonomy,
    product_text,
    types_for_db,
)
from src.pipeline.db import (
    export_csv,
    export_history_csv,
    write_db,
)
from src.pipeline.standardize import standardize


OUTLIER_MIN_GROUP = 5
OUTLIER_IQR = 3

HISTORY_DIR = config.DATA_DIR / "history"
HISTORY_CSV_PATH = (
    config.DATA_DIR / "price_history_normalized.csv"
)

# Use the same adapters for current and historical RAW data.
HISTORY_LOADERS = {
    "mercadona": apis.load_mercadona_api,
    "bm": apis.load_bm,
    "consum": apis.load_consum,
    "ahorramas": scraped.load_ahorramas,
    "aldi": scraped.load_aldi,
    "alcampo": scraped.load_alcampo,
    "dia": scraped.load_dia,
}


# --------------------------------------------------
# QUALITY FLAGS
# --------------------------------------------------

def _with_type(obs, classification, taxonomy):
    real = (
        taxonomy[taxonomy["is_negative"] == 0]
        [["type_id", "unit"]]
        .rename(columns={"unit": "type_unit"})
    )

    return (
        obs.merge(
            classification[["product_key", "type_id"]],
            on="product_key",
            how="left",
            validate="many_to_one",
        )
        .merge(
            real,
            on="type_id",
            how="left",
            validate="many_to_one",
        )
    )


def add_outlier_flags(obs, classification, taxonomy):
    """Flag extreme log-reference prices within product types."""
    df = _with_type(
        obs,
        classification,
        taxonomy,
    )

    # Use current observations as the reference distribution.
    reference = df["is_current"].fillna(False).astype(bool)

    log_ppu = np.log(
        df["price_per_unit"].where(
            df["price_per_unit"] > 0
        )
    )

    reference_ppu = log_ppu.where(reference)

    group_keys = [
        df["type_id"],
        df["unit"],
    ]

    reference_groups = reference_ppu.groupby(
        group_keys
    )

    q1 = reference_groups.transform(
        "quantile", 0.25
    )

    q3 = reference_groups.transform(
        "quantile", 0.75
    )

    size = reference_groups.transform("count")

    # Compute thresholds by type and unit, then
    # apply them to current and historical observations.
    bounds = pd.DataFrame({
        "type_id": df["type_id"],
        "unit": df["unit"],
        "q1": q1,
        "q3": q3,
        "size": size,
    })

    bounds = (
        bounds.dropna(
            subset=["type_id", "unit"]
        )
        .drop_duplicates(
            ["type_id", "unit"]
        )
    )

    checks = df[["type_id", "unit"]].merge(
        bounds,
        on=["type_id", "unit"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    iqr = checks["q3"] - checks["q1"]

    outside = (
        (
            log_ppu.reset_index(drop=True)
            < checks["q1"] - OUTLIER_IQR * iqr
        )
        |
        (
            log_ppu.reset_index(drop=True)
            > checks["q3"] + OUTLIER_IQR * iqr
        )
    )

    out = obs.copy().reset_index(drop=True)

    out["flag_outlier"] = (
        outside
        & (checks["size"] >= OUTLIER_MIN_GROUP)
        & df["type_unit"].reset_index(drop=True).notna()
    ).fillna(False).to_numpy()

    return out


def add_unit_mismatch(obs, classification, taxonomy):
    """Flag observations whose reference unit differs from the type."""
    df = _with_type(
        obs,
        classification,
        taxonomy,
    )

    out = obs.copy().reset_index(drop=True)

    out["flag_unit_mismatch"] = (
        df["type_unit"].notna()
        & df["unit"].notna()
        & (df["unit"] != df["type_unit"])
    ).to_numpy()

    return out


# --------------------------------------------------
# HISTORICAL RAW COLLECTIONS
# --------------------------------------------------

def load_historical_raw(history_dir=HISTORY_DIR):
    """Load historical RAW snapshots with their original adapters."""
    frames = []
    counts = {}

    for chain_id in config.CHAINS:
        folder = history_dir / chain_id
        loader = HISTORY_LOADERS.get(chain_id)

        if loader is None:
            raise KeyError(
                f"No historical adapter configured for {chain_id}."
            )

        counts[chain_id] = 0

        if not folder.exists():
            continue

        snapshots = sorted(
            path
            for path in folder.glob("*_raw.csv")
            if re.fullmatch(
                r"\d{4}-\d{2}-\d{2}_raw\.csv",
                path.name,
            )
        )

        for path in snapshots:
            # The same adapter processes current and historical files.
            frame = loader(path)

            if frame.empty:
                raise ValueError(
                    f"Historical snapshot is empty: {path}"
                )

            frame = frame.copy()

            frame["is_current"] = False
            frame["_snapshot_date"] = path.name[:10]

            frames.append(frame)
            counts[chain_id] += 1

    if frames:
        historical = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        historical = pd.DataFrame()

    return historical, counts


# --------------------------------------------------
# CURRENT DATA AND CLASSIFICATION PREPARATION
# --------------------------------------------------

def prepare(
    raw_dir=config.RAW_DIR,
    taxonomy_path=config.TAXONOMY_PATH,
    cache_path=config.CACHE_PATH,
    force_tfidf=False,
    history_raw=None,
    strict=False,
):
    """Load sources, standardize products, and initialize classification."""
    report = {
        "chains": {},
        "warnings": [],
    }

    frames = []

    # Load the latest complete RAW file for each supermarket.
    for chain_id in config.CHAINS:
        try:
            frame = adapters.load_chain(
                chain_id,
                raw_dir,
            )

            if frame.empty:
                raise ValueError(
                    "The current RAW file is empty."
                )

            frame = frame.copy()
            frame["is_current"] = True

            frames.append(frame)

        except Exception as exc:
            report["chains"][chain_id] = {
                "rows": 0,
                "products": 0,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

            if strict:
                raise RuntimeError(
                    f"Current data for {chain_id} "
                    f"could not be loaded: {exc}"
                ) from exc

    if not frames:
        raise ValueError(
            "No current supermarket RAW files could be loaded."
        )

    current_raw = pd.concat(
        frames,
        ignore_index=True,
    )

    # Append historical RAW data for the full build.
    if history_raw is not None and not history_raw.empty:
        raw = pd.concat(
            [current_raw, history_raw],
            ignore_index=True,
        )
    else:
        raw = current_raw.copy()

    # Reuse exactly the same standardization logic.
    tables = standardize(
        raw,
        preserve_observations=history_raw is not None,
    )

    taxonomy = load_taxonomy(
        taxonomy_path
    )

    products = tables["products"]

    texts = [
        product_text(name, category)
        for name, category in zip(
            products["name_clean"],
            products["chain_category"],
        )
    ]

    texts += list(
        taxonomy["description"]
    )

    if force_tfidf:
        vectorizer = TfidfFallback(texts)
    else:
        vectorizer = get_vectorizer(
            texts,
            cache_path,
            config.MODEL_NAME,
        )

    report["method"] = vectorizer.method
    report["model_name"] = vectorizer.model_name

    if (
        vectorizer.method == "tfidf"
        and not force_tfidf
    ):
        report["warnings"].insert(
            0,
            (
                "WARNING: embeddings unavailable for "
                f"{vectorizer.missing} texts -> "
                "whole run used TF-IDF "
                "(install requirements-embeddings.txt "
                "and rebuild)"
            ),
        )

    return {
        **tables,
        "raw": raw,
        "current_raw": current_raw,
        "taxonomy": taxonomy,
        "vectorizer": vectorizer,
        "report": report,
    }


# --------------------------------------------------
# REPORTING
# --------------------------------------------------

def _chain_stats(
    chain_id,
    current_raw,
    products,
    observations,
    classification,
):
    current_keys = observations.loc[
        observations["is_current"].fillna(False).astype(bool),
        "product_key",
    ].drop_duplicates()

    chain_prefix = f"{chain_id}:"

    keys = current_keys[
        current_keys.str.startswith(
            chain_prefix,
            na=False,
        )
    ]

    obs = observations[
        observations["product_key"].isin(keys)
        & observations["is_current"].fillna(False).astype(bool)
    ]

    cls = classification[
        classification["product_key"].isin(keys)
    ]

    classified = cls[
        cls["method"] != "unclassified"
    ]

    def pct(series):
        return (
            round(100 * float(series.mean()), 1)
            if len(series) else 0.0
        )

    return {
        "rows": int(
            (current_raw["chain_id"] == chain_id).sum()
        ),
        "products": int(len(keys)),
        "pct_ppu": pct(
            obs["price_per_unit"].notna()
        ),
        "pct_no_relevante": pct(
            classified["type_id"] == config.NO_RELEVANTE
        ),
        "pct_low_conf": pct(
            classified["low_confidence"]
        ),
        "error": None,
    }


# --------------------------------------------------
# FULL BUILD
# --------------------------------------------------

def build(
    raw_dir=config.RAW_DIR,
    db_path=config.DB_PATH,
    csv_path=config.CSV_PATH,
    taxonomy_path=config.TAXONOMY_PATH,
    cache_path=config.CACHE_PATH,
    history_dir=HISTORY_DIR,
    history_csv_path=HISTORY_CSV_PATH,
):
    """Build the current catalog and normalized historical observations."""

    # 1. Load RAW historical snapshots.
    history_raw, history_counts = load_historical_raw(
        history_dir
    )

    # 2. Load current RAW and standardize everything.
    # strict=True prevents publishing a partial current catalog.
    p = prepare(
        raw_dir=raw_dir,
        taxonomy_path=taxonomy_path,
        cache_path=cache_path,
        history_raw=history_raw,
        strict=True,
    )

    raw = p["raw"]
    current_raw = p["current_raw"]

    products = p["products"]
    observations = p["observations"]
    hits = p["hits"]

    taxonomy = p["taxonomy"]
    vectorizer = p["vectorizer"]
    report = p["report"]

    # 3. Classify product identities, not individual dates.
    classification = decide(
        products,
        hits,
        taxonomy,
        vectorizer,
        config.THRESHOLDS[vectorizer.method],
    )

    # 4. Apply quality flags to normalized observations.
    observations = add_outlier_flags(
        observations,
        classification,
        taxonomy,
    )

    observations = add_unit_mismatch(
        observations,
        classification,
        taxonomy,
    )

    # 5. Rebuild SQLite safely.
    # The original database remains intact if writing fails.
    write_db(
        db_path,
        {
            "products": products,
            "observations": observations,
            "hits": hits,
            "product_types": types_for_db(taxonomy),
            "classification": classification,
        },
    )

    # 6. Export the current catalog and normalized history.
    export_csv(
        db_path,
        csv_path,
    )

    export_history_csv(
        db_path,
        history_csv_path,
    )

    # 7. Report current collection coverage.
    for chain_id in config.CHAINS:
        if chain_id not in report["chains"]:
            report["chains"][chain_id] = _chain_stats(
                chain_id,
                current_raw,
                products,
                observations,
                classification,
            )

    report["history_snapshots"] = history_counts
    report["historical_observations"] = int(
        (~observations["is_current"].astype(bool)).sum()
    )
    report["total_observations"] = len(observations)

    return report


def format_report(report):
    """Format the pipeline execution report."""
    lines = list(report["warnings"])

    lines.append(
        f"Classifier: {report['method']} "
        f"({report['model_name']})"
    )

    lines.append(
        f"{'chain':<10} "
        f"{'rows':>6} "
        f"{'products':>9} "
        f"{'%€/unit':>8} "
        f"{'%no_rel':>8} "
        f"{'%low_conf':>9}  error"
    )

    for chain_id, stats in report["chains"].items():
        if stats["error"]:
            lines.append(
                f"{chain_id:<10} "
                f"{'-':>6} "
                f"{'-':>9} "
                f"{'-':>8} "
                f"{'-':>8} "
                f"{'-':>9}  "
                f"{stats['error']}"
            )
        else:
            lines.append(
                f"{chain_id:<10} "
                f"{stats['rows']:>6} "
                f"{stats['products']:>9} "
                f"{stats['pct_ppu']:>8} "
                f"{stats['pct_no_relevante']:>8} "
                f"{stats['pct_low_conf']:>9}"
            )

    if "history_snapshots" in report:
        lines.append("\nHistorical RAW snapshots:")

        for chain, count in report["history_snapshots"].items():
            lines.append(
                f"  {chain}: {count}"
            )

        lines.append(
            "\nNormalized observations: "
            f"{report['total_observations']}"
        )

        lines.append(
            "Historical-only observations: "
            f"{report['historical_observations']}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    result = build()

    print(
        format_report(result)
    )

    print(
        f"\nWrote {config.DB_PATH.relative_to(config.ROOT)}"
    )

    print(
        f"Wrote {config.CSV_PATH.relative_to(config.ROOT)}"
    )

    print(
        f"Wrote {HISTORY_CSV_PATH.relative_to(config.ROOT)}"
    )

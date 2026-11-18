
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
    _hash,
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
# SAVED CLASSIFICATION (written by src.similarity.build)
# --------------------------------------------------

CLASSIFICATION_COLUMNS = [
    "product_key", "type_id", "score", "runner_up_type_id",
    "margin", "low_confidence", "method", "model_name",
]
PENDING = "pending_similarity"


def load_assignments(path=config.SIM_ASSIGNMENTS_PATH):
    """The saved engine classification, or None when the similarity build has never run."""
    if not path.exists():
        return None
    return pd.read_csv(
        path,
        dtype={"product_key": str, "text_hash": str, "type_id": str, "runner_up_type_id": str},
    )


def apply_saved_assignments(products, assignments):
    """Classification from the saved engine run; never computes embeddings (no torch here, CI-safe).

    A product keeps its saved row only when its key AND its text hash match, so a renamed product is
    treated as new: type_id None, method 'pending_similarity', low_confidence True, until
    `python -m src.similarity.build` runs again. CI can therefore never downgrade a classification.
    """
    texts = [
        product_text(name, category)
        for name, category in zip(products["name_clean"], products["chain_category"])
    ]
    out = pd.DataFrame({
        "product_key": products["product_key"].to_numpy(),
        "text_hash": [_hash(t) for t in texts],
    })
    if assignments is not None and len(assignments):
        saved = assignments.drop_duplicates("product_key", keep="last")
        out = out.merge(
            saved[["text_hash"] + CLASSIFICATION_COLUMNS],
            on=["product_key", "text_hash"],
            how="left",
        )
    else:
        for col in CLASSIFICATION_COLUMNS[1:]:
            out[col] = None
    out = out[CLASSIFICATION_COLUMNS].astype(
        {c: object for c in ["type_id", "runner_up_type_id", "low_confidence", "method", "model_name"]}
    )
    out[["score", "margin"]] = out[["score", "margin"]].astype(float)
    pending = out["method"].isna()
    out.loc[pending, ["type_id", "runner_up_type_id", "method", "model_name"]] = [None, None, PENDING, None]
    out.loc[pending, ["score", "margin"]] = np.nan
    saved_low = out["low_confidence"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    out["low_confidence"] = (pending | saved_low).astype(bool)
    for col in ["type_id", "runner_up_type_id", "model_name"]:
        out[col] = out[col].where(out[col].notna(), None)
    return out


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
    """Load sources and standardize products.

    The production classification comes from saved assignments (see `apply_saved_assignments`);
    `force_tfidf` only builds a TF-IDF vectorizer for evaluation callers.
    """
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

    vectorizer = None
    if force_tfidf:
        texts = [
            product_text(name, category)
            for name, category in zip(
                products["name_clean"],
                products["chain_category"],
            )
        ]
        vectorizer = TfidfFallback(texts + list(taxonomy["description"]))

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
        ~cls["method"].isin(["unclassified", PENDING])
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
    assignments_path=config.SIM_ASSIGNMENTS_PATH,
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
    report = p["report"]

    # 3. Classify product identities from the saved similarity build (no ML here).
    classification = apply_saved_assignments(
        products,
        load_assignments(assignments_path),
    )
    saved = classification[classification["method"] != PENDING]
    report["method"] = "engine (saved assignments)"
    report["model_name"] = (
        saved["model_name"].dropna().mode().iloc[0]
        if saved["model_name"].notna().any() else None
    )
    report["pending"] = int((classification["method"] == PENDING).sum())

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

    if "pending" in report:
        lines.append(
            f"{report['pending']} products pending similarity build"
            + (" (run: python -m src.similarity.build)" if report["pending"] else "")
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

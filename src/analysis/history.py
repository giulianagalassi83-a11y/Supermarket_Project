
"""Historical price analysis using normalized pipeline data.

This module reads the normalized historical CSV produced by the
central pipeline. It does not read RAW files or repeat preprocessing.

For each supermarket, it compares the two latest available
collection dates using stable product identifiers.

Products with changed or unverified pack formats are excluded
from price-change comparisons.

Usage:
    python -m src.analysis.history
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import prices
from src.pipeline import config


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

HISTORY_PATH = config.DATA_DIR / "price_history_normalized.csv"
CHANGES_PATH = config.DATA_DIR / "price_changes.csv"

HOLD_BACK = 1.5
MIN_CATEGORY_PRODUCTS = 10
UNCHANGED = 0.001

CHANGE_COLUMNS = [
    "supermarket",
    "product_key",
    "name",
    "old_price",
    "new_price",
    "old_date",
    "new_date",
    "searched_item",
    "product_type_label",
    "comparable",
    "change_pct",
    "held_back",
]

SUMMARY_COLUMNS = [
    "chain",
    "old_date",
    "new_date",
    "products",
    "up_pct",
    "down_pct",
    "average_pct",
    "changed_pct",
    "same_pct",
]

CATEGORY_COLUMNS = [
    "category",
    "chain",
    "products",
    "average_pct",
]

REQUIRED_HISTORY_COLUMNS = {
    "chain_id",
    "supermarket",
    "product_key",
    "name",
    "collection_date",
    "is_current",
    "price",
    "regular_price",
    "offer_price",
    "pack_count",
    "unit_quantity",
    "total_quantity",
    "quantity_unit",
}


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_history(path=HISTORY_PATH):
    """Load and validate normalized historical observations."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized history not found: {path}. "
            "Run the central pipeline first."
        )

    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={
            "product_key": "string",
            "chain_id": "string",
            "collection_date": "string",
        },
        low_memory=False,
    )

    missing = REQUIRED_HISTORY_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Normalized history is missing required columns: "
            f"{sorted(missing)}"
        )

    # Normalize dates without changing the source file.
    dates = pd.to_datetime(
        df["collection_date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "Normalized history contains missing or invalid "
            "collection dates."
        )

    df["collection_date"] = dates.dt.strftime("%Y-%m-%d")

    # Prices in this file should already be numeric.
    for column in (
        "price",
        "regular_price",
        "offer_price",
        "pack_count",
        "unit_quantity",
        "total_quantity",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def _true(values):
    """Interpret boolean flags read from CSV files."""
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )



def _quality_ok(df, suffix=""):
    """Check price and quality flags for one collection date."""
    price_column = f"price{suffix}"

    valid = (
        df[price_column].notna()
        & (df[price_column] > 0)
    )

    for column in (
        "flag_missing_ppu",
        "flag_outlier",
        "flag_unavailable",
        "flag_unit_mismatch",
    ):
        column_name = f"{column}{suffix}"

        if column_name in df.columns:
            valid &= ~_true(df[column_name])

    return valid.fillna(False)



def _valid_pack(old, new):
    """Check that product format is consistent across dates."""
    result = pd.Series(True, index=old.index)

    numeric_columns = (
        "pack_count",
        "unit_quantity",
        "total_quantity",
    )

    # Compare all available numerical pack attributes.
    for column in numeric_columns:
        a = pd.to_numeric(old[column], errors="coerce")
        b = pd.to_numeric(new[column], errors="coerce")

        both_known = a.notna() & b.notna()

        same = pd.Series(
            np.isclose(
                a.fillna(0),
                b.fillna(0),
                rtol=1e-6,
                atol=1e-6,
            ),
            index=old.index,
        )

        result &= ~both_known | same

        # A size present on only one date cannot be verified
        # unless the product description is unchanged.
        one_missing = a.isna() ^ b.isna()

        same_name = (
            old["name"].astype("string")
            .str.lower().str.strip()
            ==
            new["name"].astype("string")
            .str.lower().str.strip()
        ).fillna(False)

        result &= ~one_missing | same_name

    old_unit = old["quantity_unit"].astype("string")
    new_unit = new["quantity_unit"].astype("string")

    known_units = old_unit.notna() & new_unit.notna()

    result &= (
        ~known_units
        | old_unit.eq(new_unit).fillna(False)
    )

    # When pack measurements are missing, require matching names.
    missing_size = (
        old["total_quantity"].isna()
        | new["total_quantity"].isna()
    )

    same_name = (
        old["name"].astype("string")
        .str.lower().str.strip()
        ==
        new["name"].astype("string")
        .str.lower().str.strip()
    ).fillna(False)

    result &= ~missing_size | same_name

    return result.fillna(False)


def available_dates(history):
    """Return available collection dates for every supermarket."""
    return (
        history.groupby("chain_id")["collection_date"]
        .agg(lambda dates: sorted(dates.dropna().unique()))
        .to_dict()
    )


# --------------------------------------------------
# PRICE COMPARISONS
# --------------------------------------------------

def _compare_chain(chain, history):
    """Compare products from the two latest dates of one chain."""
    rows = history[
        history["chain_id"] == chain
    ].copy()

    dates = sorted(
        rows["collection_date"].dropna().unique()
    )

    if len(dates) < 2:
        print(
            f"[SKIP] {chain}: fewer than two collection dates."
        )
        return pd.DataFrame()

    old_date, new_date = dates[-2:]

    old = rows[
        rows["collection_date"] == old_date
    ].copy()

    new = rows[
        rows["collection_date"] == new_date
    ].copy()

    # Multiple locations or conflicting records can produce
    # duplicate product keys. Exclude these rather than selecting
    # an arbitrary price.
    old_duplicates = old["product_key"].duplicated(
        keep=False
    )

    new_duplicates = new["product_key"].duplicated(
        keep=False
    )

    old = old[
        ~old_duplicates
    ].copy()

    new = new[
        ~new_duplicates
    ].copy()

    both = old.merge(
        new,
        on="product_key",
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )

    if both.empty:
        print(
            f"[SKIP] {chain}: no matching products "
            f"between {old_date} and {new_date}."
        )
        return pd.DataFrame()

    # Only compare strictly positive selling prices.
    both = both[
        (both["price_old"] > 0)
        & (both["price_new"] > 0)
    ].copy()

    if both.empty:
        return pd.DataFrame()

    # Check that the product size has not changed.
    format_ok = _valid_pack(
        both.rename(
            columns={
                "pack_count_old": "pack_count",
                "unit_quantity_old": "unit_quantity",
                "total_quantity_old": "total_quantity",
                "quantity_unit_old": "quantity_unit",
                "name_old": "name",
            }
        ),
        both.rename(
            columns={
                "pack_count_new": "pack_count",
                "unit_quantity_new": "unit_quantity",
                "total_quantity_new": "total_quantity",
                "quantity_unit_new": "quantity_unit",
                "name_new": "name",
            }
        ),
    )

    rejected = int((~format_ok).sum())

    both = both[
        format_ok
    ].copy()

    if both.empty:
        print(
            f"[SKIP] {chain}: no products with "
            "consistent pack formats."
        )
        return pd.DataFrame()


    # Validate quality flags independently for both collection dates
    old_quality = _quality_ok(both, suffix="_old")
    new_quality = _quality_ok(both, suffix="_new")

    both["history_quality_ok"] = (
        old_quality & new_quality
    )


    result = pd.DataFrame({
        "supermarket": both["supermarket_new"].values,
        "product_key": both["product_key"].values,
        "name": both["name_new"].values,
        "old_price": both["price_old"].values,
        "new_price": both["price_new"].values,
        "old_date": old_date,
        "new_date": new_date,
        "history_quality_ok": both[
            "history_quality_ok"
        ].values,
    })

    print(
        f"[COMPARE] {chain}: {old_date} -> {new_date} | "
        f"{len(result)} matched products | "
        f"{rejected} format changes excluded"
    )

    return result


def build_changes(
    path=CHANGES_PATH,
    history_path=HISTORY_PATH,
    catalog_path=None,
):
    """Build price changes from normalized historical observations.

    The catalog is used to apply the same product usability rules
    as the basket optimizer.

    All input and output paths can be overridden for testing.
    """
    path = Path(path)
    history = load_history(history_path)

    if catalog_path is None:
        catalog_path = config.CSV_PATH

    catalog_path = Path(catalog_path)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Normalized current catalog not found: {catalog_path}"
        )

    catalog = pd.read_csv(
        catalog_path,
        encoding="utf-8",
        dtype={"product_key": "string"},
    )

    # Use the optimizer's existing comparability rules.
    usable_keys = set(
        prices.usable(catalog)["product_key"]
        .astype("string")
    )

    known = (
        catalog[
            [
                "product_key",
                "searched_item",
                "product_type_label",
            ]
        ]
        .drop_duplicates("product_key")
    )

    comparisons = []

    for chain in sorted(
        history["chain_id"].dropna().unique()
    ):
        comparison = _compare_chain(
            chain,
            history,
        )

        if not comparison.empty:
            comparisons.append(comparison)

    if not comparisons:
        result = pd.DataFrame(
            columns=CHANGE_COLUMNS
        )

    else:
        result = pd.concat(
            comparisons,
            ignore_index=True,
        )

        # Attach the existing normalized classification.
        result = result.merge(
            known,
            on="product_key",
            how="left",
            validate="many_to_one",
        )

        # A comparable product must pass the current catalog
        # filters and quality checks on both historical dates.
        result["comparable"] = (
            result["product_key"].isin(usable_keys)
            & result["history_quality_ok"]
        )

        ratio = (
            result["new_price"]
            / result["old_price"]
        )

        result["change_pct"] = (
            100 * (ratio - 1)
        )

        # Large changes are kept for review but excluded
        # from rankings that apply the hold-back filter.
        result["held_back"] = (
            (ratio > HOLD_BACK)
            | (ratio < 1 / HOLD_BACK)
        )

        result = result[CHANGE_COLUMNS]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"[SAVED] {len(result)} comparisons -> {path}"
    )

    return result


def load_changes(path=CHANGES_PATH):
    """Load previously generated price changes."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. "
            "Run: python -m src.analysis.history"
        )

    return pd.read_csv(
        path,
        encoding="utf-8",
    )


# --------------------------------------------------
# TREND METRICS
# --------------------------------------------------

def average_change(changes):
    """Calculate the geometric mean price change in percent."""
    if changes.empty:
        return np.nan

    ratio = (
        1 + changes["change_pct"] / 100
    )

    ratio = ratio[
        ratio > 0
    ].dropna()

    if ratio.empty:
        return np.nan

    return 100 * (
        np.exp(np.log(ratio).mean()) - 1
    )


def chain_summary(changes):
    """Summarize changes for each supermarket."""
    if changes.empty:
        return pd.DataFrame(
            columns=SUMMARY_COLUMNS
        )

    rows = []

    for chain, group in changes.groupby("supermarket"):
        ratio = (
            1 + group["change_pct"] / 100
        )

        rows.append({
            "chain": chain,
            "old_date": group["old_date"].iloc[0],
            "new_date": group["new_date"].iloc[0],
            "products": len(group),
            "up_pct": 100 * (
                ratio > 1 + UNCHANGED
            ).mean(),
            "down_pct": 100 * (
                ratio < 1 - UNCHANGED
            ).mean(),
            "average_pct": average_change(group),
        })

    summary = pd.DataFrame(rows)

    summary["changed_pct"] = (
        summary["up_pct"]
        + summary["down_pct"]
    )

    summary["same_pct"] = (
        100 - summary["changed_pct"]
    )

    return (
        summary.sort_values(
            "changed_pct",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def category_changes(
    changes,
    min_products=MIN_CATEGORY_PRODUCTS,
):
    """Calculate average category changes for comparable products."""
    if changes.empty:
        return pd.DataFrame(
            columns=CATEGORY_COLUMNS
        )

    rows = changes[
        _true(changes["comparable"])
        & ~_true(changes["held_back"])
        & changes["searched_item"].notna()
    ]

    out = [
        {
            "category": category,
            "chain": chain,
            "products": len(group),
            "average_pct": average_change(group),
        }
        for (category, chain), group in rows.groupby(
            ["searched_item", "supermarket"]
        )
        if len(group) >= min_products
    ]

    return pd.DataFrame(
        out,
        columns=CATEGORY_COLUMNS,
    )


def biggest_moves(changes, n=8):
    """Return the largest price increases and decreases."""
    if changes.empty:
        return changes.copy(), changes.copy()

    rows = changes[
        ~_true(changes["held_back"])
    ]

    return (
        rows.nlargest(n, "change_pct"),
        rows.nsmallest(n, "change_pct"),
    )


# --------------------------------------------------
# CURRENT PROMOTIONS
# --------------------------------------------------

def offers(history_path=HISTORY_PATH):
    """Summarize current API promotions from normalized observations."""
    history = load_history(history_path)

    current = history[
        _true(history["is_current"])
    ].copy()

    # Avoid counting the same product multiple times
    # when it appears in several searches.
    current = (
        current.sort_values(
            "collection_date",
            ascending=False,
        )
        .drop_duplicates(
            ["chain_id", "product_key"]
        )
    )

    rows = []

    for chain in ("bm", "consum"):
        raw = current[
            current["chain_id"] == chain
        ].copy()

        regular = raw["regular_price"]
        offer = raw["offer_price"]

        deal = (
            (regular > 0)
            & (offer > 0)
            & (offer < regular)
        )

        discounts = (
            100 * (
                1 - offer[deal] / regular[deal]
            )
        )

        rows.append({
            "chain": config.CHAINS[chain]["name"],
            "products": len(raw),
            "on_offer": int(deal.sum()),
            "median_discount_pct": discounts.median(),
        })

    table = pd.DataFrame(rows)

    table["on_offer_pct"] = (
        100
        * table["on_offer"]
        / table["products"].replace(0, np.nan)
    )

    # Mercadona's previous selling price is retained
    # as regular_price by the API adapter when higher.
    mercadona = current[
        current["chain_id"] == "mercadona"
    ]

    previous = mercadona["regular_price"]
    selling = mercadona["price"]

    reduced = (
        (previous > 0)
        & (selling > 0)
        & (previous > selling)
    )

    discounts = (
        100 * (
            1 - selling[reduced] / previous[reduced]
        )
    )

    former = {
        "products": int(reduced.sum()),
        "median_drop_pct": discounts.median(),
    }

    return table, former


# --------------------------------------------------
# COMMAND-LINE EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    print("\nAvailable historical dates:")

    historical = load_history()

    for chain, dates in available_dates(historical).items():
        print(
            f"  {chain}: {', '.join(dates)}"
        )

    print("\nBuilding price comparisons...\n")

    changes = build_changes()
    summary = chain_summary(changes)

    if summary.empty:
        print(
            "\nNo supermarkets have two comparable "
            "collection dates yet."
        )
    else:
        print(
            "\nPrice change summary:\n"
        )

        print(
            summary.round(2).to_string(index=False)
        )

    print(
        f"\nOutput: {CHANGES_PATH.relative_to(config.ROOT)}"
    )

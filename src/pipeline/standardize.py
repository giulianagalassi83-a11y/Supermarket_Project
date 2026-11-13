
"""Standardize adapter rows into products, price observations, and search hits.

The same transformations are used for current and historical RAW data.

Products are identified by chain and native product ID.
Observations are identified by product, collection date, and location.

Missing reference prices are flagged rather than discarded.
"""

import numpy as np
import pandas as pd

from src.pipeline.adapters.common import hash_id
from src.pipeline.text import (
    PACK_MULTI,
    clean_name,
    extract_brand,
    is_private_label,
    parse_pack,
    strip_accents,
)
from src.pipeline.units import per_base


PRODUCT_COLUMNS = [
    "product_key",
    "chain_id",
    "native_id",
    "ean",
    "name",
    "name_clean",
    "brand",
    "is_private_label",
    "chain_category",
    "pack_count",
    "unit_quantity",
    "total_quantity",
    "quantity_unit",
    "url",
    "image_url",
]

OBSERVATION_COLUMNS = [
    "product_key",
    "scraped_at",
    "collection_date",
    "is_current",
    "price",
    "regular_price",
    "offer_price",
    "price_per_unit",
    "unit",
    "ppu_source",
    "pack_count",
    "unit_quantity",
    "total_quantity",
    "quantity_unit",
    "postal_code",
    "warehouse",
    "flag_missing_ppu",
    "flag_ppu_computed",
    "flag_unavailable",
]

HIT_COLUMNS = [
    "product_key",
    "search_term",
]

OBSERVATION_KEY = [
    "product_key",
    "collection_date",
    "postal_code",
    "warehouse",
]

ECHO_MIN_UNITS = 2


# --------------------------------------------------
# PRODUCT QUANTITY
# --------------------------------------------------

def choose_pack(name, quantity_raw):
    """Prefer a multipack in the name, then the quantity field."""
    if pd.notna(name) and PACK_MULTI.search(strip_accents(str(name))):
        return parse_pack(name)

    if pd.notna(quantity_raw):
        parsed = parse_pack(quantity_raw)
        if parsed:
            return parsed

    if pd.notna(name):
        return parse_pack(name)

    return None


# --------------------------------------------------
# REFERENCE PRICE
# --------------------------------------------------

def _site_echoes_price(ppu, unit, row):
    """Detect a reference price that repeats the total pack price."""
    return (
        pd.notna(row["price"])
        and row["price"] > 0
        and pd.notna(row["name"])
        and "aprox" not in strip_accents(str(row["name"]))
        and unit == row["quantity_unit"]
        and pd.notna(row["total_quantity"])
        and row["total_quantity"] >= ECHO_MIN_UNITS
        and abs(ppu - row["price"]) <= 0.01 * row["price"]
    )


def _price_per_unit(row):
    """Return normalized reference price, unit, and source."""
    has_size = (
        pd.notna(row["price"])
        and row["price"] > 0
        and pd.notna(row["total_quantity"])
        and row["total_quantity"] > 0
    )

    computed = (
        round(row["price"] / row["total_quantity"], 4)
        if has_size else np.nan
    )

    if (
        pd.notna(row["price_per_unit"])
        and pd.notna(row["ppu_label"])
    ):
        converted = per_base(
            row["price_per_unit"],
            row["ppu_label"],
        )

        if converted:
            ppu, unit = converted

            if (
                has_size
                and _site_echoes_price(ppu, unit, row)
            ):
                return computed, unit, "computed_override"

            return ppu, unit, "site"

    if has_size:
        return computed, row["quantity_unit"], "computed"

    return np.nan, np.nan, np.nan


# --------------------------------------------------
# IDENTIFIERS AND DATES
# --------------------------------------------------

def _product_keys(df):
    """Build stable chain-specific product identifiers."""
    ids = df["native_id"].astype("string").str.strip()

    ids = ids.str.replace(
        r"^(\d+)\.0$",
        r"\1",
        regex=True,
    )

    missing = (
        ids.isna()
        | ids.eq("")
        | ids.eq("nan")
        | ids.eq("None")
    ).fillna(True)

    if missing.any():
        fallback = [
            hash_id(name, quantity)
            for name, quantity in zip(
                df.loc[missing, "name"],
                df.loc[missing, "quantity_raw"],
            )
        ]

        ids.loc[missing] = fallback

    df["native_id"] = ids

    return (
        df["chain_id"].astype("string")
        + ":"
        + ids
    )


def _collection_dates(df):
    """Extract collection dates without changing the original timestamps."""
    dates = (
        df["scraped_at"]
        .astype("string")
        .str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    )

    if "_snapshot_date" in df.columns:
        fallback = df["_snapshot_date"].astype("string")
        dates = dates.fillna(fallback)

    return dates


def _current_flags(df):
    """Interpret the current-snapshot marker safely."""
    if "is_current" not in df.columns:
        return pd.Series(True, index=df.index)

    values = df["is_current"]

    return values.map(
        lambda value: (
            value is True
            or value == 1
            or value == "True"
            or value == "true"
        ) if pd.notna(value) else False
    ).astype(bool)


# --------------------------------------------------
# STANDARDIZATION
# --------------------------------------------------

def standardize(raw, preserve_observations=False):
    """Standardize current and, optionally, historical adapter rows.

    Parameters
    ----------
    raw : pandas.DataFrame
        Rows already transformed into the common adapter schema.

    preserve_observations : bool
        False: preserve the original one-observation-per-product
        behavior for existing pipeline consumers.

        True: retain one observation per product, collection date,
        and location, including historical collections.

    Returns
    -------
    dict
        DataFrames: products, observations, hits.
    """
    if raw.empty:
        raise ValueError("No adapter rows available to standardize.")

    df = raw.copy()

    # Identify products consistently across collection dates.
    df["product_key"] = _product_keys(df)

    # Preserve collection dates and source information.
    df["collection_date"] = _collection_dates(df)
    df["is_current"] = _current_flags(df)

    # Normalize product names and brands.
    df["brand"] = [
        extract_brand(name, brand)
        for name, brand in zip(df["name"], df["brand"])
    ]

    df["is_private_label"] = [
        is_private_label(brand, name)
        for brand, name in zip(df["brand"], df["name"])
    ]

    df["name_clean"] = [
        clean_name(name, brand)
        for name, brand in zip(df["name"], df["brand"])
    ]

    # Extract quantities and multipack information.
    packs = [
        choose_pack(name, quantity)
        for name, quantity in zip(
            df["name"],
            df["quantity_raw"],
        )
    ]

    df["pack_count"] = [
        pack[0] if pack else np.nan
        for pack in packs
    ]

    df["unit_quantity"] = [
        pack[1] if pack else np.nan
        for pack in packs
    ]

    df["quantity_unit"] = [
        pack[2] if pack else np.nan
        for pack in packs
    ]

    df["total_quantity"] = (
        df["pack_count"] * df["unit_quantity"]
    )

    # Normalize reference prices using the same rules
    # for current and historical collections.
    ppu = df.apply(
        _price_per_unit,
        axis=1,
        result_type="expand",
    )

    df["price_per_unit"] = ppu[0]
    df["unit"] = ppu[1]
    df["ppu_source"] = ppu[2]

    # Quality flags.
    df["flag_missing_ppu"] = (
        df["price_per_unit"].isna()
    )

    df["flag_ppu_computed"] = (
        df["ppu_source"].isin(
            ["computed", "computed_override"]
        )
    )

    df["flag_unavailable"] = df["available"].map({
        "False": True,
        "True": False,
        False: True,
        True: False,
    })

    # Prefer current data when constructing product metadata.
    # Within the same source, prefer the latest timestamp.
    ordered = df.sort_values(
        by=["is_current", "scraped_at"],
        ascending=[False, False],
        kind="stable",
        na_position="last",
    )

    products = (
        ordered
        .drop_duplicates(
            subset=["product_key"],
            keep="first",
        )
        [PRODUCT_COLUMNS]
        .reset_index(drop=True)
    )

    # Historical mode retains different dates and locations.
    # Repeated search hits for the same observation are removed.
    if preserve_observations:
        observations = (
            ordered
            .drop_duplicates(
                subset=OBSERVATION_KEY,
                keep="first",
            )
            [OBSERVATION_COLUMNS]
            .reset_index(drop=True)
        )

    else:
        # Backward-compatible behavior for current-only consumers.
        observations = (
            ordered
            .drop_duplicates(
                subset=["product_key"],
                keep="first",
            )
            [OBSERVATION_COLUMNS]
            .reset_index(drop=True)
        )

    # Search hits are product-level relationships.
    # Repeated hits across dates are not duplicated.
    hits = (
        df.loc[
            df["search_term"].notna(),
            HIT_COLUMNS,
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return {
        "products": products,
        "observations": observations,
        "hits": hits,
    }

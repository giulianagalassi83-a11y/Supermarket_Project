"""Step 2 + 3: turn adapter rows into products / price observations / search hits, with flags.

Nothing is dropped: a missing price per unit becomes `flag_missing_ppu`.
"""

import numpy as np
import pandas as pd

from src.pipeline.text import PACK_MULTI, clean_name, extract_brand, is_private_label, parse_pack, strip_accents
from src.pipeline.units import per_base

PRODUCT_COLUMNS = [
    "product_key", "chain_id", "native_id", "ean", "name", "name_clean", "brand", "is_private_label",
    "chain_category", "pack_count", "unit_quantity", "total_quantity", "quantity_unit", "url", "image_url",
]
OBSERVATION_COLUMNS = [
    "product_key", "scraped_at", "price", "regular_price", "offer_price", "price_per_unit", "unit",
    "ppu_source", "postal_code", "warehouse",
    "flag_missing_ppu", "flag_ppu_computed", "flag_unavailable",
]


def choose_pack(name, quantity_raw):
    """A multipack in the name ("6 x 1l") wins, then the size field, then any size in the name."""
    if PACK_MULTI.search(strip_accents(name)):
        return parse_pack(name)
    return parse_pack(quantity_raw) or parse_pack(name)


ECHO_MIN_UNITS = 2  # a pack of at least this many base units can't cost the same per unit as in total


def _site_echoes_price(ppu, unit, row):
    """The site repeated the item price as the €/unit of a multi-unit pack.

    DIA shows "Detergente … 25 lavados" 2.95 € as "(2,95 €/LAVADO)"; the real value is 0.118.
    Only this pattern is overridden: other mismatches (drained weight, items sold by weight,
    rolls inside a "1 ud" pack) are cases where the site knows better than our parsed size.
    Items sold by weight ("pieza 2,5 kg aprox") legitimately show price = €/kg.
    """
    return (
        pd.notna(row["price"]) and row["price"] > 0
        and "aprox" not in strip_accents(row["name"])
        and unit == row["quantity_unit"]
        and row["total_quantity"] >= ECHO_MIN_UNITS
        and abs(ppu - row["price"]) <= 0.01 * row["price"]
    )


def _price_per_unit(row):
    """(price per base unit, unit, source): the site's value if usable, else computed from the pack."""
    has_size = pd.notna(row["price"]) and pd.notna(row["total_quantity"]) and row["total_quantity"] > 0
    computed = round(row["price"] / row["total_quantity"], 4) if has_size else np.nan
    if pd.notna(row["price_per_unit"]) and pd.notna(row["ppu_label"]):
        converted = per_base(row["price_per_unit"], row["ppu_label"])
        if converted:
            ppu, unit = converted
            if has_size and _site_echoes_price(ppu, unit, row):
                return computed, unit, "computed_override"
            return ppu, unit, "site"
    if has_size:
        return computed, row["quantity_unit"], "computed"
    return np.nan, np.nan, np.nan


def standardize(raw):
    """raw: COMMON_COLUMNS rows (one or many chains) → {'products', 'observations', 'hits'}."""
    df = raw.copy()
    df["product_key"] = df["chain_id"] + ":" + df["native_id"].astype(str)
    df["brand"] = [extract_brand(n, b) for n, b in zip(df["name"], df["brand"])]
    df["is_private_label"] = [is_private_label(b, n) for b, n in zip(df["brand"], df["name"])]
    df["name_clean"] = [clean_name(n, b) for n, b in zip(df["name"], df["brand"])]

    packs = [choose_pack(n, q) for n, q in zip(df["name"], df["quantity_raw"])]
    df["pack_count"] = [p[0] if p else np.nan for p in packs]
    df["unit_quantity"] = [p[1] if p else np.nan for p in packs]
    df["quantity_unit"] = [p[2] if p else np.nan for p in packs]
    df["total_quantity"] = df["pack_count"] * df["unit_quantity"]

    ppu = df.apply(_price_per_unit, axis=1, result_type="expand")
    df["price_per_unit"], df["unit"], df["ppu_source"] = ppu[0], ppu[1], ppu[2]
    df["flag_missing_ppu"] = df["price_per_unit"].isna()
    df["flag_ppu_computed"] = df["ppu_source"].isin(["computed", "computed_override"])
    # available: False → unavailable, unknown (NaN) stays unknown
    df["flag_unavailable"] = df["available"].map({"False": True, "True": False, False: True, True: False})

    grouped = df.groupby("product_key", sort=False)
    products = grouped[PRODUCT_COLUMNS[1:]].first().reset_index()[PRODUCT_COLUMNS]
    observations = grouped[OBSERVATION_COLUMNS[1:]].first().reset_index()[OBSERVATION_COLUMNS]
    hits = (
        df.loc[df["search_term"].notna(), ["product_key", "search_term"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return {"products": products, "observations": observations, "hits": hits}

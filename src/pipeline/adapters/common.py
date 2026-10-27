"""Shared pieces for the per-chain adapters."""

import hashlib

import numpy as np
import pandas as pd

from src.pipeline.text import parse_price

# Every adapter returns exactly these columns, one row per raw row (per search term for Mercadona API).
# `price_per_unit` is still "per ppu_label" here; standardize.py converts it to the base unit.
COMMON_COLUMNS = [
    "chain_id", "native_id", "ean", "name", "brand", "chain_category",
    "price", "regular_price", "offer_price", "price_per_unit", "ppu_label",
    "quantity_raw", "available", "url", "image_url", "postal_code", "warehouse",
    "scraped_at", "search_term",
]


def read_raw(path):
    """Read a raw CSV as text; blank/whitespace-only cells become NaN."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    return df.apply(lambda col: col.str.strip()).replace("", np.nan)


def hash_id(name, quantity_raw):
    """Stable generated ID for products the chain gives no ID for."""
    key = f"{name}|{'' if pd.isna(quantity_raw) else quantity_raw}"
    return "h" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def prices(series):
    return series.map(parse_price).astype(float)


def finish(df, chain_id):
    """Add missing columns as NaN, set the chain, return the columns in COMMON_COLUMNS order."""
    df = df.copy()
    df["chain_id"] = chain_id
    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if df["search_term"].notna().any():
        df["search_term"] = df["search_term"].str.lower()
    return df[COMMON_COLUMNS].reset_index(drop=True)

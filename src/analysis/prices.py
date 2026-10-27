"""Shared pricing over data/products_normalized.csv, used by compare_types.ipynb and the dashboard.

Every function filters with `usable()` itself, so callers pass the whole normalized table.
A type id like "any:leche" means "any real type of the search term leche".
"""

import pandas as pd

from src.pipeline import config

ANY_PREFIX = "any:"
STATS = {"median": "median", "min": "min"}


def load_normalized(path=config.CSV_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run: python -m src.pipeline.build")
    return pd.read_csv(path, encoding="utf-8")


def usable(df):
    """Rows we can compare: a real product type and a trustworthy price per unit in the type's unit."""
    keep = (
        df["product_type"].notna()
        & (df["product_type"] != config.NO_RELEVANTE)
        & df["price_per_unit"].notna()
        & (df["flag_outlier"].fillna(0) == 0)
        & (df["flag_missing_ppu"].fillna(0) == 0)
        & (df["flag_unit_mismatch"].fillna(0) == 0)
    )
    return df[keep]


def type_rows(df, type_id):
    """Usable rows of one type; for "any:<term>", every real type of the term in its most common unit."""
    df = usable(df)
    if not type_id.startswith(ANY_PREFIX):
        return df[df["product_type"] == type_id]
    rows = df[df["searched_item"] == type_id[len(ANY_PREFIX):]]
    if rows.empty:
        return rows
    return rows[rows["unit"] == rows["unit"].mode().iat[0]]


def type_unit(df, type_id):
    units = type_rows(df, type_id)["unit"]
    return units.mode().iat[0] if len(units) else None


def type_price(df, type_id, chain, stat="median"):
    """€ per unit of a type in one chain (median or cheapest); NaN if the chain has none."""
    values = type_rows(df, type_id)
    values = values.loc[values["supermarket"] == chain, "price_per_unit"]
    return values.agg(STATS[stat]) if len(values) else float("nan")


def price_table(df, type_ids, stat="median"):
    """Types × chains table of € per unit."""
    parts = []
    for type_id in type_ids:
        rows = type_rows(df, type_id)
        parts.append(rows.groupby("supermarket")["price_per_unit"].agg(STATS[stat]).rename(type_id))
    return pd.DataFrame(parts)


def basket_cost(df, basket, chain, stat="median"):
    """(total € for the types the chain has, list of type ids it lacks) for basket = {type_id: quantity}."""
    total, missing = 0.0, []
    for type_id, quantity in basket.items():
        price = type_price(df, type_id, chain, stat)
        if pd.isna(price):
            missing.append(type_id)
        else:
            total += price * quantity
    return total, missing

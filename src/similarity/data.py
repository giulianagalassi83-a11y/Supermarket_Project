"""Data access for the similarity engine: current products with their search term, type and text."""

import sqlite3

import pandas as pd

from src.pipeline import config
from src.pipeline.classify import product_text
from src.pipeline.standardize import PRODUCT_COLUMNS


def load_products(db_path=config.DB_PATH):
    """One row per current product: PRODUCT_COLUMNS + search_term (first hit) + type_id + text."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run: python -m src.pipeline.build")
    cols = ", ".join(f"p.{c}" for c in PRODUCT_COLUMNS)
    with sqlite3.connect(db_path) as con:
        products = pd.read_sql_query(
            f"""
            SELECT {cols}, c.type_id
            FROM products p
            LEFT JOIN product_classification c ON c.product_key = p.product_key
            WHERE EXISTS (SELECT 1 FROM price_observations o
                          WHERE o.product_key = p.product_key AND o.is_current = 1)
            """,
            con, dtype={"ean": str},
        )
        hits = pd.read_sql_query("SELECT product_key, search_term FROM search_hits", con)
    order = {t: i for i, t in enumerate(config.SEARCH_TERMS)}
    hits["order"] = hits["search_term"].map(order)
    first = hits.sort_values(["product_key", "order"]).drop_duplicates("product_key")
    products = products.merge(first[["product_key", "search_term"]], on="product_key", how="left")
    products["text"] = [product_text(n, c) for n, c in zip(products["name_clean"], products["chain_category"])]
    products["mtext"] = [match_text(n, b) for n, b in zip(products["name"], products["brand"])]
    return products.reset_index(drop=True)


def barcode_pairs(products):
    """BM–Consum pairs sharing an EAN that appears exactly once in each chain: ean, key_bm, key_consum, search_term."""
    p = products[products["chain_id"].isin(["bm", "consum"])].copy()
    p["ean"] = p["ean"].astype("string").str.strip()
    p = p[p["ean"].notna() & (p["ean"] != "")]
    counts = p.groupby(["chain_id", "ean"])["product_key"].transform("size")
    p = p[counts == 1]
    bm = p[p["chain_id"] == "bm"][["ean", "product_key", "search_term"]].rename(columns={"product_key": "key_bm"})
    consum = p[p["chain_id"] == "consum"][["ean", "product_key"]].rename(columns={"product_key": "key_consum"})
    pairs = bm.merge(consum, on="ean", validate="one_to_one")
    return pairs[["ean", "key_bm", "key_consum", "search_term"]].sort_values("ean").reset_index(drop=True)


def match_text(name, brand):
    """Text used for cross-chain matching: brand + full name (name_clean drops the brand, which matters here)."""
    name = str(name).strip()
    if brand is None or (isinstance(brand, float) and brand != brand) or not str(brand).strip():
        return name
    brand = str(brand).strip()
    return name if brand.lower() in name.lower() else f"{brand} {name}"

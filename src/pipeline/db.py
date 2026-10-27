"""Step 5: SQLite database (rebuilt from scratch every run) and the flat CSV export."""

import sqlite3

import pandas as pd

from src.pipeline import config

SCHEMA = """
CREATE TABLE chains (
    chain_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    collection_method TEXT
);
CREATE TABLE products (
    product_key TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL REFERENCES chains(chain_id),
    native_id TEXT, ean TEXT, name TEXT NOT NULL, name_clean TEXT, brand TEXT,
    is_private_label INTEGER, chain_category TEXT,
    pack_count REAL, unit_quantity REAL, total_quantity REAL, quantity_unit TEXT,
    url TEXT, image_url TEXT
);
CREATE TABLE price_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_key TEXT NOT NULL REFERENCES products(product_key),
    scraped_at TEXT, price REAL, regular_price REAL, offer_price REAL,
    price_per_unit REAL, unit TEXT, ppu_source TEXT, postal_code TEXT, warehouse TEXT,
    flag_missing_ppu INTEGER, flag_ppu_computed INTEGER, flag_outlier INTEGER,
    flag_unavailable INTEGER, flag_unit_mismatch INTEGER
);
CREATE TABLE search_hits (
    product_key TEXT NOT NULL REFERENCES products(product_key),
    search_term TEXT NOT NULL,
    PRIMARY KEY (product_key, search_term)
);
CREATE TABLE product_types (
    type_id TEXT PRIMARY KEY,
    search_term TEXT, label TEXT, description TEXT, is_negative INTEGER, unit TEXT
);
CREATE TABLE product_classification (
    product_key TEXT PRIMARY KEY REFERENCES products(product_key),
    type_id TEXT REFERENCES product_types(type_id),
    score REAL, runner_up_type_id TEXT, margin REAL, low_confidence INTEGER,
    method TEXT, model_name TEXT
);

-- One row per product with its latest price; this is what products_normalized.csv contains.
-- searched_item = search term of the assigned type, else (no_relevante / unclassified) the first hit.
CREATE VIEW v_products_latest AS
SELECT
    c.name AS supermarket,
    COALESCE(pt.search_term,
             (SELECT MIN(h.search_term) FROM search_hits h WHERE h.product_key = p.product_key)) AS searched_item,
    pc.type_id AS product_type,
    pt.label AS product_type_label,
    p.name, p.brand, p.is_private_label,
    o.price, o.price_per_unit, o.unit,
    pc.low_confidence,
    o.flag_missing_ppu, o.flag_ppu_computed, o.flag_outlier, o.flag_unavailable, o.flag_unit_mismatch,
    p.product_key, p.url
FROM products p
JOIN chains c ON c.chain_id = p.chain_id
JOIN price_observations o ON o.observation_id = (
    SELECT o2.observation_id FROM price_observations o2
    WHERE o2.product_key = p.product_key
    ORDER BY o2.scraped_at DESC, o2.observation_id DESC LIMIT 1)
LEFT JOIN product_classification pc ON pc.product_key = p.product_key
LEFT JOIN product_types pt ON pt.type_id = pc.type_id;
"""

EXPORT_COLUMNS = [
    "supermarket", "searched_item", "product_type", "product_type_label", "name", "brand",
    "is_private_label", "price", "price_per_unit", "unit", "low_confidence",
    "flag_missing_ppu", "flag_ppu_computed", "flag_outlier", "flag_unavailable", "flag_unit_mismatch",
    "product_key", "url",
]

# dict key in `tables` → SQL table name
TABLE_NAMES = {
    "products": "products",
    "observations": "price_observations",
    "hits": "search_hits",
    "product_types": "product_types",
    "classification": "product_classification",
}


def chains_table():
    return pd.DataFrame(
        [{"chain_id": cid, **info} for cid, info in config.CHAINS.items()]
    )[["chain_id", "name", "collection_method"]]


def _insert(con, table, df):
    columns = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    df = df[[c for c in df.columns if c in columns]]
    df = df.astype(object).where(df.notna(), None)
    for col in df.columns:  # booleans → 0/1 for SQLite
        df[col] = df[col].map(lambda v: int(v) if isinstance(v, bool) else v)
    df.to_sql(table, con, if_exists="append", index=False)


def write_db(path, tables):
    """Delete and recreate the database from the given DataFrames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        _insert(con, "chains", chains_table())
        for key, table in TABLE_NAMES.items():
            if key in tables:
                _insert(con, table, tables[key])
    con.close()


def export_csv(db_path, csv_path):
    """Write v_products_latest to CSV and return it."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM v_products_latest", con)
    con.close()
    df = df[EXPORT_COLUMNS]
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return df

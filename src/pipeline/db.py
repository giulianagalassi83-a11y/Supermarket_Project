
"""SQLite storage and exports for current and historical supermarket prices.

The database is rebuilt from standardized data.
A temporary database is created first, so a failed build
does not delete the previous database.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from contextlib import closing
import pandas as pd

from src.pipeline import config


# --------------------------------------------------
# DATABASE SCHEMA
# --------------------------------------------------

SCHEMA = """
CREATE TABLE chains (
    chain_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    collection_method TEXT
);

CREATE TABLE products (
    product_key TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL REFERENCES chains(chain_id),
    native_id TEXT,
    ean TEXT,
    name TEXT NOT NULL,
    name_clean TEXT,
    brand TEXT,
    is_private_label INTEGER,
    chain_category TEXT,
    pack_count REAL,
    unit_quantity REAL,
    total_quantity REAL,
    quantity_unit TEXT,
    url TEXT,
    image_url TEXT
);

CREATE TABLE price_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_key TEXT NOT NULL REFERENCES products(product_key),

    scraped_at TEXT,
    collection_date TEXT,
    is_current INTEGER NOT NULL DEFAULT 0,

    price REAL,
    regular_price REAL,
    offer_price REAL,
    price_per_unit REAL,
    unit TEXT,
    ppu_source TEXT,

    pack_count REAL,
    unit_quantity REAL,
    total_quantity REAL,
    quantity_unit TEXT,

    postal_code TEXT,
    warehouse TEXT,

    flag_missing_ppu INTEGER,
    flag_ppu_computed INTEGER,
    flag_outlier INTEGER,
    flag_unavailable INTEGER,
    flag_unit_mismatch INTEGER
);

CREATE INDEX idx_observations_product_date
ON price_observations (
    product_key,
    collection_date
);

CREATE INDEX idx_observations_current
ON price_observations (
    is_current,
    product_key
);

CREATE TABLE search_hits (
    product_key TEXT NOT NULL REFERENCES products(product_key),
    search_term TEXT NOT NULL,
    PRIMARY KEY (product_key, search_term)
);

CREATE TABLE product_types (
    type_id TEXT PRIMARY KEY,
    search_term TEXT,
    label TEXT,
    description TEXT,
    is_negative INTEGER,
    unit TEXT
);

CREATE TABLE product_classification (
    product_key TEXT PRIMARY KEY REFERENCES products(product_key),
    type_id TEXT REFERENCES product_types(type_id),
    score REAL,
    runner_up_type_id TEXT,
    margin REAL,
    low_confidence INTEGER,
    method TEXT,
    model_name TEXT
);

-- Current catalog for basket optimization.
-- Historical-only products are excluded.
-- The latest observation is selected from the current RAW source.

CREATE VIEW v_products_latest AS
SELECT
    c.name AS supermarket,

    COALESCE(
        pt.search_term,
        (
            SELECT MIN(h.search_term)
            FROM search_hits h
            WHERE h.product_key = p.product_key
        )
    ) AS searched_item,

    pc.type_id AS product_type,
    pt.label AS product_type_label,

    p.name,
    p.brand,
    p.is_private_label,

    o.price,
    o.price_per_unit,
    o.unit,

    pc.low_confidence,

    o.flag_missing_ppu,
    o.flag_ppu_computed,
    o.flag_outlier,
    o.flag_unavailable,
    o.flag_unit_mismatch,

    p.product_key,
    p.url

FROM products p

JOIN chains c
    ON c.chain_id = p.chain_id

JOIN price_observations o
    ON o.observation_id = (
        SELECT o2.observation_id
        FROM price_observations o2
        WHERE o2.product_key = p.product_key
          AND o2.is_current = 1
        ORDER BY
            o2.collection_date DESC,
            o2.scraped_at DESC,
            o2.observation_id DESC
        LIMIT 1
    )

LEFT JOIN product_classification pc
    ON pc.product_key = p.product_key

LEFT JOIN product_types pt
    ON pt.type_id = pc.type_id;


-- Normalized historical observations.
-- Every collection date is retained, including current observations.

CREATE VIEW v_price_history AS
SELECT
    c.name AS supermarket,
    p.chain_id,
    p.product_key,
    p.native_id,
    p.ean,

    p.name,
    p.brand,

    pc.type_id AS product_type,
    pt.label AS product_type_label,

    COALESCE(
        pt.search_term,
        (
            SELECT MIN(h.search_term)
            FROM search_hits h
            WHERE h.product_key = p.product_key
        )
    ) AS searched_item,

    o.scraped_at,
    o.collection_date,
    o.is_current,

    o.price,
    o.regular_price,
    o.offer_price,
    o.price_per_unit,
    o.unit,
    o.ppu_source,

    o.pack_count,
    o.unit_quantity,
    o.total_quantity,
    o.quantity_unit,

    o.postal_code,
    o.warehouse,

    o.flag_missing_ppu,
    o.flag_ppu_computed,
    o.flag_outlier,
    o.flag_unavailable,
    o.flag_unit_mismatch,

    pc.low_confidence

FROM price_observations o

JOIN products p
    ON p.product_key = o.product_key

JOIN chains c
    ON c.chain_id = p.chain_id

LEFT JOIN product_classification pc
    ON pc.product_key = p.product_key

LEFT JOIN product_types pt
    ON pt.type_id = pc.type_id;
"""


# --------------------------------------------------
# EXPORT COLUMNS
# --------------------------------------------------

EXPORT_COLUMNS = [
    "supermarket",
    "searched_item",
    "product_type",
    "product_type_label",
    "name",
    "brand",
    "is_private_label",
    "price",
    "price_per_unit",
    "unit",
    "low_confidence",
    "flag_missing_ppu",
    "flag_ppu_computed",
    "flag_outlier",
    "flag_unavailable",
    "flag_unit_mismatch",
    "product_key",
    "url",
]

HISTORY_EXPORT_COLUMNS = [
    "supermarket",
    "chain_id",
    "product_key",
    "native_id",
    "ean",
    "name",
    "brand",
    "product_type",
    "product_type_label",
    "searched_item",
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
    "flag_outlier",
    "flag_unavailable",
    "flag_unit_mismatch",
    "low_confidence",
]

TABLE_NAMES = {
    "products": "products",
    "observations": "price_observations",
    "hits": "search_hits",
    "product_types": "product_types",
    "classification": "product_classification",
}


# --------------------------------------------------
# DATABASE HELPERS
# --------------------------------------------------

def chains_table():
    """Build the supermarket reference table."""
    return pd.DataFrame(
        [
            {"chain_id": chain_id, **info}
            for chain_id, info in config.CHAINS.items()
        ]
    )[
        ["chain_id", "name", "collection_method"]
    ]


def _insert(con, table, df):
    """Insert a DataFrame into an existing SQLite table."""
    if df.empty:
        return

    columns = [
        row[1]
        for row in con.execute(
            f"PRAGMA table_info({table})"
        )
    ]

    data = df[
        [col for col in df.columns if col in columns]
    ].copy()

    data = (
        data.astype(object)
        .where(data.notna(), None)
    )

    for col in data.columns:
        data[col] = data[col].map(
            lambda value: (
                int(value)
                if isinstance(value, bool)
                else value
            )
        )

    data.to_sql(
        table,
        con,
        if_exists="append",
        index=False,
    )


def _temporary_path(directory, suffix):
    """Create a temporary path in the destination directory."""
    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, filename = tempfile.mkstemp(
        prefix=".supermarket_",
        suffix=suffix,
        dir=directory,
    )

    os.close(fd)

    return Path(filename)


# --------------------------------------------------
# DATABASE BUILD
# --------------------------------------------------


def write_db(path, tables):
    """Build SQLite safely and replace the existing database on success."""
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = _temporary_path(
        path.parent,
        ".db",
    )

    try:
        # Ensure the SQLite connection is explicitly closed.
        # This is important on Windows, where open database
        # connections can prevent temporary file deletion.
        with closing(sqlite3.connect(temp_path)) as con:

            con.execute("PRAGMA foreign_keys = ON")

            con.executescript(SCHEMA)

            with con:
                _insert(
                    con,
                    "chains",
                    chains_table(),
                )

                for key, table_name in TABLE_NAMES.items():
                    if key in tables:
                        _insert(
                            con,
                            table_name,
                            tables[key],
                        )

            # Validate the database before publishing it.
            product_count = con.execute(
                "SELECT COUNT(*) FROM products"
            ).fetchone()[0]

            current_count = con.execute(
                "SELECT COUNT(*) FROM v_products_latest"
            ).fetchone()[0]

            observation_count = con.execute(
                "SELECT COUNT(*) FROM price_observations"
            ).fetchone()[0]

            if (
                product_count == 0
                or current_count == 0
                or observation_count == 0
            ):
                raise ValueError(
                    "Database validation failed: "
                    "missing products or price observations."
                )

            violations = con.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

            if violations:
                raise ValueError(
                    f"Foreign key violations: {violations[:5]}"
                )

            print(
                "[DB VALIDATION] "
                f"Products: {product_count}, "
                f"Current products: {current_count}, "
                f"Observations: {observation_count}"
            )

        # The SQLite connection is now closed.
        # Replace the destination only after successful validation.
        os.replace(
            temp_path,
            path,
        )

    finally:
        # Remove temporary files left by failed builds.
        if temp_path.exists():
            try:
                temp_path.unlink()
            except PermissionError:
                print(
                    "WARNING: Could not remove temporary database: "
                    f"{temp_path}"
                )

# --------------------------------------------------
# CSV EXPORT HELPERS
# --------------------------------------------------

def _export_view(db_path, csv_path, view, columns):
    """Export a database view with a safe temporary CSV."""
    db_path = Path(db_path)
    csv_path = Path(csv_path)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            f"SELECT * FROM {view}",
            con,
        )

    df = df[columns]

    temp_path = _temporary_path(
        csv_path.parent,
        ".csv",
    )

    try:
        df.to_csv(
            temp_path,
            index=False,
            encoding="utf-8",
        )

        os.replace(
            temp_path,
            csv_path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()

    return df


# --------------------------------------------------
# CURRENT CATALOG EXPORT
# --------------------------------------------------

def export_csv(db_path, csv_path):
    """Export the current normalized catalog."""
    return _export_view(
        db_path,
        csv_path,
        "v_products_latest",
        EXPORT_COLUMNS,
    )


# --------------------------------------------------
# HISTORICAL EXPORT
# --------------------------------------------------

def export_history_csv(db_path, csv_path):
    """Export normalized observations for all collection dates."""
    df = _export_view(
        db_path,
        csv_path,
        "v_price_history",
        HISTORY_EXPORT_COLUMNS,
    )

    # Preserve chronological order in the historical CSV.
    df = df.sort_values(
        [
            "supermarket",
            "product_key",
            "collection_date",
            "scraped_at",
        ],
        kind="stable",
        na_position="last",
    )

    temp_path = _temporary_path(
        Path(csv_path).parent,
        ".csv",
    )

    try:
        df.to_csv(
            temp_path,
            index=False,
            encoding="utf-8",
        )

        os.replace(
            temp_path,
            csv_path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()

    return df

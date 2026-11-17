
"""Validate seven-chain outputs and freshness of the six automated chains."""

import json
import sqlite3
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXPECTED = {
    "Mercadona",
    "Bm",
    "Consum",
    "Alcampo",
    "Ahorramás",
    "Dia",
    "Aldi",
}

ACTIVE = {
    "mercadona",
    "bm",
    "consum",
    "alcampo",
    "ahorramas",
    "aldi",
}


# --------------------------------------------------
# VALIDATE OUTPUT FILES
# --------------------------------------------------

for filename in (
    "products_normalized.csv",
    "price_history_normalized.csv",
    "price_changes.csv",
    "supermarkets.db",
):

    path = DATA / filename

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Missing or empty pipeline output: {path}"
        )


products = pd.read_csv(
    DATA / "products_normalized.csv"
)

observations = pd.read_csv(
    DATA / "price_history_normalized.csv"
)

changes = pd.read_csv(
    DATA / "price_changes.csv"
)


# --------------------------------------------------
# VALIDATE SUPERMARKET COVERAGE
# --------------------------------------------------

if (
    len(products) < 1000
    or len(observations) < len(products)
):
    raise RuntimeError(
        "Suspiciously small or inconsistent normalized data"
    )

if set(products["supermarket"].dropna()) != EXPECTED:
    raise RuntimeError(
        "Expected seven chains in current normalized products"
    )

if not EXPECTED.issubset(
    set(changes["supermarket"].dropna())
):
    raise RuntimeError(
        "Missing one or more supermarket comparisons"
    )


# --------------------------------------------------
# VALIDATE COLLECTION DATES
# --------------------------------------------------

manifests = (
    ROOT
    / "downloaded_collectors"
    / "_collector_manifests"
)

dates = {
    json.loads(
        p.read_text(encoding="utf-8")
    )["collection_date"]
    for p in manifests.glob("*.json")
}

if len(dates) != 1:
    raise RuntimeError(
        "Expected one common collection date"
    )

day = dates.pop()

current = observations.loc[
    observations["is_current"]
    .astype(str)
    .str.lower()
    .isin(["1", "true", "1.0"])
]

for chain in sorted(ACTIVE):

    rows = current.loc[
        current["chain_id"] == chain
    ]

    if (
        rows.empty
        or set(
            rows["collection_date"]
            .dropna()
            .astype(str)
        ) != {day}
        or rows["collection_date"].isna().any()
    ):
        raise RuntimeError(
            f"Current observations for {chain} "
            f"do not match {day}"
        )

    print(
        f"Fresh: {chain}, {day}, "
        f"{len(rows)} observations"
    )


# --------------------------------------------------
# VALIDATE RETAINED DIA DATA
# --------------------------------------------------

dia = current.loc[
    current["chain_id"] == "dia"
]

if dia.empty or dia["collection_date"].isna().any():
    raise RuntimeError(
        "DIA current observations are missing"
    )

print(
    "DIA retained, latest date:",
    dia["collection_date"].max()
)


# --------------------------------------------------
# VALIDATE SQLITE DATABASE
# --------------------------------------------------

with sqlite3.connect(DATA / "supermarkets.db") as con:

    product_count = con.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    history_count = con.execute(
        "SELECT COUNT(*) FROM price_observations"
    ).fetchone()[0]

    current_count = con.execute(
        "SELECT COUNT(*) FROM v_products_latest"
    ).fetchone()[0]

    if (
        product_count < 1000
        or history_count != len(observations)
        or current_count != len(products)
    ):
        raise RuntimeError(
            "Unexpected SQLite product or observation counts"
        )

    if con.execute(
        "PRAGMA quick_check"
    ).fetchone()[0] != "ok":
        raise RuntimeError(
            "SQLite integrity check failed"
        )


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\nPIPELINE VALIDATION PASSED")

print(f"Current products: {len(products)}")
print(f"Price observations: {len(observations)}")
print(f"Price comparisons: {len(changes)}")
print(f"SQLite products: {product_count}")
print(f"New collection date: {day}")
print("Automated supermarkets: 6")
print("Total supermarkets: 7")

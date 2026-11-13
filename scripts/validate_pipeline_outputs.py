"""Sanity-check pipeline outputs before committing any collected data."""
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED = {"Mercadona", "Bm", "Consum", "Alcampo", "Ahorramás", "Dia", "Aldi"}

for filename in ("products_normalized.csv", "price_history_normalized.csv", "price_changes.csv"):
    path = DATA / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty pipeline output: {path}")

products = pd.read_csv(DATA / "products_normalized.csv")
observations = pd.read_csv(DATA / "price_history_normalized.csv")
changes = pd.read_csv(DATA / "price_changes.csv")

if len(products) < 1000 or len(observations) < len(products):
    raise RuntimeError("Suspiciously small or inconsistent normalized data")
if "supermarket" not in products or set(products["supermarket"].dropna()) != EXPECTED:
    raise RuntimeError("Expected seven chains in current normalized products")
if "supermarket" not in changes or not EXPECTED.issubset(set(changes["supermarket"].dropna())):
    raise RuntimeError("Missing one or more supermarket comparisons")

with sqlite3.connect(DATA / "supermarkets.db") as con:
    product_count = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    history_count = con.execute("SELECT COUNT(*) FROM price_observations").fetchone()[0]
    if product_count < 1000 or history_count < product_count:
        raise RuntimeError("Unexpected SQLite product/observation counts")

print(f"PASS: products={len(products)}, observations={len(observations)}, "
      f"comparisons={len(changes)}, SQLite products={product_count}")

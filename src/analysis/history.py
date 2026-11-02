"""Price changes between the collection dates we already hold, used by the dashboard's Trends page.

Three chains were collected on two dates. This is two points per chain, so it shows price changes,
not a trend line:
    Aldi, Dia: 23 Sep (recovered from git) and 30 Sep, joined on the chain's product id.
    Mercadona: 23 Sep (Selenium) and 8 Oct (API), joined on names that appear once in each file.

Usage (from the project root):  python -m src.analysis.history
"""

import subprocess

import numpy as np
import pandas as pd

from src.analysis import prices
from src.pipeline import config
from src.pipeline.text import strip_accents

HISTORY_DIR = config.DATA_DIR / "history"
CHANGES_PATH = config.DATA_DIR / "price_changes.csv"
OLD_COMMIT = "25de24d"  # the first data commit: Aldi and Dia as collected on 23 Sep
OLD_FILES = {"aldi": "aldi_products_raw.csv", "dia": "dia_products_raw.csv"}
HOLD_BACK = 1.5  # a price multiplied or divided by more than this goes to review, not to the rankings
MIN_CATEGORY_PRODUCTS = 10
UNCHANGED = 0.001


def recover_snapshots():
    """Copy the 23 Sep Aldi and Dia files out of git into data/history/ (skips files already there)."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for chain, name in OLD_FILES.items():
        target = HISTORY_DIR / f"{chain}_2026-09-23.csv"
        if not target.exists():
            spec = f"{OLD_COMMIT}:data/raw/{name}"
            target.write_bytes(subprocess.run(["git", "show", spec], cwd=config.ROOT, capture_output=True, check=True).stdout)


def _number(values):
    return pd.to_numeric(values.astype(str).str.replace("€", "").str.replace(",", ".").str.strip(), errors="coerce")


def _date(values):
    return str(values.min())[:10]


def _pairs(chain, old, new):
    return pd.DataFrame({
        "supermarket": config.CHAINS[chain]["name"], "product_key": new["product_key"].values,
        "name": new["name"].values, "old_price": old.values, "new_price": new["new_price"].values,
    })


def _by_id(chain):
    """Same product id on both dates."""
    old = pd.read_csv(HISTORY_DIR / f"{chain}_2026-09-23.csv", encoding="utf-8")
    new = pd.read_csv(config.RAW_DIR / OLD_FILES[chain], encoding="utf-8")
    both = old.dropna(subset=["product_id"]).drop_duplicates("product_id").merge(
        new.dropna(subset=["product_id"]).drop_duplicates("product_id"), on="product_id", suffixes=("_old", ""))
    both = both.assign(new_price=_number(both["price"]), product_key=chain + ":" + both["product_id"].astype(str))
    pairs = _pairs(chain, _number(both["price_old"]), both)
    return pairs.assign(old_date=_date(old["scraped_at"]), new_date=_date(new["scraped_at"]))


def _name_key(names):
    return names.map(lambda n: " ".join(strip_accents(str(n)).lower().split()))


def _mercadona():
    """Selenium against API: only names with one format in the first file and one row in the second."""
    old = pd.read_csv(config.RAW_DIR / config.RAW_FILES["mercadona_selenium"], encoding="utf-8")
    new = pd.read_csv(config.RAW_DIR / config.RAW_FILES["mercadona_api"], encoding="utf-8")
    old["key"], new["key"] = _name_key(old["name"]), _name_key(new["name"])
    formats = old.drop_duplicates(["key", "quantity_raw"])
    formats = formats[formats.groupby("key")["key"].transform("size") == 1]
    both = formats.merge(new[new.groupby("key")["key"].transform("size") == 1], on="key", suffixes=("_old", ""))
    both = both.assign(new_price=_number(both["unit_price"]),
                       product_key="mercadona:" + both["product_id"].astype(str).str.removesuffix(".0"))
    pairs = _pairs("mercadona", _number(both["price"]), both)
    return pairs.assign(old_date=_date(old["scraped_at"]), new_date=_date(new["scraped_at"]))


def build_changes(path=CHANGES_PATH):
    """Join every chain's two dates and write data/price_changes.csv."""
    recover_snapshots()
    pairs = pd.concat([_by_id("aldi"), _by_id("dia"), _mercadona()], ignore_index=True)
    pairs = pairs[(pairs["old_price"] > 0) & (pairs["new_price"] > 0)].copy()
    normalized = prices.load_normalized()
    known = normalized[["product_key", "searched_item", "product_type_label"]].drop_duplicates("product_key")
    pairs = pairs.merge(known, on="product_key", how="left")
    pairs["comparable"] = pairs["product_key"].isin(prices.usable(normalized)["product_key"])
    ratio = pairs["new_price"] / pairs["old_price"]
    pairs["change_pct"] = 100 * (ratio - 1)
    pairs["held_back"] = (ratio > HOLD_BACK) | (ratio < 1 / HOLD_BACK)
    pairs.to_csv(path, index=False, encoding="utf-8")
    return pairs


def load_changes(path=CHANGES_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run: python -m src.analysis.history")
    return pd.read_csv(path, encoding="utf-8")


def average_change(changes):
    """Geometric mean of the price changes, in %."""
    return 100 * (np.exp(np.log1p(changes["change_pct"] / 100).mean()) - 1)


def chain_summary(changes):
    """One row per chain: dates, products compared, share up/unchanged/down, average change."""
    rows = []
    for chain, group in changes.groupby("supermarket"):
        ratio = 1 + group["change_pct"] / 100
        rows.append({
            "chain": chain, "old_date": group["old_date"].iat[0], "new_date": group["new_date"].iat[0],
            "products": len(group), "up_pct": 100 * (ratio > 1 + UNCHANGED).mean(),
            "down_pct": 100 * (ratio < 1 - UNCHANGED).mean(), "average_pct": average_change(group),
        })
    summary = pd.DataFrame(rows)
    summary["changed_pct"] = summary["up_pct"] + summary["down_pct"]
    summary["same_pct"] = 100 - summary["changed_pct"]
    return summary.sort_values("changed_pct", ascending=False).reset_index(drop=True)


def category_changes(changes, min_products=MIN_CATEGORY_PRODUCTS):
    """Average change per category and chain, only where the chain has enough comparable products."""
    rows = changes[changes["comparable"] & ~changes["held_back"] & changes["searched_item"].notna()]
    out = [
        {"category": category, "chain": chain, "products": len(group), "average_pct": average_change(group)}
        for (category, chain), group in rows.groupby(["searched_item", "supermarket"])
        if len(group) >= min_products
    ]
    return pd.DataFrame(out)


def biggest_moves(changes, n=8):
    """(rises, falls): the largest changes, leaving out those held back for review."""
    rows = changes[~changes["held_back"]]
    return rows.nlargest(n, "change_pct"), rows.nsmallest(n, "change_pct")


def offers():
    """Share of products on offer today, from the regular and offer prices the APIs return."""
    rows = []
    for chain in ("bm", "consum"):
        raw = pd.read_csv(config.RAW_DIR / config.RAW_FILES[chain], encoding="utf-8").drop_duplicates("product_id")
        deal = raw[(raw["offer_price"] > 0) & (raw["offer_price"] < raw["regular_price"])]
        rows.append({"chain": config.CHAINS[chain]["name"], "products": len(raw), "on_offer": len(deal),
                     "median_discount_pct": 100 * (1 - deal["offer_price"] / deal["regular_price"]).median()})
    table = pd.DataFrame(rows)
    table["on_offer_pct"] = 100 * table["on_offer"] / table["products"]

    mercadona = pd.read_csv(config.RAW_DIR / config.RAW_FILES["mercadona_api"], encoding="utf-8")
    was = mercadona[mercadona["previous_unit_price"].notna()]
    former = {"products": len(was), "median_drop_pct": 100 * (1 - was["unit_price"] / was["previous_unit_price"]).median()}
    return table, former


if __name__ == "__main__":
    print(chain_summary(build_changes()).round(1).to_string(index=False))
    print(f"\nWrote {CHANGES_PATH.relative_to(config.ROOT)}")

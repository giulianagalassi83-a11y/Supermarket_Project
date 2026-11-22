"""Exact product-to-product price comparisons from the similarity engine's 'same' matches.

Prices are compared per unit (€/kg, €/L, €/unit) in each product's most common unit: item prices
mislead for produce sold per kilo at one chain and per piece at another, and for size mismatches.

These tables sit beside the type medians; they do not change type prices or the optimizer.
Ahorramás is left out: its scraper sometimes stores the €/kg reference price as the item price
(e.g. a 200 g cheese at 19,75 €), which would make it look up to 5× dearer on the same product.
"""

import pandas as pd

from src.pipeline import config

# Chains left out of the same-product table. Ahorramás was here until its scraper's
# price extraction was fixed on 22 Nov 2026 (it stored cents only or the €/kg price).
EXCLUDED = ()


def _confidence(matches):
    """Per product: the weakest confidence among its 'same' matches (high > balanced)."""
    same = matches[matches["level"] == "same"]
    long = pd.concat([same[["product_key_a", "confidence"]].rename(columns={"product_key_a": "product_key"}),
                      same[["product_key_b", "confidence"]].rename(columns={"product_key_b": "product_key"})])
    rank = long["confidence"].map({"high": 1, "balanced": 0})
    return long.assign(rank=rank).groupby("product_key")["rank"].min()


def _usable(df):
    keep = df["price_per_unit"].notna()
    for flag in ("flag_outlier", "flag_unit_mismatch"):
        if flag in df:
            keep &= df[flag].fillna(0) == 0
    return df[keep]


def same_product_table(df, matches, canonical, exclude=EXCLUDED):
    """One row per canonical product: name, unit, € per unit per chain, cheapest chain, gap % and confidence."""
    cols = ["product_key", "supermarket", "name", "price_per_unit", "unit"] + [
        f for f in ("flag_outlier", "flag_unit_mismatch") if f in df]
    rows = _usable(canonical[["canonical_id", "product_key"]].merge(df[cols], on="product_key"))
    rows = rows[~rows["supermarket"].isin(exclude)]
    if not rows.empty:
        unit = rows.groupby("canonical_id")["unit"].agg(lambda u: u.mode().iat[0])
        rows = rows[rows["unit"] == rows["canonical_id"].map(unit)]
    rows = rows.drop_duplicates(["canonical_id", "supermarket"])
    if rows.empty:
        return pd.DataFrame(columns=["canonical_id", "name", "unit", "chains", "cheapest", "gap_pct", "confidence"])
    wide = rows.pivot(index="canonical_id", columns="supermarket", values="price_per_unit")
    chains = list(wide.columns)
    out = wide.assign(chains=wide.notna().sum(axis=1))
    out = out[out["chains"] >= 2]
    out["cheapest"] = out[chains].idxmin(axis=1)
    out["gap_pct"] = 100 * (out[chains].max(axis=1) / out[chains].min(axis=1) - 1)
    out["name"] = rows.groupby("canonical_id")["name"].first()
    out["unit"] = rows.groupby("canonical_id")["unit"].first()
    conf = rows.set_index("product_key")["canonical_id"].to_frame().join(_confidence(matches))
    out["confidence"] = conf.groupby("canonical_id")["rank"].min().map({1: "high", 0: "balanced"})
    return out.reset_index()[["canonical_id", "name", "unit", *chains, "chains", "cheapest", "gap_pct", "confidence"]]


def matched_gap(a="Bm", b="Consum", df=None, matches=None, canonical=None):
    """Median € per unit gap of chain a over chain b on the same product (for the report)."""
    if df is None:
        df = pd.read_csv(config.CSV_PATH, encoding="utf-8")
    matches = pd.read_csv(config.SIM_MATCHES_PATH) if matches is None else matches
    canonical = pd.read_csv(config.SIM_CANONICAL_PATH) if canonical is None else canonical
    table = same_product_table(df, matches, canonical, exclude=())
    both = table.dropna(subset=[a, b]) if {a, b} <= set(table.columns) else table.iloc[0:0]
    gap = 100 * (both[a] / both[b] - 1) if len(both) else pd.Series(dtype=float)
    return {"a": a, "b": b, "n": int(len(both)), "median_gap_pct": float(gap.median()) if len(gap) else float("nan"),
            "same_price_share": float((gap.abs() <= 0.5).mean()) if len(gap) else float("nan"),
            "a_cheaper_share": float((gap < -0.5).mean()) if len(gap) else float("nan")}

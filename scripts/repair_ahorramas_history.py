"""Repair the AhorraMás history snapshots taken before the price-extraction fix.

Before the fix (22 Nov 2026) the scraper stored the first "x,yy €" on each product card.
That number was sometimes the item price and sometimes the price per kg/L/unit.
For each old row there are only two readings of the stored number X:

    item:  the price was X
    unit:  X was the price per kg/L/unit, so the price was X * pack size

The pack size comes from the product name (today's price / today's price per unit).
Today's correctly scraped price is used ONLY to choose between the two readings:
a reading is accepted when it lands within ±TOL of today's price and the other one
does not (or both readings are the same, pack size ~1). The kept price is always
computed from the old number, so real price changes inside ±TOL survive.
Rows that fit neither reading, have no pack size, or are no longer sold are dropped.

Originals are kept as YYYY-MM-DD_raw_original.csv (ignored by the pipeline);
the repaired file replaces YYYY-MM-DD_raw.csv and gets a "repair" column.

Usage:  python -m scripts.repair_ahorramas_history
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "history" / "ahorramas"
REFERENCE = "2026-11-22"            # first snapshot taken with the fixed scraper
DATES = ["2026-10-04", "2026-11-10"]  # 17 and 20 Nov hold only the cents: not repairable
TOL = 0.20


def repair(old, now, tol=TOL):
    """Return the repaired rows of `old` with a `repair` column ("item" or "unit")."""
    ref = now.drop_duplicates("url").copy()
    ref["size"] = ref["price"] / ref["price_per_unit"]
    m = old.merge(ref[["url", "price", "size"]].rename(columns={"price": "price_now"}),
                  on="url", how="inner")

    x = pd.to_numeric(m["price"], errors="coerce")
    as_unit = x * m["size"]

    def near(v):
        return (v / m["price_now"]).between(1 - tol, 1 + tol)

    same = m["size"].between(1 - tol, 1 + tol)
    item = near(x) & (same | ~near(as_unit))
    unit = near(as_unit) & ~near(x) & ~same

    m["repair"] = None
    m.loc[item, "repair"] = "item"
    m.loc[unit, "repair"] = "unit"
    m.loc[unit, "price"] = as_unit[unit].round(2)

    out = m[m["repair"].notna()].copy()
    # The scraper's own price per unit, recomputed from the repaired price.
    out["price_per_unit"] = (out["price"] / out["size"]).round(4).where(out["size"].notna())
    return out[list(old.columns) + ["repair"]]


def main():
    now = pd.read_csv(FOLDER / f"{REFERENCE}_raw.csv", encoding="utf-8-sig")

    for day in DATES:
        raw = FOLDER / f"{day}_raw.csv"
        original = FOLDER / f"{day}_raw_original.csv"
        processed = FOLDER / f"{day}_processed.csv"

        if not original.exists():
            raw.rename(original)

        old = pd.read_csv(original, encoding="utf-8-sig")
        fixed = repair(old, now)
        fixed.to_csv(raw, index=False, encoding="utf-8-sig")

        # The pipeline never overwrites a processed snapshot: remove it so it is rebuilt.
        processed.unlink(missing_ok=True)

        counts = fixed["repair"].value_counts().to_dict()
        print(f"{day}: kept {len(fixed)} of {len(old)} rows {counts}")


if __name__ == "__main__":
    main()

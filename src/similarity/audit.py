"""Task 9 Step 5: the blind human audit of the matching engine and of the Gemini judge.

    python -m src.similarity.audit make    # data/similarity/audit/: two spreadsheets to fill in + the hidden key
    python -m src.similarity.audit score   # after filling them: results/kappa.csv and results/audit_matches.csv

Raters see only the two products (name, brand, size, chain, link). The engine's decision, its probability, the
size tier and Gemini's label stay in audit_key.csv, which raters should not open before they finish.
"""

import sys

import numpy as np
import pandas as pd

from src.pipeline import config
from src.similarity import judge

AUDIT_DIR = config.SIM_DIR / "audit"
KEY_PATH = AUDIT_DIR / "audit_key.csv"
RATERS = {"felipe": None, "teammate": 40}  # rater → rows to label (None = all); the teammate's rows overlap Felipe's
LABELS = ["same", "equivalent", "different", "unsure"]


def _size(r):
    if pd.isna(r["unit_quantity"]) or not r["unit_quantity"]:
        return ""
    pack = f"{int(r['pack_count'])} x " if pd.notna(r["pack_count"]) and r["pack_count"] > 1 else ""
    return f"{pack}{r['unit_quantity']:g} {r['quantity_unit'] if pd.notna(r['quantity_unit']) else ''}".strip()


def _sheet(rows, path):
    """Rater spreadsheet: one row per pair, a dropdown for the label, frozen header, wide name columns."""
    from openpyxl.worksheet.datavalidation import DataValidation

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        rows.to_excel(xl, sheet_name="audit", index=False)
        ws = xl.sheets["audit"]
        dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
        col = rows.columns.get_loc("label") + 1
        letter = ws.cell(row=1, column=col).column_letter
        dv.add(f"{letter}2:{letter}{len(rows) + 1}")
        ws.add_data_validation(dv)
        ws.freeze_panes = "B2"
        widths = {"name": 55, "brand": 16, "size": 12, "chain": 11, "link": 12, "label": 13, "notes": 40}
        for i, c in enumerate(rows.columns, start=1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = next(
                (w for k, w in widths.items() if c.endswith(k)), 8)


def make(n=100, seed=config.SEED):
    from src.similarity.data import load_products

    silver = pd.read_csv(config.SIM_DIR / "silver_candidates.csv")
    llm = pd.read_csv(config.SIM_DIR / "llm_pair_labels.csv")[["key_a", "key_b", "label", "rationale"]]
    silver = silver.merge(llm.rename(columns={"label": "llm_label", "rationale": "llm_rationale"}),
                          on=["key_a", "key_b"], how="left")
    sample = judge.sample_audit(silver, n=n, seed=seed)
    sample = sample.sample(frac=1, random_state=seed + 1).reset_index(drop=True)  # hide the group order
    sample.insert(0, "audit_id", np.arange(1, len(sample) + 1))

    by_key = load_products().drop_duplicates("product_key").set_index("product_key")
    view = pd.DataFrame({"audit_id": sample["audit_id"]})
    for side, col in (("A", "key_a"), ("B", "key_b")):
        p = by_key.loc[sample[col]]
        view[f"{side}_chain"] = p["chain_id"].to_numpy()
        view[f"{side}_name"] = p["name"].to_numpy()
        view[f"{side}_brand"] = p["brand"].fillna("").to_numpy()
        view[f"{side}_size"] = [_size(r) for _, r in p.iterrows()]
        view[f"{side}_link"] = p["url"].fillna("").to_numpy()
    view["label"], view["notes"] = "", ""

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(KEY_PATH, index=False)
    rng = np.random.default_rng(seed)
    for rater, k in RATERS.items():
        rows = view if k is None else view.loc[np.sort(rng.choice(len(view), size=k, replace=False))]
        _sheet(rows, AUDIT_DIR / f"audit_{rater}.xlsx")
    print(sample.groupby(["group", "chain_a"]).size().unstack(fill_value=0).to_string())
    print("tiers:", sample["tier"].value_counts().to_dict())
    print(f"written to {AUDIT_DIR}")


def _read(rater):
    path = AUDIT_DIR / f"audit_{rater}.xlsx"
    if not path.exists():
        return None
    d = pd.read_excel(path)[["audit_id", "label"]]
    d["label"] = d["label"].astype("string").str.strip().str.lower()
    d = d[d["label"].isin(LABELS[:3])]  # blank and "unsure" rows are left out
    return d.set_index("audit_id")["label"].rename(rater)


def score():
    key = pd.read_csv(KEY_PATH).set_index("audit_id")
    raters = {r: s for r in RATERS if (s := _read(r)) is not None and len(s)}
    if not raters:
        raise SystemExit(f"no labels yet: fill {AUDIT_DIR}/audit_<rater>.xlsx first")
    series = {**raters, "gemini": key["llm_label"].rename("gemini")}
    rows = []
    names = list(series)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = pd.concat([series[a], series[b]], axis=1, join="inner").dropna()
            if len(both) >= 5:
                rows.append({"a": a, "b": b, "n": len(both), "agreement": round((both[a] == both[b]).mean(), 3),
                             "kappa": round(judge.kappa(both[a], both[b]), 3)})
    kappa = pd.DataFrame(rows)
    (config.SIM_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    kappa.to_csv(config.SIM_RESULTS_DIR / "kappa.csv", index=False)
    print(kappa.to_string(index=False))

    human = key.join(raters["felipe"] if "felipe" in raters else next(iter(raters.values())), how="inner")
    lab = human.columns[-1]
    human["match"] = human[lab].isin(["same", "equivalent"])
    human["same"] = human[lab] == "same"
    out = (human.groupby(["group", "chain_a"]).agg(n=("match", "size"), same=("same", "mean"), match=("match", "mean"))
           .round(3).reset_index())
    total = human.groupby("group").agg(n=("match", "size"), same=("same", "mean"), match=("match", "mean")).round(3)
    out = pd.concat([out, total.reset_index().assign(chain_a="all")], ignore_index=True)
    out.to_csv(config.SIM_RESULTS_DIR / "audit_matches.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "make"]()

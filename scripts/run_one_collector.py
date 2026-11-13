"""Run one RAW collector and stage its outputs for the pipeline job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from scripts.run_full_collection import COLLECTORS, RAW_NAMES, ROOT, RUN_DATE, run_one

COLLECTOR_FILES = {
    "mercadona": ["mercadona"],
    "bm_consum": ["bm", "consum"],
    "alcampo": ["alcampo"],
    "ahorramas": ["ahorramas"],
    "dia": ["dia"],
    "aldi": ["aldi"],
}
COLLECTOR_NOTEBOOKS = dict(COLLECTORS)
STAGING = ROOT / "artifact_payload"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(name: str) -> None:
    os.chdir(ROOT)
    notebook = ROOT / COLLECTOR_NOTEBOOKS[name]
    if not notebook.is_file():
        raise FileNotFoundError(f"Notebook missing: {notebook}")

    for chain in COLLECTOR_FILES[name]:
        dated = ROOT / "data" / "history" / chain / f"{RUN_DATE}_raw.csv"
        if dated.exists():
            raise FileExistsError(f"Historical snapshot already exists: {dated}")

    run_one(name, str(notebook.relative_to(ROOT)))

    if STAGING.exists():
        shutil.rmtree(STAGING)

    files = []
    for chain in COLLECTOR_FILES[name]:
        raw = ROOT / "data" / "raw" / RAW_NAMES[chain]
        dated = ROOT / "data" / "history" / chain / f"{RUN_DATE}_raw.csv"
        for path in (raw, dated):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty RAW: {path}")
            df = pd.read_csv(path, encoding="utf-8-sig")
            
            # Validate using each supermarket's original RAW schema.
            if df.empty or "name" not in df.columns:
                raise RuntimeError(f"Unexpected RAW content: {path}")

            if chain == "mercadona":
                required = ["unit_price"]

            elif chain in {"bm", "consum"}:
                required = ["regular_price", "offer_price"]

            else:
                required = ["price"]

            missing = [col for col in required if col not in df.columns]

            if missing:
                raise RuntimeError(
                    f"Missing price columns in {path}: {missing}"
                )

            if chain in {"bm", "consum"}:
                regular = pd.to_numeric(
                    df["regular_price"], errors="coerce"
                )
                offer = pd.to_numeric(
                    df["offer_price"], errors="coerce"
                )
                prices = offer.where(offer.gt(0), regular)

            else:
                price_column = (
                    "unit_price" if chain == "mercadona" else "price"
                )
                prices = pd.to_numeric(
                    df[price_column], errors="coerce"
                )

            valid_names = (
                df["name"].notna()
                & df["name"].astype(str).str.strip().ne("")
            )
            valid_rows = valid_names & prices.gt(0).fillna(False)

            if not valid_rows.any():
                raise RuntimeError(
                    f"No valid products found in {path}"
                )

            invalid_count = int((~valid_rows).sum())

            print(
                f"{chain}: {len(df)} RAW rows, "
                f"{invalid_count} rows requiring validation.",
                flush=True,
            )

            destination = STAGING / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            files.append({
                "path": path.relative_to(ROOT).as_posix(),
                "rows": len(df),
                "sha256": digest(path),
            })
        if digest(raw) != digest(dated):
            raise RuntimeError(f"Latest and historical RAW differ for {chain}")

    manifest = {"collector": name, "collection_date": RUN_DATE, "files": files}
    manifest_path = STAGING / "_collector_manifests" / f"{name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"PASS: {name}, staged {len(files)} RAW CSV files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("collector", choices=list(COLLECTOR_FILES))
    collect(parser.parse_args().collector)

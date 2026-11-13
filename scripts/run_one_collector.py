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
            if df.empty or "price" not in df.columns:
                raise RuntimeError(f"Unexpected RAW content: {path}")
            numeric = pd.to_numeric(df["price"], errors="coerce")
            if numeric.isna().any() or numeric.le(0).any():
                raise RuntimeError(f"Invalid prices in RAW: {path}")
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

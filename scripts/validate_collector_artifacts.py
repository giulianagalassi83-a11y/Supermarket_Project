"""Check six collector artifacts before rebuilding or publishing data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.run_one_collector import COLLECTOR_FILES, RAW_NAMES, ROOT

MANIFEST_DIR = ROOT / "_collector_manifests"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not MANIFEST_DIR.is_dir():
        raise RuntimeError("Collector manifests are missing")

    dates = set()
    for collector, chains in COLLECTOR_FILES.items():
        manifest_path = MANIFEST_DIR / f"{collector}.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing collector artifact: {collector}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["collector"] != collector:
            raise RuntimeError(f"Wrong collector in {manifest_path}")
        date = manifest["collection_date"]
        dates.add(date)
        expected = set()
        for chain in chains:
            expected.add(f"data/raw/{RAW_NAMES[chain]}")
            expected.add(f"data/history/{chain}/{date}_raw.csv")
        if {item["path"] for item in manifest["files"]} != expected:
            raise RuntimeError(f"Wrong RAW paths for {collector}")
        for item in manifest["files"]:
            file = ROOT / item["path"]
            if not file.is_file() or digest(file) != item["sha256"]:
                raise RuntimeError(f"Missing or modified RAW artifact: {file}")
            frame = pd.read_csv(file, encoding="utf-8-sig")
            if frame.empty or len(frame) != item["rows"]:
                raise RuntimeError(f"Row count mismatch: {file}")
        print(f"OK: {collector}, date={date}")

    if len(dates) != 1:
        raise RuntimeError(
            f"Collection dates differ: {sorted(dates)}. "
            "Start a new workflow run rather than mixing dates."
        )
    print(f"PASS: six collectors, seven supermarkets, date={next(iter(dates))}")


if __name__ == "__main__":
    main()


"""Verify five collector artifacts and integrate six fresh RAW datasets.

DIA remains in the pipeline using its previously saved RAW and history.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.run_one_collector import (
    COLLECTOR_FILES,
    RAW_NAMES,
    ROOT,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ACTIVE = {
    key: chains
    for key, chains in COLLECTOR_FILES.items()
    if key != "dia"
}

DOWNLOAD = ROOT / "downloaded_collectors"
MANIFEST_DIR = DOWNLOAD / "_collector_manifests"


# --------------------------------------------------
# FILE INTEGRITY
# --------------------------------------------------

def digest(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


# --------------------------------------------------
# VALIDATE AND INTEGRATE COLLECTOR OUTPUTS
# --------------------------------------------------

def main() -> None:

    if not MANIFEST_DIR.is_dir():
        raise RuntimeError(
            "Collector manifests are missing"
        )

    available_collectors = {
        p.stem
        for p in MANIFEST_DIR.glob("*.json")
    }

    if available_collectors != set(ACTIVE):
        raise RuntimeError(
            "Expected exactly five active collector manifests"
        )

    # Preserve DIA's previous RAW dataset.
    dia = ROOT / "data/raw" / RAW_NAMES["dia"]

    if (
        not dia.is_file()
        or pd.read_csv(
            dia, encoding="utf-8-sig"
        ).empty
    ):
        raise RuntimeError(
            "Previous DIA RAW is missing or empty"
        )

    dates = set()
    verified = []

    for collector, chains in ACTIVE.items():

        manifest_path = (
            MANIFEST_DIR / f"{collector}.json"
        )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if manifest["collector"] != collector:
            raise RuntimeError(
                f"Wrong collector: {collector}"
            )

        day = manifest["collection_date"]

        if date.fromisoformat(day).isoformat() != day:
            raise RuntimeError(
                f"Invalid collection date: {day}"
            )

        dates.add(day)

        expected = set()

        for chain in chains:
            expected.add(
                f"data/raw/{RAW_NAMES[chain]}"
            )

            expected.add(
                f"data/history/{chain}/{day}_raw.csv"
            )

        items = manifest["files"]

        if (
            len(items) != len(expected)
            or {item["path"] for item in items} != expected
        ):
            raise RuntimeError(
                f"Wrong RAW paths for {collector}"
            )

        for item in items:

            relative = Path(item["path"])
            source = DOWNLOAD / relative
            target = ROOT / relative

            if (
                not source.is_file()
                or digest(source) != item["sha256"]
            ):
                raise RuntimeError(
                    f"Missing or modified artifact: {source}"
                )

            frame = pd.read_csv(
                source,
                encoding="utf-8-sig"
            )

            if frame.empty or len(frame) != item["rows"]:
                raise RuntimeError(
                    f"Row count mismatch: {source}"
                )

            # Never overwrite an existing historical snapshot.
            if (
                relative.parts[1] == "history"
                and target.exists()
            ):
                raise FileExistsError(
                    f"Historical RAW already exists: {target}"
                )

            verified.append((source, target))

        print(f"OK: {collector}, date={day}")

    # All automated supermarkets must share one collection date.
    if len(dates) != 1:
        raise RuntimeError(
            f"Collection dates differ: {sorted(dates)}"
        )

    # Integrate files only after validating every collector.
    for source, target in verified:

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if target.relative_to(ROOT).parts[1] == "history":

            with source.open("rb") as src:
                with target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)

        else:
            shutil.copy2(source, target)

    print(
        "PASS: five collectors, six fresh supermarkets, "
        f"date={next(iter(dates))}; DIA preserved"
    )


if __name__ == "__main__":
    main()

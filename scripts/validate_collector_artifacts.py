
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
# PRICE SANITY: compare with the previous snapshot
# --------------------------------------------------

PRICE_COLUMN = {"mercadona": "unit_price", "bm": "regular_price", "consum": "regular_price"}
MAX_MOVED_SHARE = 0.20  # normal weeks: at most 3 % of products move more than 1.5x


def moved_share(chain: str, day: str, frame: pd.DataFrame, history: Path):
    """Share of products whose price moved more than 1.5x since the previous good snapshot."""
    from src.pipeline.config import EXCLUDED_SNAPSHOTS

    previous = sorted(
        p for p in (history / chain).glob("*_raw.csv")
        if p.name[:10] < day and (chain, p.name[:10]) not in EXCLUDED_SNAPSHOTS
    )
    if not previous:
        return None
    col = PRICE_COLUMN.get(chain, "price")
    key = "product_id" if frame["product_id"].notna().mean() > 0.5 else "url"

    def prices(df):
        return pd.to_numeric(df.set_index(df[key].astype(str))[col], errors="coerce").dropna().groupby(level=0).first()

    new, old = prices(frame), prices(pd.read_csv(previous[-1], encoding="utf-8-sig"))
    shared = new.index.intersection(old.index)
    if len(shared) < 30:
        return None
    ratio = new[shared] / old[shared]
    return float(((ratio > 1.5) | (ratio < 1 / 1.5)).mean())


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

            # A site markup change usually breaks every price at once.
            if relative.parts[1] == "history":
                share = moved_share(
                    relative.parts[2], day, frame,
                    ROOT / "data" / "history"
                )
                if share is not None and share > MAX_MOVED_SHARE:
                    raise RuntimeError(
                        f"{share:.0%} of {relative.parts[2]} prices moved "
                        "more than 1.5x since the last snapshot; "
                        "check the scraper"
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

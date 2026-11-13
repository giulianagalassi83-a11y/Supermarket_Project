"""Run the six RAW-only notebooks (seven supermarket chains) in GitHub Actions.

No notebook source files are modified. The checked-out repository is the storage
location; the workflow commits data only after the whole pipeline succeeds.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = datetime.now(ZoneInfo("Europe/Madrid")).date().isoformat()

# One notebook collects both BM and Consum.
COLLECTORS = [
    ("mercadona", "api_clients/mercadona_api.ipynb"),
    ("bm_consum", "api_clients/bm_consum_api.ipynb"),
    ("alcampo", "scrapers/alcampo.ipynb"),
    ("ahorramas", "scrapers/ahorramas.ipynb"),
    ("dia", "scrapers/dia.ipynb"),
    ("aldi", "scrapers/aldi.ipynb"),
]
RAW_NAMES = {
    "mercadona": "mercadona_api_products_raw.csv",
    "bm": "bm_products_raw.csv",
    "consum": "consum_products_raw.csv",
    "alcampo": "alcampo_products_raw.csv",
    "ahorramas": "ahorramas_products_raw.csv",
    "dia": "dia_products_raw.csv",
    "aldi": "aldi_products_raw.csv",
}


def preflight() -> None:
    """Avoid accidentally overwriting a previous snapshot on the same date."""
    for name, notebook_rel in COLLECTORS:
        if not (ROOT / notebook_rel).is_file():
            raise FileNotFoundError(f"Notebook not found: {notebook_rel}")
    existing = [
        str(ROOT / "data" / "history" / chain / f"{RUN_DATE}_raw.csv")
        for chain in RAW_NAMES
        if (ROOT / "data" / "history" / chain / f"{RUN_DATE}_raw.csv").exists()
    ]
    if existing:
        raise FileExistsError(
            "Today's historical RAW snapshot already exists for one or more chains. "
            "Run on a different collection date or review the files manually.\n"
            + "\n".join(existing)
        )


def configure_notebook(notebook: nbformat.NotebookNode, kind: str) -> None:
    """Force full collection and headless Chrome in memory, not in the .ipynb."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    old_outputs = [x for x in ("PROCESSED_DIR", "PROCESSED_PATH", "PROCESSED_HISTORY_PATH")
                   if re.search(rf"\b{x}\b", all_code)]
    if old_outputs:
        raise RuntimeError(f"{kind}: notebook still references old PROCESSED paths: {old_outputs}")

    test_setting_found = False
    headless_found = False
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        # Capture only an assignment on its own line; never edit the original file.
        source, count = re.subn(
            r"(?m)^(\s*TEST_MODE\s*=\s*)(?:True|False)(\s*(?:#.*)?)$",
            r"\g<1>False\g<2>",
            cell.source,
        )
        if count:
            test_setting_found = True
        if kind in {"alcampo", "ahorramas", "dia", "aldi"}:
            source, n = re.subn(
                r"(?m)^(\s*)options\s*=\s*Options\(\)\s*$",
                r'\g<1>options = Options()\n\g<1>options.add_argument("--headless=new")\n'
                r'\g<1>options.add_argument("--no-sandbox")\n'
                r'\g<1>options.add_argument("--disable-dev-shm-usage")',
                source,
            )
            headless_found = headless_found or n > 0
        cell.source = source

    if not test_setting_found:
        raise RuntimeError(f"{kind}: TEST_MODE assignment not found")
    if kind in {"alcampo", "ahorramas", "dia", "aldi"} and not headless_found:
        raise RuntimeError(f"{kind}: Chrome Options() initialization not found")

    # Validate the complete search summary immediately before any notebook save.
    # No partial collection should be published by an unattended weekly run.
    expected_statuses = {
        "mercadona": '{"OK"}',
        "bm_consum": '{"OK", "OK (500 cap)"}',
        "alcampo": '{"stable"}',
        "ahorramas": '{"complete"}',
        "dia": '{"OK"}',
        "aldi": '{"OK"}',
    }
    if kind == "mercadona":
        expected_len = "len(subcategories)"
    elif kind == "bm_consum":
        expected_len = "len(CHAINS) * len(SEARCH_TERMS)"
    else:
        expected_len = "len(SEARCH_TERMS)"

    check = f'''
# GitHub Actions: fail before saving if collection is incomplete.
print(df_summary.to_string(index=False), flush=True)
if df_summary.empty or len(df_summary) != {expected_len}:
    raise RuntimeError("{kind}: not all expected searches/categories were recorded")
if not df_summary["status"].isin({expected_statuses[kind]}).all():
    raise RuntimeError("{kind}: one or more searches did not finish successfully")
'''
    if kind not in {"mercadona", "bm_consum"}:
        check += '''
if "rows_extracted" in df_summary.columns and not df_summary["rows_extracted"].gt(0).all():
    raise RuntimeError("Some searches returned zero products")
'''
    # Append as the last cell *before* notebook save, rather than after.
    # Notebook save is the final code cell in each RAW-only version.
    code_positions = [i for i, c in enumerate(notebook.cells) if c.cell_type == "code"]
    save_position = code_positions[-1]
    save_source = notebook.cells[save_position].source
    if "TEST_MODE" not in save_source or "RAW" not in save_source:
        raise RuntimeError(f"{kind}: unexpected last save cell; inspect the notebook")
    notebook.cells.insert(save_position, nbformat.v4.new_code_cell(check))


def run_one(kind: str, relative_path: str) -> None:
    path = ROOT / relative_path
    print(f"\n===== {kind.upper()}: {relative_path} =====", flush=True)
    notebook = nbformat.read(path, as_version=4)
    configure_notebook(notebook, kind)
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=1800,
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )

    try:
        executed = client.execute()
    except Exception:
        print(
            f"\nFAILED: {kind}. Printing notebook diagnostics:",
            flush=True,
        )
        for cell in client.nb.cells:
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    print(output.get("text", ""), end="", flush=True)
                elif output.output_type == "error":
                    print(
                        "\n".join(output.get("traceback", [])),
                        flush=True,
                    )
        raise
    else:
        for cell in executed.cells:
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    print(output.get("text", ""), end="", flush=True)

    print(f"\nPASS: {kind}", flush=True)


def verify_outputs() -> None:
    import pandas as pd

    for chain, raw_file in RAW_NAMES.items():
        current = ROOT / "data" / "raw" / raw_file
        dated = ROOT / "data" / "history" / chain / f"{RUN_DATE}_raw.csv"
        for path in (current, dated):
            if not path.exists():
                raise FileNotFoundError(f"Expected output missing: {path}")
            frame = pd.read_csv(path, encoding="utf-8-sig", nrows=10)
            if frame.empty:
                raise ValueError(f"Empty RAW output: {path}")
        print(f"OUTPUT OK: {chain}: {current.name} + {dated.name}", flush=True)


def main() -> int:
    os.chdir(ROOT)
    print(f"Project: {ROOT}\nMadrid collection date: {RUN_DATE}", flush=True)
    preflight()
    for kind, path in COLLECTORS:
        run_one(kind, path)
    verify_outputs()
    print("\nPASS: all seven chains generated current and historical RAW files.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

"""Run the API notebooks in TEST_MODE on GitHub Actions.

This is a smoke test, not a production collection. It does not publish
RAW, processed, historical, database, or dashboard files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "api_clients" / "mercadona_api.ipynb",
    ROOT / "api_clients" / "bm_consum_api.ipynb",
]


def force_test_mode(notebook: nbformat.NotebookNode, filename: str) -> None:
    """Override checked-in notebook settings in memory, never on disk."""
    found = False
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source, count = re.subn(
            r"(?m)^(\s*TEST_MODE\s*=\s*)(?:True|False)(\s*(?:#.*)?)$",
            r"\g<1>True\g<2>",
            cell.source,
        )
        if count:
            found = True
            if filename == "mercadona_api.ipynb":
                source, n = re.subn(
                    r"(?m)^(\s*TEST_MAX_CATEGORIES\s*=\s*)\d+(\s*(?:#.*)?)$",
                    r"\g<1>2\g<2>",
                    source,
                )
                if n != 1:
                    raise RuntimeError("Mercadona: expected TEST_MAX_CATEGORIES setting")
            if filename == "bm_consum_api.ipynb":
                source, n = re.subn(
                    r'(?m)^(\s*TEST_TERMS\s*=\s*)\[[^\n]*\](\s*(?:#.*)?)$',
                    r'\g<1>["leche"]\g<2>',
                    source,
                )
                if n != 1:
                    raise RuntimeError("BM/Consum: expected TEST_TERMS setting")
            cell.source = source
            break
    if not found:
        raise RuntimeError(f"{filename}: TEST_MODE setting not found")


def assert_no_production_processing(notebook: nbformat.NotebookNode, filename: str) -> None:
    """Catch the old notebook versions before making external API requests."""
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    forbidden = ("PROCESSED_DIR", "PROCESSED_PATH", "PROCESSED_HISTORY_PATH")
    present = [name for name in forbidden if re.search(rf"\b{name}\b", code)]
    if present:
        raise RuntimeError(
            f"{filename} still has old processed-file configuration: {present}. "
            "Commit the RAW-only notebook version before running Actions."
        )


def run_notebook(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Notebook missing: {path}")

    notebook = nbformat.read(path, as_version=4)
    assert_no_production_processing(notebook, path.name)
    force_test_mode(notebook, path.name)

    print(f"\n===== Running {path.relative_to(ROOT)} (TEST_MODE=True) =====", flush=True)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    executed = client.execute()
    for cell in executed.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.get("text", ""), end="", flush=True)
    print(f"\n===== PASS: {path.name} =====", flush=True)


def main() -> int:
    for path in NOTEBOOKS:
        run_notebook(path)
    print("\nPASS: Mercadona and BM/Consum notebook smoke tests.")
    print("No output files were uploaded, committed, or published.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

#!/usr/bin/env python3
"""Clear outputs from all notebooks under tutorial/notebooks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

NOTEBOOK_ROOT = Path(__file__).resolve().parent / "notebooks"


def _clear_notebook_outputs(nb: nbformat.NotebookNode) -> bool:
    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    return changed


def _process_notebook(path: Path, write: bool) -> bool:
    raw_text = path.read_text(encoding="utf-8")
    nb = nbformat.reads(raw_text, as_version=nbformat.NO_CONVERT)
    changed = _clear_notebook_outputs(nb)
    if not changed:
        return False
    cleaned_text = nbformat.writes(nb, version=nbformat.NO_CONVERT)
    if write:
        path.write_text(cleaned_text, encoding="utf-8")
    return raw_text != cleaned_text


def find_notebooks(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(nb for nb in root.rglob("*.ipynb") if nb.is_file())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report notebooks that would change; do not rewrite them.",
    )
    args = parser.parse_args(argv)

    notebooks = find_notebooks(NOTEBOOK_ROOT)
    if not notebooks:
        print(f"No notebooks found under {NOTEBOOK_ROOT}")
        return 0

    dirty = []
    for nb_path in notebooks:
        try:
            modified = _process_notebook(nb_path, write=not args.check)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to process {nb_path}: {exc}", file=sys.stderr)
            return 1
        if modified:
            dirty.append(nb_path)

    if args.check:
        if dirty:
            for nb in dirty:
                print(nb)
            return 1
        return 0

    if dirty:
        for nb in dirty:
            print(f"Cleared outputs in {nb}")
    else:
        print("All notebooks already clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

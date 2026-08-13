"""Shared bits of the plot/table scripts.

Every script reads exactly one results/*.csv, writes into plots/ or
tables/, and exits 0 with a note when its input is not assembled yet --
that is the missing-tolerance contract, enforced here so no script can
forget it. --results/--out exist so tests can point a script at synthetic
data without touching the real results/ directory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
# The scripts run as plain `python plot_x.py`, so the shared harness
# (cinm_experiments) is not importable until its parent is on the path.
sys.path.insert(0, str(HERE.parent))


def geomean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x))))


def parse_dirs(kind: str) -> tuple[pathlib.Path, pathlib.Path]:
    """(results_dir, out_dir) from the standard CLI; kind is "plots" or
    "tables" and only picks the default output directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=pathlib.Path, default=HERE / "results")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / kind)
    args = parser.parse_args()
    return args.results, args.out


def load_or_skip(results_dir: pathlib.Path, name: str) -> pd.DataFrame | None:
    """The input CSV, or None after printing the standard skip note. Exit 0
    on None is the caller's (one-line) responsibility."""
    path = results_dir / name
    if not path.exists():
        print(f"{name} not assembled yet -- nothing to draw (looked in {path})")
        return None
    return pd.read_csv(path)


def tex(s) -> str:
    """Escape a data string (benchmark/fn/config names are full of
    underscores) for use in a tabular cell."""
    return str(s).replace("_", "\\_")


def write_tex(out_dir: pathlib.Path, name: str, lines: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def save_fig(fig, out_dir: pathlib.Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")

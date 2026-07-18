"""Experiment-specific analysis/plotting for the CINM 1.0 vs CINM 2.0
comparison. Kept separate from experiment.py so the pipeline definition and
the figure-making code don't tangle."""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def geomean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x))))


def print_summary(comparison: pd.DataFrame) -> None:
    print(f"\n{len(comparison)} matched (benchmark, dpus, tasklets) configs\n")
    print(f"{'benchmark':20s} {'n_pairs':>8s} {'geomean speedup':>18s}")
    for fn_name, sub in comparison.groupby("fn_name"):
        print(f"{fn_name:20s} {len(sub):8d} {geomean(sub['speedup']):18.3f}")
    print(f"\n{'OVERALL':20s} {len(comparison):8d} {geomean(comparison['speedup']):18.3f}")


def plot_speedup(comparison: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path:
    """Bar chart of geomean speedup per benchmark, with whiskers showing the
    CINM 2.0 25th-75th percentile spread across its BO search seeds."""
    funcs = sorted(comparison["fn_name"].unique())
    means, lo, hi = [], [], []
    for f in funcs:
        sub = comparison[comparison.fn_name == f]
        means.append(geomean(sub["speedup"]))
        lo.append(geomean(sub["cinm1_ms"] / sub["cinm2_p75"]))
        hi.append(geomean(sub["cinm1_ms"] / sub["cinm2_p25"]))

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(funcs)), 5))
    x = np.arange(len(funcs))
    colors = [plt.get_cmap("Dark2")(i % 8) for i in range(len(funcs))]
    err_lo = np.array(means) - np.array(lo)
    err_hi = np.array(hi) - np.array(means)
    ax.bar(x, means, color=colors, yerr=[err_lo, err_hi], capsize=4)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(funcs, rotation=45, ha="right")
    ax.set_ylabel("Geomean speedup, CINM 2.0 vs CINM 1.0\n(net time, same dpus/tasklets)")
    ax.set_title(
        "CINM 2.0 vs CINM 1.0 codegen at matched hardware configs\n"
        "(bars: CINM2 seed-median; whiskers: CINM2 25th-75th pct across seeds)"
    )
    fig.tight_layout()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cinm1_vs_cinm2_speedup.{ext}", dpi=200)
    print(f"\nWrote {out_dir / 'cinm1_vs_cinm2_speedup.pdf'}")
    return out_dir / "cinm1_vs_cinm2_speedup.pdf"

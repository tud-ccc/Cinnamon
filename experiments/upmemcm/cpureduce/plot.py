#!/usr/bin/env python3
"""
Plot reduce microbenchmark results: time_iter_ns vs M.

Usage:
  python3 plot.py                          # reads output/results.csv, writes output/cpureduce.png
  python3 plot.py --csv my.csv --out fig.png
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


def elem_label(val, _):
    if val >= 1024 * 1024:
        return f"{val / (1024 * 1024):.0f}M"
    if val >= 1024:
        return f"{val / 1024:.0f}K"
    return f"{val:.0f}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--csv",
        default="output/results.csv",
        help="Input CSV (default: output/results.csv)",
    )
    p.add_argument(
        "--out",
        default="output/cpureduce.png",
        help="Output PNG (default: output/cpureduce.png)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df["M"], df["time_iter_ns"], s=6, alpha=0.25, color="steelblue")
    med = df.groupby("M")["time_iter_ns"].median().sort_index()
    ax.plot(
        med.index,
        med.values,
        color="steelblue",
        marker="o",
        markersize=4,
        linewidth=1.4,
        label="median",
    )

    # ── regression: time = a · M^b (power law, linear in log-log) ───────────
    log2_x = np.log2(med.index.to_numpy(dtype=float))
    log2_y = np.log2(med.values.astype(float))
    b, log2_a = np.polyfit(log2_x, log2_y, 1)
    a = 2.0**log2_a
    x_fit = np.geomspace(med.index.min(), med.index.max(), 200)
    y_fit = a * x_fit**b
    ax.plot(x_fit, y_fit, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    formula = f"t = {a:.2f} · M^{b:.3f} ns"
    print(f"  {formula}")
    ax.text(
        0.03,
        0.97,
        formula,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(elem_label))
    ax.set_xlabel("M  (array size in i32 elements)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    ax.set_ylabel("time_iter_ns  (ns per loop iteration)")
    ax.set_title("Reduce iteration time vs array size  (dashed = a·M^b fit)")

    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

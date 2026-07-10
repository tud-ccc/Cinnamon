#!/usr/bin/env python3
"""
Plot memcpy microbenchmark results: time_iter_ns and effective bandwidth vs T.

Usage:
  python3 plot.py                          # reads output/results.csv, writes output/cpucpy.png
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


def bytes_label(val, _):
    """Format a bytes value as KB/MB for axis tick labels."""
    if val >= 1024 * 1024:
        return f"{val / (1024*1024):.0f}M"
    if val >= 1024:
        return f"{val / 1024:.0f}K"
    return f"{val:.0f}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="output/results.csv",
                   help="Input CSV (default: output/results.csv)")
    p.add_argument("--out", default="output/cpucpy.png",
                   help="Output PNG (default: output/cpucpy.png)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # bytes copied per iteration: T i32 elements = T * 4 bytes
    df["copy_bytes"] = df["T"] * 4
    m_values = sorted(df["M"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(m_values)))

    fig, ax_time = plt.subplots(figsize=(7, 5))

    fit_lines = []
    for m, color in zip(m_values, colors):
        sub = df[df["M"] == m]
        label = f"M={m // (1024*1024)}M" if m >= 1024*1024 else f"M={m}"

        # ── time_iter_ns vs T ─────────────────────────────────────────────
        ax_time.scatter(sub["copy_bytes"], sub["time_iter_ns"],
                        s=6, alpha=0.25, color=color)
        med = sub.groupby("copy_bytes")["time_iter_ns"].median().sort_index()
        ax_time.plot(med.index, med.values, color=color, marker="o",
                     markersize=4, linewidth=1.4, label=label)

        # ── regression: time = a · size^b (power law, linear in log-log) ───
        log2_x = np.log2(med.index.to_numpy(dtype=float))
        log2_y = np.log2(med.values.astype(float))
        b, log2_a = np.polyfit(log2_x, log2_y, 1)
        a = 2.0 ** log2_a
        x_fit = np.geomspace(med.index.min(), med.index.max(), 200)
        y_fit = a * x_fit ** b
        ax_time.plot(x_fit, y_fit, color="red", linestyle="--", linewidth=1.0,
                     alpha=0.8)
        y0_fit = 0.63 * x_fit ** 0.907
        ax_time.plot(x_fit, y0_fit, color="green", linestyle="--", linewidth=1.0,
                     alpha=0.8)
        formula = f"{label}: t = {a:.2f} · size^{b:.3f} ns"
        fit_lines.append((formula, color))
        print(f"  {formula}")

    annotation = "\n".join(f for f, _ in fit_lines)
    ax_time.text(0.03, 0.97, annotation, transform=ax_time.transAxes,
                 fontsize=7.5, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax_time.set_xscale("log", base=2)
    ax_time.xaxis.set_major_formatter(FuncFormatter(bytes_label))
    ax_time.set_xlabel("Copy size per iteration (bytes)  [= T × 4]")
    ax_time.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_time.legend()
    ax_time.set_yscale("linear")
    ax_time.set_ylabel("time_iter_ns  (ns per loop iteration)")
    ax_time.set_title("Iteration time vs copy size  (dashed = a·size^b fit)")

    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

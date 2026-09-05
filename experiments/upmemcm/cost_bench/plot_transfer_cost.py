#!/usr/bin/env python3
"""Plot gather / scatter predicted vs measured transfer cost.

For each of gather and scatter:
  - reads <in-dir>/gather.csv  (or scatter.csv)
  - groups by (bytes_per_dpu, num_dpus) and averages elapsed_ns over iterations
  - computes predicted_ns from the formula below (edit the TODO line)
  - scatter: predicted_ns (x) vs elapsed_ns (y), diagonal = perfect prediction
  - points coloured by num_dpus (plasma)

Usage:
  python3 plot_transfer_cost.py --in-dir aggregated --out plots/transfer_cost.png
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from cinm_experiments import plots as shared_plots  # noqa: E402


def _group(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read CSV and average elapsed_ns over iterations."""
    df = pd.read_csv(csv_path)
    return df.groupby(["bytes_per_dpu", "num_dpus"], as_index=False)[
        "elapsed_ns"
    ].mean()


def load_gather(csv_path: pathlib.Path) -> pd.DataFrame:
    df = _group(csv_path)
    # ── TODO: gather prediction formula ───────────────────────────────────────
    df["predicted_ns"] = 10e6 * (
        0.00335152653
        + 5.4368487e-4 * df["num_dpus"]
        + 3.32992140e-5 * df["bytes_per_dpu"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    df = df.dropna(subset=["predicted_ns"])
    df = df[np.isfinite(df["predicted_ns"]) & np.isfinite(df["elapsed_ns"])]
    return df.reset_index(drop=True)


def load_scatter(csv_path: pathlib.Path) -> pd.DataFrame:
    df = _group(csv_path)
    # ── TODO: scatter prediction formula ──────────────────────────────────────
    df["predicted_ns"] = 10e6 * (
        0.09567462004205413
        + 1.43986036e-4 * df["num_dpus"]
        + 8.51439834e-6 * df["bytes_per_dpu"]
    )
    # ─────────────────────────────────────────────────────────────────────────
    df = df.dropna(subset=["predicted_ns"])
    df = df[np.isfinite(df["predicted_ns"]) & np.isfinite(df["elapsed_ns"])]
    return df.reset_index(drop=True)


def make_plot(
    df_gather: pd.DataFrame, df_scatter: pd.DataFrame, out_path: pathlib.Path
):
    all_vals = np.concatenate(
        [
            df_gather["elapsed_ns"].values,
            df_gather["predicted_ns"].values,
            df_scatter["elapsed_ns"].values,
            df_scatter["predicted_ns"].values,
        ]
    )
    pad = 1.15
    lo = all_vals.min() / pad
    hi = all_vals.max() * pad

    for df in (df_gather, df_scatter):
        df["transfersize_bytes"] = df["bytes_per_dpu"] * df["num_dpus"]

    all_dpus = np.concatenate(
        [
            df_gather["transfersize_bytes"].values,
            df_scatter["transfersize_bytes"].values,
        ]
    )
    dpu_norm = LogNorm(vmin=all_dpus.min(), vmax=all_dpus.max())
    k_min = int(np.floor(np.log2(all_dpus.min())))
    k_max = int(np.ceil(np.log2(all_dpus.max())))
    dpu_ticks = [2**i for i in range(k_min, k_max + 1)]

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    sc = None
    for ax, df, title in [
        (ax0, df_gather, "Gather"),
        (ax1, df_scatter, "Scatter"),
    ]:
        sc = shared_plots.plot_measured_vs_predicted(
            df["predicted_ns"],
            df["elapsed_ns"],
            ax=ax,
            color=df["transfersize_bytes"],
            cmap="plasma",
            norm=dpu_norm,
            xlabel="predicted (ns)",
            ylabel="measured (ns)",
            title=title,
            lim=(lo, hi),
            legend=False,
        )

    cbar = fig.colorbar(sc, cax=cax, label="Total transfer size", ticks=dpu_ticks)
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"$2^{{{int(round(np.log2(v)))}}}$" if v > 2048 else f"{v:g}"
        )
    )

    fig.suptitle("Transfer cost: predicted vs measured")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--in-dir",
        default="aggregated",
        help="Directory containing gather.csv / scatter.csv (default: aggregated)",
    )
    p.add_argument(
        "--out",
        default="plots/transfer_cost.png",
        help="Output plot path (default: plots/transfer_cost.png)",
    )
    args = p.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    df_gather = load_gather(in_dir / "gather.csv")
    df_scatter = load_scatter(in_dir / "scatter.csv")

    if df_gather.empty or df_scatter.empty:
        raise SystemExit(
            "predicted_ns is all NaN — fill in the formula in load_transfer()"
        )

    make_plot(df_gather, df_scatter, pathlib.Path(args.out))


if __name__ == "__main__":
    main()

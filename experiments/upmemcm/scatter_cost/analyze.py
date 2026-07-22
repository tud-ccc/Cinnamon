#!/usr/bin/env python3
"""
Analyze scatter_bench.cpp's results.csv (columns: num_dpus, blocks_per_dpu,
block_size, iter, ns). All latencies are converted to milliseconds up front
and reported/plotted in ms throughout.

Produces:
  - plots/latency_vs_block_size.png   (lines per num_dpus, fixed blocks_per_dpu)
  - plots/latency_vs_num_dpus.png     (lines per block_size, fixed blocks_per_dpu)
  - plots/heatmap_blocks_vs_size.png  (blocks_per_dpu x block_size, fixed num_dpus)
  - plots/regression_fit.png          (measured vs. predicted, best template)
  - a regression table (stdout): several feature-transform templates fit by
    OLS and compared by RMSE/R^2, e.g. raw dims vs. log2(block_size) vs.
    log2(everything) vs. total_bytes -- same idea as reduce_cost's
    fit_overhead_term.py TEMPLATES dict, simplified to plain OLS.

Usage:
  python3 analyze.py results.csv
  python3 analyze.py results.csv --out-dir plots --blocks-per-dpu 24 --num-dpus 2048
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

# ── Regression templates: name -> feature function (df -> [n, k] matrix) ────

TEMPLATES = {
    "linear (dpus, blocks, size)":
        lambda d: np.column_stack([d.num_dpus, d.blocks_per_dpu, d.block_size]),
    "log2(size)":
        lambda d: np.column_stack([d.num_dpus, d.blocks_per_dpu, np.log2(d.block_size)]),
    "log2(dpus), log2(size)":
        lambda d: np.column_stack([np.log2(d.num_dpus), d.blocks_per_dpu, np.log2(d.block_size)]),
    "log2(dpus, blocks, size)":
        lambda d: np.column_stack([np.log2(d.num_dpus), np.log2(d.blocks_per_dpu), np.log2(d.block_size)]),
    "total_bytes":
        lambda d: np.column_stack([d.num_dpus * d.blocks_per_dpu * d.block_size]),
    "log2(total_bytes)":
        lambda d: np.column_stack([np.log2(d.num_dpus * d.blocks_per_dpu * d.block_size)]),
    "log2(dpus), log2(bytes_per_dpu)":
        lambda d: np.column_stack([np.log2(d.num_dpus), np.log2(d.blocks_per_dpu * d.block_size)]),
}


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS with intercept; returns pred, coef, intercept, rmse, r2."""
    Xi = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    pred = Xi @ coef
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"pred": pred, "intercept": float(coef[0]), "coef": coef[1:],
             "rmse": rmse, "r2": r2}


def fit_all_templates(data: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    fits = {}
    for name, feature_fn in TEMPLATES.items():
        X = feature_fn(data)
        fit = fit_ols(X, y)
        fits[name] = fit
        rows.append({"template": name, "rmse_ms": fit["rmse"], "r2": fit["r2"]})
    table = pd.DataFrame(rows).sort_values("rmse_ms").reset_index(drop=True)
    return table, fits


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_latency_vs_block_size(agg: pd.DataFrame, blocks_per_dpu: int, out_path: pathlib.Path):
    sub = agg[agg["blocks_per_dpu"] == blocks_per_dpu]
    if sub.empty:
        return
    dpus = sorted(sub["num_dpus"].unique())
    norm = LogNorm(vmin=min(dpus), vmax=max(dpus))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in dpus:
        s = sub[sub["num_dpus"] == n].sort_values("block_size")
        ax.plot(s["block_size"], s["ms"], marker="o", markersize=3,
                linewidth=1.5, color=cmap(norm(n)))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("block size (bytes)")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Scatter latency vs. block size (blocks_per_dpu={blocks_per_dpu})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, label="Number of DPUs")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_num_dpus(agg: pd.DataFrame, blocks_per_dpu: int, out_path: pathlib.Path):
    sub = agg[agg["blocks_per_dpu"] == blocks_per_dpu]
    if sub.empty:
        return
    sizes = sorted(sub["block_size"].unique())
    norm = LogNorm(vmin=min(sizes), vmax=max(sizes))
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(7, 5))
    for sz in sizes:
        s = sub[sub["block_size"] == sz].sort_values("num_dpus")
        ax.plot(s["num_dpus"], s["ms"], marker="o", markersize=3,
                linewidth=1.5, color=cmap(norm(sz)))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of DPUs")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Scatter latency vs. number of DPUs (blocks_per_dpu={blocks_per_dpu})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, label="block size (bytes)")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(agg: pd.DataFrame, num_dpus: int, out_path: pathlib.Path):
    sub = agg[agg["num_dpus"] == num_dpus]
    if sub.empty:
        return
    pivot = sub.pivot(index="blocks_per_dpu", columns="block_size", values="ms")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                        norm=LogNorm(vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values)),
                        cmap="viridis", shading="nearest")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block size (bytes)")
    ax.set_ylabel("blocks per DPU")
    ax.set_title(f"Scatter latency (ms), num_dpus={num_dpus}")
    fig.colorbar(im, ax=ax, label="latency (ms)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_regression_fits(data: pd.DataFrame, y: np.ndarray, fits: dict,
                             table: pd.DataFrame, best_name: str, out_path: pathlib.Path):
    """One square measured-vs-predicted subplot per regression template, 2
    per row. The best template's title is red. Every subplot in a row shares
    the same color scale (num_dpus), so only one colorbar is drawn per row
    (at the row's right edge) instead of one per subplot."""
    names = list(table["template"])
    ncols = 2
    nrows = (len(names) + ncols - 1) // ncols

    norm = LogNorm(vmin=data.num_dpus.min(), vmax=data.num_dpus.max())
    cmap = plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Powers of 2 spanning the data range, for the colorbar ticks.
    k_min = int(np.floor(np.log2(data.num_dpus.min())))
    k_max = int(np.ceil(np.log2(data.num_dpus.max())))
    dpu_ticks = [2 ** k for k in range(k_min, k_max + 1)]

    # Shared axis limits across every subplot for direct comparability.
    all_pred = np.concatenate([fits[nm]["pred"] for nm in names])
    pad = 1.15
    lo = min(all_pred.min(), y.min()) / pad
    hi = max(all_pred.max(), y.max()) * pad

    # constrained_layout (not tight_layout) -- tight_layout mishandles a
    # manually built GridSpec combined with set_aspect("equal"), silently
    # squeezing every axes down to a sliver.
    fig = plt.figure(figsize=(5.2 * ncols + 1.2, 5.2 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.06],
                          wspace=0.1, hspace=0.15)

    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        fit = fits[name]
        ax.scatter(fit["pred"], y, s=8, alpha=0.5, c=data.num_dpus, cmap=cmap, norm=norm)
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("predicted latency (ms)")
        ax.set_ylabel("measured latency (ms)")
        is_best = name == best_name
        ax.set_title(f"{name}\nR²={fit['r2']:.3f}, RMSE={fit['rmse']:.4g} ms",
                     color="red" if is_best else "black",
                     fontweight="bold" if is_best else "normal")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=8)

    for r in range(nrows):
        cax = fig.add_subplot(gs[r, ncols])
        cbar = fig.colorbar(sm, cax=cax, ticks=dpu_ticks, label="Number of DPUs")
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())

    fig.suptitle("Regression template comparison (best fit in red)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Path to results.csv from scatter_bench")
    parser.add_argument("--out-dir", default="plots", help="Output directory for plots (default: plots)")
    parser.add_argument("--blocks-per-dpu", type=int, default=None,
                        help="blocks_per_dpu to slice for the vs-block-size/vs-num-dpus plots "
                             "(default: the largest value present)")
    parser.add_argument("--num-dpus", type=int, default=None,
                        help="num_dpus to slice for the heatmap (default: the largest value present)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"{args.csv}: no rows (empty or header-only)")
    df["ms"] = df["ns"] / 1e6

    agg = (df.groupby(["num_dpus", "blocks_per_dpu", "block_size"])["ms"]
             .mean().reset_index())
    agg["total_bytes"] = agg.num_dpus * agg.blocks_per_dpu * agg.block_size

    blocks_per_dpu = args.blocks_per_dpu or int(agg["blocks_per_dpu"].max())
    num_dpus = args.num_dpus or int(agg["num_dpus"].max())
    out_dir = pathlib.Path(args.out_dir)

    plot_latency_vs_block_size(agg, blocks_per_dpu, out_dir / "latency_vs_block_size.png")
    plot_latency_vs_num_dpus(agg, blocks_per_dpu, out_dir / "latency_vs_num_dpus.png")
    plot_heatmap(agg, num_dpus, out_dir / "heatmap_blocks_vs_size.png")

    y = agg["ms"].to_numpy(dtype=float)
    table, fits = fit_all_templates(agg, y)
    print(f"=== Regression templates (n={len(agg)} aggregated configs) ===")
    print(table.to_string(index=False))

    best_name = table.iloc[0]["template"]
    plot_all_regression_fits(agg, y, fits, table, best_name, out_dir / "regression_fit.png")
    print(f"\nBest fit: {best_name}")
    print(f"  intercept = {fits[best_name]['intercept']:.4g}")
    print(f"  coef      = {list(np.round(fits[best_name]['coef'], 4))}")
    print(f"\nPlots written to {out_dir}/")


if __name__ == "__main__":
    main()

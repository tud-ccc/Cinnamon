#!/usr/bin/env python3
"""
Plot benchmark cost vs. dpus / bytes_per_dpu / total_bytes for the aggregated
CSVs (gather, scatter, launch, free, total), plus a cost-model calibration
plot (measured vs. predicted cost) when --oracle is given.

For each input CSV, plots are split per "problem" (fn_name) and written to
<out-dir>/<fn_name>/<plot_name>.png. Use --filter to only generate plots
whose name contains a given substring.

Usage:
  python3 plot_cost.py --out-dir plots
  python3 plot_cost.py --filter dpus --out-dir plots
  python3 plot_cost.py --oracle ../../data/gemv_prim_red_oracle --filter cost_calibration
"""

import argparse
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from tqdm import tqdm

X_AXES = {
    "dpus": "Number of DPUs",
    "bytes_per_dpu": "Bytes per DPU",
    "total_bytes": "Total bytes (bytes_per_dpu * dpus)",
}


# ── Cost-model calibration data prep ──────────────────────────────────────────

def compute_measured_cost(agg_dir: pathlib.Path, fn_name: str) -> pd.DataFrame:
    """Return columns [config_id, measured_cost] (ns), one row per config_id.

    For each (config_id, iteration), net_cost = total - free - alloc, i.e.
    the per-call wall-clock time with the DPU-set alloc/free session overhead
    removed (the cost model predicts kernel + transfer cost, not session
    setup/teardown). alloc/free/total each fire exactly once per iteration,
    so duplicate rows for the same (config_id, iteration) — which happen when
    runs/ was populated across multiple invocations of run_configs.py — are
    averaged rather than summed. measured_cost is the mean of net_cost over
    iterations, per config_id.
    """
    total = pd.read_csv(agg_dir / "total.csv").rename(columns={"iter": "iteration"})
    free = pd.read_csv(agg_dir / "free.csv")
    alloc = pd.read_csv(agg_dir / "alloc.csv")

    select = lambda df: df[df["fn_name"] == fn_name]

    total_g = select(total).groupby(["config_id", "iteration"])["elapsed_ns"].mean()
    free_g = select(free).groupby(["config_id", "iteration"])["elapsed_ns"].mean()
    alloc_g = select(alloc).groupby(["config_id", "iteration"])["elapsed_ns"].mean()

    joined = pd.concat({"total": total_g, "free": free_g, "alloc": alloc_g}, axis=1).dropna()
    joined["net_cost"] = joined["total"] - joined["free"] - joined["alloc"]

    return (
        joined.groupby("config_id")["net_cost"]
        .mean()
        .rename("measured_cost")
        .reset_index()
    )


def compute_n_launches(agg_dir: pathlib.Path, fn_name: str) -> pd.DataFrame:
    """Return columns [config_id, n_launches], one row per config_id.

    n_launches = mean over iterations of the number of dpu_launch calls
    recorded for that iteration (rows in launch.csv for that (config_id,
    iteration)). dpu_launch may fire multiple times per iteration (e.g. one
    launch per reduction-tree stage), so a per-launch overhead would scale
    with this count rather than being a flat per-trial constant.
    """
    launch = pd.read_csv(agg_dir / "launch.csv")
    launch = launch[launch["fn_name"] == fn_name]
    counts = launch.groupby(["config_id", "iteration"]).size()
    return (
        counts.groupby("config_id")
        .mean()
        .rename("n_launches")
        .reset_index()
    )


def compute_measured_launch_cost(agg_dir: pathlib.Path, fn_name: str) -> pd.DataFrame:
    """Return columns [config_id, measured_launch_cost] (ns), one row per config_id.

    For each (config_id, iteration), launch cost = sum of all dpu_launch call
    durations in that iteration (the kernel may launch multiple times per
    iteration, e.g. one launch per reduction-tree stage). measured_launch_cost
    is the mean of that per-iteration sum, over iterations, per config_id —
    for comparison against a cost model that predicts only the on-DPU kernel
    cost (no transfer/alloc/free overhead).
    """
    launch = pd.read_csv(agg_dir / "launch.csv")
    launch = launch[launch["fn_name"] == fn_name]
    per_iter = launch.groupby(["config_id", "iteration"])["elapsed_ns"].mean()
    return (
        per_iter.groupby("config_id")
        .mean()
        .rename("measured_launch_cost")
        .reset_index()
    )


def augment_pool(pool_csv: pathlib.Path, measured: pd.DataFrame) -> pd.DataFrame:
    """pool.csv + measured_cost + error, joined on row index == config_id."""
    pool = pd.read_csv(pool_csv)
    pool = pool.merge(measured, left_index=True, right_on="config_id", how="left")
    pool = pool.drop(columns=["config_id"])
    pool["error"] = pool["measured_cost"] - pool["cost"]
    # cost == inf marks a cost-model timeout, not a real prediction — exclude
    # those rows from the error rather than reporting a meaningless -inf.
    pool.loc[~np.isfinite(pool["cost"]), "error"] = np.nan
    return pool


def find_function_pools(oracle_dir: pathlib.Path):
    """Yield (fn_name, pool_csv_path) for each infer_{fn_name}/pool.csv in an oracle dir."""
    for sub in sorted(oracle_dir.glob("infer_*")):
        if not sub.is_dir():
            continue
        pool = sub / "pool.csv"
        if not pool.exists():
            print(f"  warning: {sub} has no pool.csv, skipping", file=sys.stderr)
            continue
        yield sub.name.removeprefix("infer_"), pool


# ── Plot workers (top-level functions so ProcessPoolExecutor can pickle them) ─

def plot_metric_vs_x(df: pd.DataFrame, metric: str, fn_name: str, xcol: str,
                      out_path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df[xcol], df["elapsed_ms"], s=8, alpha=0.35, color="steelblue", label="samples")
    medians = df.groupby(xcol)["elapsed_ms"].median().sort_index()
    ax.plot(medians.index, medians.values, color="crimson", marker="o",
            markersize=4, linewidth=1.2, label="median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_AXES.get(xcol, xcol))
    ax.set_ylabel("elapsed_ms")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title(f"{metric}: elapsed_ms vs {xcol} ({fn_name})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cost_calibration(pool: pd.DataFrame, fn_name: str, out_path: pathlib.Path):
    data = pool.dropna(subset=["measured_cost"])
    data = data[np.isfinite(data["cost"]) & np.isfinite(data["measured_cost"])]
    if data.empty:
        return

    data = data[(data['dpus'] >= 2) & (data["measured_cost"] > 10e4)]

    x = data["cost"]                 # ms -> ms
    y = data["measured_cost"] / 1e6  # ns -> ms

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(x, y, s=10, alpha=0.7, c=data["dpus"], cmap="viridis", norm=LogNorm())
    k_min = int(np.floor(np.log2(data["dpus"].min())))
    k_max = int(np.ceil(np.log2(data["dpus"].max())))
    dpu_ticks = [2 ** k for k in range(k_min, k_max + 1)]
    cbar = fig.colorbar(sc, ax=ax, label="Number of DPUs", ticks=dpu_ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted cost (ms)")
    ax.set_ylabel("measured cost (ms)")
    ax.set_title(f"{fn_name}: measured vs predicted cost")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Residual (measured - predicted) vs number of dpu_launch calls, linear
    # scales — a per-launch overhead (paid every time the array is launched,
    # not just once per trial) should show up as a trend here rather than
    # against raw dpu count.
    x2 = data["dpus"]
    residual = (y - x) / data["n_launches"]
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.scatter(x2, residual, s=10, alpha=0.5, color="darkorange", label="configs")
    trend = pd.DataFrame({"n_launches": x2, "residual": residual}) \
        .groupby("n_launches")["residual"].median().sort_index()
    ax2.plot(trend.index, trend.values, color="crimson", marker="o",
             markersize=4, linewidth=1.2, label="median")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1, label="0")
    ax2.set_xlabel("Number of DPUs")
    ax2.set_ylabel("(measured cost - predicted cost) / (n launches) (ms)")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_title(f"{fn_name}: cost residual per DPU launch vs number of DPUs")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    err_out_path = out_path.parent / "cost_calibration_error_vs_launches.png"
    fig2.savefig(err_out_path, dpi=150)
    plt.close(fig2)


def plot_launch_calibration(pool: pd.DataFrame, fn_name: str, out_path: pathlib.Path):
    data = pool.dropna(subset=["measured_launch_cost"])
    data = data[np.isfinite(data["cost"]) & np.isfinite(data["measured_launch_cost"])]
    if data.empty:
        return

    # This makes a better fit but 1. there is no way to integrate it generically into the cost model and 2. it doesn't get rid of the outliers
    # correction = lambda dpus,mramCols: 0.209862 - 6.424585947849452e-05 * mramCols + 7.869968869100642e-06 * dpus

    # correction = lambda dpus, mramCols: -2.347115 -0.001803433284655423 * dpus + 0.3805487552732298 * np.log2(dpus)
    correction = lambda a, b: 0

    # data = data[data['dpus'] >= 8]

    x = data["cost"] + correction(data["dpus"], data["mramCol"])  # ms -> ms (kernel-only oracle, no transfer cost)
    y = data["measured_launch_cost"] / 1e6  # ns -> ms

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(x, y, s=10, alpha=0.7, c=data["dpus"], cmap="viridis", norm=LogNorm())
    k_min = int(np.floor(np.log2(data["dpus"].min())))
    k_max = int(np.ceil(np.log2(data["dpus"].max())))
    dpu_ticks = [2 ** k for k in range(k_min, k_max + 1)]
    cbar = fig.colorbar(sc, ax=ax, label="Number of DPUs", ticks=dpu_ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted kernel cost (ms)")
    ax.set_ylabel("measured launch cost (ms)")
    ax.set_title(f"{fn_name}: measured vs predicted kernel cost")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Residual (measured - predicted) vs number of dpu_launch calls, linear
    # scales — a per-launch overhead (paid every time the array is launched,
    # not just once per trial) should show up as a trend here rather than
    # against raw dpu count.
    x2 = data["dpus"]
    residual = y - x 
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.scatter(x2, residual, s=10, alpha=0.5, color="darkorange", label="configs")
    trend = pd.DataFrame({"n_launches": x2, "residual": residual}) \
        .groupby("n_launches")["residual"].median().sort_index()
    ax2.plot(trend.index, trend.values, color="crimson", marker="o",
             markersize=4, linewidth=1.2, label="median")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1, label="0")
    ax2.set_xlabel("Number of DPUs")
    ax2.set_ylabel("measured cost - predicted cost (ms)")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_title(f"{fn_name}: cost residual (single DPU launch) vs number of DPUs")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    err_out_path = out_path.parent / "launch_calibration_error_vs_launches.png"
    fig2.savefig(err_out_path, dpi=150)
    plt.close(fig2)


# ── Task collection (runs in the main process) ────────────────────────────────

def collect_metric_tasks(in_dir: pathlib.Path, out_dir: pathlib.Path, name_filter: str):
    """Return [(label, func, args), ...] for every gather/scatter/launch/free/total plot."""
    tasks = []
    for csv_path in sorted(in_dir.glob("*.csv")):
        metric = csv_path.stem
        df = pd.read_csv(csv_path)
        df["elapsed_ms"] = df["elapsed_ns"] / 1e6

        has_bytes = "bytes_per_dpu" in df.columns
        if has_bytes:
            df["total_bytes"] = df["bytes_per_dpu"] * df["dpus"]
        xcols = ["dpus"] + (["bytes_per_dpu", "total_bytes"] if has_bytes else [])

        for fn_name, group in df.groupby("fn_name"):
            for xcol in xcols:
                plot_name = f"{metric}_vs_{xcol}"
                if name_filter and name_filter not in plot_name:
                    continue
                out_path = out_dir / fn_name / f"{plot_name}.png"
                tasks.append((f"{plot_name} ({fn_name})", plot_metric_vs_x,
                              (group.copy(), metric, fn_name, xcol, out_path)))
    return tasks


def collect_calibration_tasks(oracle_dir: pathlib.Path, agg_dir: pathlib.Path,
                               out_dir: pathlib.Path, pool_out_dir: pathlib.Path,
                               name_filter: Optional[str]):
    plot_name = "cost_calibration"
    if name_filter and name_filter not in plot_name:
        return []

    tasks = []
    pool_out_dir.mkdir(parents=True, exist_ok=True)
    for fn_name, pool_csv in find_function_pools(oracle_dir):
        measured = compute_measured_cost(agg_dir, fn_name)
        if measured.empty:
            print(f"  {fn_name}: no aggregated data, skipping calibration", file=sys.stderr)
            continue
        n_launches = compute_n_launches(agg_dir, fn_name)
        measured = measured.merge(n_launches, on="config_id", how="left")
        pool = augment_pool(pool_csv, measured)
        pool.to_csv(pool_out_dir / f"{fn_name}_pool.csv", index=False)

        out_path = out_dir / fn_name / f"{plot_name}.png"
        tasks.append((f"{plot_name} ({fn_name})", plot_cost_calibration,
                      (pool, fn_name, out_path)))
    return tasks


def collect_launch_calibration_tasks(kernel_oracle_dir: pathlib.Path, agg_dir: pathlib.Path,
                                      out_dir: pathlib.Path, name_filter: Optional[str]):
    plot_name = "launch_calibration"
    if name_filter and name_filter not in plot_name:
        return []

    tasks = []
    for fn_name, pool_csv in find_function_pools(kernel_oracle_dir):
        measured = compute_measured_launch_cost(agg_dir, fn_name)
        if measured.empty:
            print(f"  {fn_name}: no aggregated data, skipping launch calibration", file=sys.stderr)
            continue
        pool = pd.read_csv(pool_csv)
        pool = pool.merge(measured, left_index=True, right_on="config_id", how="left")
        pool = pool.drop(columns=["config_id"])

        out_path = out_dir / fn_name / f"{plot_name}.png"
        tasks.append((f"{plot_name} ({fn_name})", plot_launch_calibration,
                      (pool, fn_name, out_path)))


    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", default="aggregated",
                         help="Directory containing aggregated CSVs (default: aggregated)")
    parser.add_argument("--out-dir", default="plots",
                         help="Root output directory for plots (default: plots)")
    parser.add_argument("--filter", default=None,
                         help="Only generate plots whose name contains this substring, "
                              "e.g. --filter dpus or --filter cost_calibration "
                              "(default: all plots)")
    parser.add_argument("--oracle", default=None,
                         help="Path to an oracle directory, containing one "
                              "infer_{fn_name}/pool.csv subdir per problem "
                              "(same layout as run_configs.py --data). Enables the "
                              "cost_calibration plot; omit to skip it.")
    parser.add_argument("--pool-out-dir", default="pool_measured",
                         help="Where to write cost-model-augmented pool CSVs "
                              "(default: pool_measured; only used with --oracle)")
    parser.add_argument("--kernel-oracle", default=None,
                         help="Path to a second oracle directory predicting only the "
                              "on-DPU kernel cost (no transfer/alloc/free), same "
                              "infer_{fn_name}/pool.csv layout and config_id ordering "
                              "as --oracle. Enables the launch_calibration plot "
                              "(measured launch cost vs predicted kernel cost); "
                              "omit to skip it.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                         help="Parallel plotting processes (default: cpu count)")
    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)

    tasks = collect_metric_tasks(in_dir, out_dir, args.filter)
    if args.oracle:
        tasks += collect_calibration_tasks(
            pathlib.Path(args.oracle), in_dir, out_dir,
            pathlib.Path(args.pool_out_dir), args.filter)
    if args.kernel_oracle:
        tasks += collect_launch_calibration_tasks(
            pathlib.Path(args.kernel_oracle), in_dir, out_dir, args.filter)

    if not tasks:
        print("No plots matched.", file=sys.stderr)
        sys.exit(1)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(func, *fargs): label for label, func, fargs in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Plotting"):
            label = futures[fut]
            try:
                fut.result()
            except Exception as e:
                tqdm.write(f"  FAILED {label}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

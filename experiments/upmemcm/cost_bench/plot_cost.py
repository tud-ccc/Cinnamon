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
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from cinm_experiments import plots as shared_plots  # noqa: E402
from cinm_experiments import measurements  # noqa: E402

X_AXES = {
    "dpus": "Number of DPUs",
    "bytes_per_dpu": "Bytes per DPU",
    "total_bytes": "Total bytes (bytes_per_dpu * dpus)",
}


# ── Cost-model calibration data prep ──────────────────────────────────────────
#
# Aggregated CSVs (produced by cinm_experiments.aggregate.aggregate_run) tag
# every measurement row with fn_name/label plus every config.csv param column
# (dpus, mramCol, ...) -- there is no positional config_id the way there was
# under the old run_configs.py-driven layout. So a config is identified
# internally by `label` (compile_run.Config.label, unique within fn_name),
# and joined back onto an oracle pool.csv (which has no `label`, just the raw
# param columns) via those param columns -- `key_cols` below, the same
# columns compare_oracles.py's _config_key_cols identifies (everything before
# 'visited').


def _key_cols(pool: pd.DataFrame) -> list[str]:
    """Columns that form the config vector: everything before 'visited'."""
    cols = list(pool.columns)
    cut = cols.index("visited") if "visited" in cols else len(cols)
    return [c for c in cols[:cut] if c != "cost"]


def compute_measured_cost(
    agg_dir: pathlib.Path, fn_name: str, key_cols: list
) -> pd.DataFrame:
    """Return columns [*key_cols, measured_cost] (ns), one row per config.

    For each (label, iteration), net_cost = total - free - alloc, i.e. the
    per-call wall-clock time with the DPU-set alloc/free session overhead
    removed (the cost model predicts kernel + transfer cost, not session
    setup/teardown). alloc/free/total each fire exactly once per iteration,
    so duplicate rows for the same (label, iteration) are averaged rather
    than summed. measured_cost is the mean of net_cost over iterations, per
    config.
    """
    total = pd.read_csv(agg_dir / "total.csv").rename(columns={"iter": "iteration"})
    free = pd.read_csv(agg_dir / "free.csv")
    alloc = pd.read_csv(agg_dir / "alloc.csv")

    def select(df):
        return df[df["fn_name"] == fn_name]

    total_g = select(total).groupby(["label", "iteration"])["elapsed_ns"].mean()
    free_g = select(free).groupby(["label", "iteration"])["elapsed_ns"].mean()
    alloc_g = select(alloc).groupby(["label", "iteration"])["elapsed_ns"].mean()

    joined = pd.concat(
        {"total": total_g, "free": free_g, "alloc": alloc_g}, axis=1
    ).dropna()
    joined["net_cost"] = joined["total"] - joined["free"] - joined["alloc"]

    measured = (
        joined.groupby("label")["net_cost"].mean().rename("measured_cost").reset_index()
    )
    key_df = select(total).drop_duplicates("label")[["label"] + key_cols]
    return measured.merge(key_df, on="label")[key_cols + ["measured_cost"]]


def compute_n_launches(
    agg_dir: pathlib.Path, fn_name: str, key_cols: list
) -> pd.DataFrame:
    """Return columns [*key_cols, n_launches], one row per config.

    n_launches = mean over iterations of the number of dpu_launch calls
    recorded for that iteration (rows in launch.csv for that (label,
    iteration)). dpu_launch may fire multiple times per iteration (e.g. one
    launch per reduction-tree stage), so a per-launch overhead would scale
    with this count rather than being a flat per-trial constant.
    """
    launch = pd.read_csv(agg_dir / "launch.csv")
    launch = launch[launch["fn_name"] == fn_name]
    counts = launch.groupby(["label", "iteration"]).size()
    n_launches = counts.groupby("label").mean().rename("n_launches").reset_index()
    key_df = launch.drop_duplicates("label")[["label"] + key_cols]
    return n_launches.merge(key_df, on="label")[key_cols + ["n_launches"]]


def compute_measured_launch_cost(
    agg_dir: pathlib.Path, fn_name: str, key_cols: list
) -> pd.DataFrame:
    """Return columns [*key_cols, measured_launch_cost] (ns), one row per config.

    For each (label, iteration), launch cost = sum of all dpu_launch call
    durations in that iteration (the kernel may launch multiple times per
    iteration, e.g. one launch per reduction-tree stage). measured_launch_cost
    is the mean of that per-iteration sum, over iterations, per config —
    for comparison against a cost model that predicts only the on-DPU kernel
    cost (no transfer/alloc/free overhead).
    """
    launch = pd.read_csv(agg_dir / "launch.csv")
    launch = launch[launch["fn_name"] == fn_name]
    per_iter = launch.groupby(["label", "iteration"])["elapsed_ns"].mean()
    measured = (
        per_iter.groupby("label").mean().rename("measured_launch_cost").reset_index()
    )
    key_df = launch.drop_duplicates("label")[["label"] + key_cols]
    return measured.merge(key_df, on="label")[key_cols + ["measured_launch_cost"]]


def compute_fresh_predicted_cost(
    predicted_all: pd.DataFrame, fn_name: str, key_cols: list
) -> pd.DataFrame:
    """[*key_cols, fresh_cost] (ms) -- total predicted cost recomputed by
    *today's* cinm-opt binary at compile time (sum of every distinct block's
    block_total_ms across a config's ir/cost.csv, see aggregate.
    aggregate_predicted_costs), for preferring over a pool.csv's own `cost`
    column in augment_pool -- that column was computed whenever the oracle
    pool was last (re)generated, and goes stale if the cost model changes
    afterward without regenerating it."""
    df = predicted_all[predicted_all["fn_name"] == fn_name]
    if df.empty:
        return pd.DataFrame(columns=key_cols + ["fresh_cost"])
    per_block = df.drop_duplicates(["label", "block_id"])[
        ["label", "block_id", "block_total_ms"]
    ]
    fresh = (
        per_block.groupby("label")["block_total_ms"]
        .sum()
        .rename("fresh_cost")
        .reset_index()
    )
    key_df = df.drop_duplicates("label")[["label"] + key_cols]
    return fresh.merge(key_df, on="label")[key_cols + ["fresh_cost"]]


def augment_pool(
    pool: pd.DataFrame,
    measured: pd.DataFrame,
    key_cols: list,
    fresh: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """pool.csv + measured_cost + error, joined on the config's param
    columns. When `fresh` (compute_fresh_predicted_cost) is given, its
    per-config cost -- recomputed from today's compile, not whenever the
    pool was last generated -- overrides pool.csv's own `cost` column
    wherever available (the original is kept as `stale_cost`, for
    comparison); rows the pool has but that were never (re)compiled fall
    back to the pool's own value."""
    pool = pool.merge(measured, on=key_cols, how="left")
    if fresh is not None and not fresh.empty:
        pool = pool.merge(fresh, on=key_cols, how="left")
        pool["stale_cost"] = pool["cost"]
        pool["cost"] = pool["fresh_cost"].where(
            pool["fresh_cost"].notna(), pool["cost"]
        )
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


# ── Category-breakdown data prep ──────────────────────────────────────────────
#
# predicted_costs.csv (aggregate.aggregate_predicted_costs' output, one row
# per (config, block, category, cost_label)) and the per-type aggregated CSVs
# (aggregate.aggregate_run's output, one row per (config, iteration)) are each
# reshaped to a common long format [*key_cols, label, bucket, {measured,
# predicted}_ms] -- bucket in {launch, scatter, gather, copy, unaccounted},
# see measurements.PREDICTED_TO_MEASURED -- then inner-joined on
# (*key_cols, label, bucket) so every plot below works off one flat frame.

# Non-parameter columns in predicted_costs.csv (aggregate_predicted_costs);
# everything else is a config param shared with the measured-side aggregated
# CSVs (both are tagged from the same config.csv).
_PREDICTED_NON_PARAM_COLS = frozenset(
    {
        "block_id",
        "location",
        "category",
        "cost_label",
        "cost_ms",
        "block_total_ms",
        "system",
        "fn_name",
        "label",
    }
)

_BUCKET_ORDER = [
    "launch",
    "scatter:block",
    "scatter:sg",
    "scatter:bc",
    "gather",
    "copy",
    "unaccounted",
]


def predicted_key_cols(predicted_all: pd.DataFrame) -> list[str]:
    return [c for c in predicted_all.columns if c not in _PREDICTED_NON_PARAM_COLS]


def compute_predicted_breakdown_long(
    predicted_all: pd.DataFrame, fn_name: str, key_cols: list
) -> pd.DataFrame:
    """[*key_cols, label, bucket, predicted_ms], long format: every (category,
    cost_label) row in predicted_costs.csv mapped onto its measured bucket
    (measurements.predicted_bucket) and summed per (label, bucket) -- a
    config's total predicted cost in a bucket may come from more than one
    (category, cost_label), e.g. kernel's "kernel" + "launchOverhead" both
    feed "launch"."""
    df = predicted_all[predicted_all["fn_name"] == fn_name].copy()
    if df.empty:
        return pd.DataFrame(columns=key_cols + ["label", "bucket", "predicted_ms"])
    df["bucket"] = [
        measurements.predicted_bucket(c, l)
        for c, l in zip(df["category"], df["cost_label"])
    ]
    grouped = (
        df.groupby(["label", "bucket"])["cost_ms"]
        .sum()
        .reset_index()
        .rename(columns={"cost_ms": "predicted_ms"})
    )
    key_df = df.drop_duplicates("label")[["label"] + key_cols]
    return grouped.merge(key_df, on="label")[
        key_cols + ["label", "bucket", "predicted_ms"]
    ]


def _load_measured_csv(agg_dir: pathlib.Path, fn_name: str, csv_type: str):
    path = agg_dir / f"{csv_type}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "iter" in df.columns and "iteration" not in df.columns:
        df = df.rename(columns={"iter": "iteration"})
    df = df[df["fn_name"] == fn_name]
    return df if not df.empty else None


def compute_measured_breakdown_long(
    agg_dir: pathlib.Path, fn_name: str, key_cols: list
) -> pd.DataFrame:
    """[*key_cols, label, bucket, measured_ms], long format -- the
    aggregated-CSV equivalent of measurements.net_breakdown_ms (net = total -
    alloc - free, split into scatter/gather/copy/launch plus whatever's left
    over as "unaccounted"), computed once for every config via the combined
    per-measurement-type CSVs instead of directory-by-directory."""
    total = _load_measured_csv(agg_dir, fn_name, "total")
    if total is None:
        return pd.DataFrame(columns=key_cols + ["label", "bucket", "measured_ms"])

    def mean_by_iter(csv_type):
        df = _load_measured_csv(agg_dir, fn_name, csv_type)
        if df is None:
            return pd.Series(dtype=float)
        return df.groupby(["label", "iteration"])["elapsed_ns"].mean()

    def sum_by_iter(csv_type):
        df = _load_measured_csv(agg_dir, fn_name, csv_type)
        if df is None:
            return pd.Series(dtype=float)
        return df.groupby(["label", "iteration"])["elapsed_ns"].sum()

    def sum_by_iter_and_kind(csv_type) -> dict[str, pd.Series]:
        """Like sum_by_iter, but split by scatter.csv's `kind` column (see
        timers.c's XferRecord.kind) into one Series per kind ("block"/"sg"/
        "bc") instead of summing every kind into one bucket -- so scatter:sg
        can be compared against its own prediction instead of being averaged
        together with scatter:block/scatter:bc into a single "scatter"
        number that the predicted side no longer produces (see
        measurements.PREDICTED_TO_MEASURED)."""
        df = _load_measured_csv(agg_dir, fn_name, csv_type)
        if df is None or "kind" not in df.columns:
            return {}
        return {
            str(kind): group.groupby(["label", "iteration"])["elapsed_ns"].sum()
            for kind, group in df.groupby("kind")
        }

    total_g = mean_by_iter("total")
    free_g = mean_by_iter("free")
    alloc_g = mean_by_iter("alloc")
    joined = pd.concat({"total": total_g, "free": free_g, "alloc": alloc_g}, axis=1)
    joined["free"] = joined["free"].fillna(0)
    joined["alloc"] = joined["alloc"].fillna(0)
    joined = joined.dropna(subset=["total"])
    joined["net"] = joined["total"] - joined["free"] - joined["alloc"]
    net_ms = (joined.groupby("label")["net"].mean() / 1e6).rename("net_ms")

    wide = pd.DataFrame({"net_ms": net_ms})
    scatter_bucket_cols = []
    for kind, per_iter in sum_by_iter_and_kind("scatter").items():
        bucket = f"scatter:{kind}"
        scatter_bucket_cols.append(bucket)
        wide[bucket] = (
            (per_iter.groupby("label").mean() / 1e6).reindex(wide.index).fillna(0.0)
        )
    for bucket in ["scatter:block", "scatter:sg", "scatter:bc"]:
        if bucket not in wide.columns:
            wide[bucket] = 0.0
            scatter_bucket_cols.append(bucket)
    for bucket, csv_type in [
        ("gather", "gather"),
        ("copy", "copy"),
        ("launch", "launch"),
    ]:
        per_iter = sum_by_iter(csv_type)
        per_label = (
            (per_iter.groupby("label").mean() / 1e6)
            if len(per_iter)
            else pd.Series(dtype=float)
        )
        wide[bucket] = per_label.reindex(wide.index).fillna(0.0)
    wide["unaccounted"] = wide["net_ms"] - wide[
        scatter_bucket_cols + ["gather", "copy", "launch"]
    ].sum(axis=1)
    wide = wide.drop(columns=["net_ms"]).reset_index()

    long = wide.melt(
        id_vars=["label"],
        value_vars=_BUCKET_ORDER,
        var_name="bucket",
        value_name="measured_ms",
    )
    key_df = total.drop_duplicates("label")[["label"] + key_cols]
    return long.merge(key_df, on="label")[key_cols + ["label", "bucket", "measured_ms"]]


def build_breakdown_frame(
    agg_dir: pathlib.Path, predicted_all: pd.DataFrame, fn_name: str
) -> pd.DataFrame:
    """[*key_cols, label, bucket, measured_ms, predicted_ms] for one function
    -- the shared input to every plot_breakdown_* function below."""
    key_cols = predicted_key_cols(predicted_all)
    measured_long = compute_measured_breakdown_long(agg_dir, fn_name, key_cols)
    if measured_long.empty:
        return measured_long
    predicted_long = compute_predicted_breakdown_long(predicted_all, fn_name, key_cols)
    return measured_long.merge(
        predicted_long, on=key_cols + ["label", "bucket"], how="inner"
    )


# ── Plot workers (top-level functions so ProcessPoolExecutor can pickle them) ─


def plot_metric_vs_x(
    df: pd.DataFrame, metric: str, fn_name: str, xcol: str, out_path: pathlib.Path
):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(
        df[xcol], df["elapsed_ms"], s=8, alpha=0.35, color="steelblue", label="samples"
    )
    medians = df.groupby(xcol)["elapsed_ms"].median().sort_index()
    ax.plot(
        medians.index,
        medians.values,
        color="crimson",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="median",
    )
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

    data = data[(data["dpus"] >= 2) & (data["measured_cost"] > 10e4)]

    x = data["cost"]  # ms -> ms
    y = data["measured_cost"] / 1e6  # ns -> ms

    shared_plots.plot_measured_vs_predicted(
        x,
        y,
        out_path=out_path,
        color=data["dpus"],
        color_label="Number of DPUs",
        cbar_ticks=shared_plots.log2_ticks(data["dpus"]),
        xlabel="predicted cost (ms)",
        ylabel="measured cost (ms)",
        title=f"{fn_name}: measured vs predicted cost",
    )

    # Residual (measured - predicted) vs number of dpu_launch calls, linear
    # scales — a per-launch overhead (paid every time the array is launched,
    # not just once per trial) should show up as a trend here rather than
    # against raw dpu count.
    x2 = data["dpus"]
    residual = (y - x) / data["n_launches"]
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.scatter(x2, residual, s=10, alpha=0.5, color="darkorange", label="configs")
    trend = (
        pd.DataFrame({"n_launches": x2, "residual": residual})
        .groupby("n_launches")["residual"]
        .median()
        .sort_index()
    )
    ax2.plot(
        trend.index,
        trend.values,
        color="crimson",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="median",
    )
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
    def correction(a, b):
        return 0

    # data = data[data['dpus'] >= 8]

    x = data["cost"] + correction(
        data["dpus"], data["mramCol"]
    )  # ms -> ms (kernel-only oracle, no transfer cost)
    y = data["measured_launch_cost"] / 1e6  # ns -> ms

    shared_plots.plot_measured_vs_predicted(
        x,
        y,
        out_path=out_path,
        color=data["dpus"],
        color_label="Number of DPUs",
        cbar_ticks=shared_plots.log2_ticks(data["dpus"]),
        xlabel="predicted kernel cost (ms)",
        ylabel="measured launch cost (ms)",
        title=f"{fn_name}: measured vs predicted kernel cost",
    )

    # Residual (measured - predicted) vs number of dpu_launch calls, linear
    # scales — a per-launch overhead (paid every time the array is launched,
    # not just once per trial) should show up as a trend here rather than
    # against raw dpu count.
    x2 = data["dpus"]
    residual = y - x
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.scatter(x2, residual, s=10, alpha=0.5, color="darkorange", label="configs")
    trend = (
        pd.DataFrame({"n_launches": x2, "residual": residual})
        .groupby("n_launches")["residual"]
        .median()
        .sort_index()
    )
    ax2.plot(
        trend.index,
        trend.values,
        color="crimson",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="median",
    )
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


def _present_buckets(df: pd.DataFrame) -> list[str]:
    present = set(df["bucket"].unique())
    return [b for b in _BUCKET_ORDER if b in present]


def plot_breakdown_error_box(df: pd.DataFrame, fn_name: str, out_path: pathlib.Path):
    """One box per category of (measured - predicted) / measured -- the fast
    "which category is biased" check, meant to be looked at before any of the
    other, more detailed breakdown plots."""
    data = df.dropna(subset=["measured_ms", "predicted_ms"])
    data = data[data["measured_ms"] > 0]
    if data.empty:
        return
    order = _present_buckets(data)
    rel_error = (data["measured_ms"] - data["predicted_ms"]) / data["measured_ms"]
    data = data.assign(rel_error=rel_error)

    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2, 4.5))
    ax.boxplot(
        [data.loc[data["bucket"] == b, "rel_error"] for b in order],
        tick_labels=order,
        showfliers=False,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("(measured - predicted) / measured")
    ax.set_title(f"{fn_name}: per-category relative error")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_breakdown_scatter(df: pd.DataFrame, fn_name: str, out_path: pathlib.Path):
    """Faceted measured-vs-predicted scatter, one panel per category --
    per-category extension of plot_cost_calibration's single (total-cost)
    panel."""
    data = df.dropna(subset=["measured_ms", "predicted_ms"])
    data = data[(data["measured_ms"] > 0) & (data["predicted_ms"] > 0)]
    if data.empty:
        return
    order = _present_buckets(data)
    has_dpus = "dpus" in data.columns

    fig, axes = plt.subplots(
        1, len(order), figsize=(4.2 * len(order), 4.2), squeeze=False
    )
    for ax, bucket in zip(axes[0], order):
        sub = data[data["bucket"] == bucket]
        shared_plots.plot_measured_vs_predicted(
            sub["predicted_ms"],
            sub["measured_ms"],
            ax=ax,
            color=sub["dpus"] if has_dpus else None,
            log_color=True,
            xlabel="predicted (ms)",
            ylabel="measured (ms)",
            title=bucket,
            legend=False,
        )
    fig.suptitle(f"{fn_name}: measured vs predicted, per category")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration_and_breakdown(
    pool: pd.DataFrame, df: pd.DataFrame, fn_name: str, out_path: pathlib.Path
):
    """cost_calibration.png (total measured vs. predicted cost, one big
    panel) stacked over cost_breakdown_scatter.png (the same, per category)
    in a single 4x3 figure: row 0-1/col 0-1 is the calibration scatter,
    row 0-1/col 2 its colorbar, rows 2-3 are the per-category panels
    reflowed from 1x6 into 2x3 (see plot_cost_calibration/
    plot_breakdown_scatter for the two panels' data prep -- this just
    shares their drawing code, so the two stay in sync instead of
    accidentally diverging)."""
    cal_data = pool.dropna(subset=["measured_cost"])
    cal_data = cal_data[
        np.isfinite(cal_data["cost"]) & np.isfinite(cal_data["measured_cost"])
    ]
    cal_data = cal_data[(cal_data["dpus"] >= 2) & (cal_data["measured_cost"] > 10e4)]

    bd_data = df.dropna(subset=["measured_ms", "predicted_ms"])
    bd_data = bd_data[(bd_data["measured_ms"] > 0) & (bd_data["predicted_ms"] > 0)]
    if cal_data.empty or bd_data.empty:
        return
    order = _present_buckets(bd_data)[:6]
    has_dpus = "dpus" in bd_data.columns

    fig = plt.figure(figsize=(12.6, 4.2 * 2 + 4.2 * 2))
    gs = fig.add_gridspec(4, 3)

    cal_ax = fig.add_subplot(gs[0:2, 0:2])
    sc = shared_plots.plot_measured_vs_predicted(
        cal_data["cost"],
        cal_data["measured_cost"] / 1e6,
        ax=cal_ax,
        color=cal_data["dpus"],
        color_label="Number of DPUs",
        cbar_ticks=shared_plots.log2_ticks(cal_data["dpus"]),
        xlabel="predicted cost (ms)",
        ylabel="measured cost (ms)",
        title="total (calibration)",
        metrics=True,
    )
    if sc is not None:
        cbar_cell = fig.add_subplot(gs[0:2, 2])
        cbar_cell.axis("off")
        cbar_ax = cbar_cell.inset_axes([0.3, 0.05, 0.25, 0.9])
        cbar = fig.colorbar(
            sc, cax=cbar_ax, ticks=shared_plots.log2_ticks(cal_data["dpus"])
        )
        cbar.set_label("Number of DPUs")
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    ncols_bottom = 3
    for i, bucket in enumerate(order):
        row, col = divmod(i, ncols_bottom)
        ax = fig.add_subplot(gs[2 + row, col])
        sub = bd_data[bd_data["bucket"] == bucket]
        # Only label the bottom-most panel of each column (no panel below it)
        # and the leftmost column, so "predicted (ms)"/"measured (ms)" each
        # print once per column/row instead of on every one of the 6 panels.
        is_bottom = (i + ncols_bottom) >= len(order)
        is_left = col == 0
        shared_plots.plot_measured_vs_predicted(
            sub["predicted_ms"],
            sub["measured_ms"],
            ax=ax,
            color=sub["dpus"] if has_dpus else None,
            log_color=True,
            xlabel="predicted (ms)" if is_bottom else "",
            ylabel="measured (ms)" if is_left else "",
            title=bucket,
            legend=False,
            metrics=True,
        )

    fig.suptitle(f"{fn_name}: cost calibration + per-category breakdown")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_breakdown_error_heatmap(
    df: pd.DataFrame,
    fn_name: str,
    out_path: pathlib.Path,
    xdim: str = "dpus",
    ydim: str = "mramCol",
):
    """Per-category relative error over two config dims (color-coded scatter,
    not a binned heatmap -- configs aren't on a dense regular grid), diverging
    colormap centered at 0 -- surfaces regions of the space where a specific
    category's prediction breaks down."""
    data = df.dropna(subset=["measured_ms", "predicted_ms"])
    data = data[data["measured_ms"] > 0]
    if data.empty or xdim not in data.columns or ydim not in data.columns:
        return
    order = _present_buckets(data)
    rel_error = (data["measured_ms"] - data["predicted_ms"]) / data["measured_ms"]
    data = data.assign(rel_error=rel_error)
    vmax = float(data["rel_error"].abs().quantile(0.95)) or 1.0

    fig, axes = plt.subplots(
        1, len(order), figsize=(4.6 * len(order), 4.2), squeeze=False
    )
    for ax, bucket in zip(axes[0], order):
        sub = data[data["bucket"] == bucket]
        sc = ax.scatter(
            sub[xdim],
            sub[ydim],
            c=sub["rel_error"],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            s=18,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel(xdim)
        ax.set_ylabel(ydim)
        ax.set_title(bucket)
        fig.colorbar(sc, ax=ax, label="(measured - predicted) / measured")
    fig.suptitle(f"{fn_name}: per-category relative error over ({xdim}, {ydim})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_breakdown_composition(
    df: pd.DataFrame, fn_name: str, out_path: pathlib.Path, n_configs: int = 8
):
    """Paired stacked bars (predicted vs measured) for a handful of
    representative configs spread across the dpus range, each bar normalized
    to 100% -- shows where the predicted *composition* diverges from
    measured (e.g. "predicted says scatter is 80% of cost, measured says
    50%"), independent of the total cost's absolute scale, which otherwise
    spans several orders of magnitude across dpu counts and would make a
    plain ms-scale stacked bar unreadable."""
    data = df.dropna(subset=["measured_ms", "predicted_ms"])
    if data.empty or "dpus" not in data.columns:
        return
    order = _present_buckets(data)
    labels_by_dpus = data.drop_duplicates("label").sort_values("dpus")["label"].tolist()
    if len(labels_by_dpus) > n_configs:
        idx = np.linspace(0, len(labels_by_dpus) - 1, n_configs).round().astype(int)
        chosen = [labels_by_dpus[i] for i in idx]
    else:
        chosen = labels_by_dpus

    pivot_pred = data.pivot_table(
        index="label", columns="bucket", values="predicted_ms", aggfunc="sum"
    ).reindex(index=chosen, columns=order, fill_value=0.0)
    pivot_meas = data.pivot_table(
        index="label", columns="bucket", values="measured_ms", aggfunc="sum"
    ).reindex(index=chosen, columns=order, fill_value=0.0)
    # Normalize each config's bar to 100% -- a category's *share* of total
    # cost is what's comparable across configs of very different absolute
    # scale; clip(lower=...) guards against an all-zero row (e.g. a bucket
    # with genuinely 0 predicted cost everywhere) causing a 0/0 divide.
    pivot_pred = pivot_pred.div(pivot_pred.sum(axis=1).clip(lower=1e-12), axis=0) * 100
    pivot_meas = pivot_meas.div(pivot_meas.sum(axis=1).clip(lower=1e-12), axis=0) * 100
    dpus_by_label = data.drop_duplicates("label").set_index("label")["dpus"]

    colors = plt.get_cmap("tab10").colors
    x = np.arange(len(chosen))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(chosen) * 1.3), 5))
    bottom_pred = np.zeros(len(chosen))
    bottom_meas = np.zeros(len(chosen))
    for i, bucket in enumerate(order):
        vals_pred = pivot_pred[bucket].to_numpy()
        vals_meas = pivot_meas[bucket].to_numpy()
        color = colors[i % len(colors)]
        ax.bar(
            x - width / 2,
            vals_pred,
            width,
            bottom=bottom_pred,
            color=color,
            label=bucket,
        )
        ax.bar(
            x + width / 2, vals_meas, width, bottom=bottom_meas, color=color, hatch="//"
        )
        bottom_pred += vals_pred
        bottom_meas += vals_meas

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(dpus_by_label[lbl])) for lbl in chosen])
    ax.set_xlabel("dpus  (left bar = predicted, right hatched bar = measured)")
    ax.set_ylabel("share of total cost (%)")
    ax.set_title(f"{fn_name}: predicted vs measured cost composition")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_failure_summary(failures_df: pd.DataFrame, out_path: pathlib.Path):
    """Bar chart of failure counts per (stage, signature), pooled across
    every function -- the first thing to look at in the crash report (see
    cinm_experiments.failures), before drilling into where in the config
    space a given signature clusters (plot_failure_scatter)."""
    if failures_df.empty:
        return
    counts = (
        failures_df.groupby(["stage", "signature"]).size().sort_values(ascending=False)
    )
    labels = [f"{stage}:{sig}" for stage, sig in counts.index]
    colors = [
        "crimson" if stage == "compile" else "steelblue" for stage, _ in counts.index
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4.5))
    ax.bar(labels, counts.values, color=colors)
    ax.set_ylabel("count")
    ax.set_title("Compile/run failures by signature")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_failure_scatter(
    failures_df: pd.DataFrame,
    fn_name: str,
    out_path: pathlib.Path,
    xdim: str = "dpus",
    ydim: str = "mramCol",
):
    """Failing configs' (xdim, ydim), colored by signature -- surfaces
    whether a signature clusters in a region of the space (e.g. only at very
    high dpu counts), which would signal that fn's config_filter needs
    tightening rather than that config being individually broken."""
    data = failures_df[failures_df["fn_name"] == fn_name]
    if data.empty or xdim not in data.columns or ydim not in data.columns:
        return
    sigs = sorted(data["signature"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, sig in enumerate(sigs):
        sub = data[data["signature"] == sig]
        ax.scatter(sub[xdim], sub[ydim], s=18, alpha=0.7, label=sig, color=cmap(i % 10))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(xdim)
    ax.set_ylabel(ydim)
    ax.set_title(f"{fn_name}: failing configs by signature")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Task collection (runs in the main process) ────────────────────────────────


def _fn_matches(fn_name: str, fn_filter: Optional[str]) -> bool:
    """Whether fn_name should be included given --fn-filter (None/empty means
    every function). fn_filter is a comma-separated set of exact fn_name
    values -- unlike --filter (a substring match on plot *names*), a
    substring match on fn_name itself would be ambiguous (e.g. "4MB" also
    matches "red_64MB")."""
    return not fn_filter or fn_name in fn_filter.split(",")


def collect_metric_tasks(
    in_dir: pathlib.Path,
    out_dir: pathlib.Path,
    name_filter: str,
    fn_filter: Optional[str] = None,
):
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
            if not _fn_matches(fn_name, fn_filter):
                continue
            for xcol in xcols:
                plot_name = f"{metric}_vs_{xcol}"
                if name_filter and name_filter not in plot_name:
                    continue
                out_path = out_dir / fn_name / f"{plot_name}.png"
                tasks.append(
                    (
                        f"{plot_name} ({fn_name})",
                        plot_metric_vs_x,
                        (group.copy(), metric, fn_name, xcol, out_path),
                    )
                )
    return tasks


def collect_calibration_tasks(
    oracle_dir: pathlib.Path,
    agg_dir: pathlib.Path,
    out_dir: pathlib.Path,
    pool_out_dir: pathlib.Path,
    name_filter: Optional[str],
    predicted_dir: Optional[pathlib.Path] = None,
    fn_filter: Optional[str] = None,
):
    plot_name = "cost_calibration"
    if name_filter and name_filter not in plot_name:
        return []

    predicted_all = None
    if predicted_dir is not None:
        predicted_csv = predicted_dir / "predicted_costs.csv"
        if predicted_csv.exists():
            predicted_all = pd.read_csv(predicted_csv)
        else:
            print(
                f"  {predicted_csv} not found, cost_calibration will use each "
                "pool's own (possibly stale) cost column",
                file=sys.stderr,
            )

    tasks = []
    pool_out_dir.mkdir(parents=True, exist_ok=True)
    for fn_name, pool_csv in find_function_pools(oracle_dir):
        if not _fn_matches(fn_name, fn_filter):
            continue
        pool = pd.read_csv(pool_csv)
        key_cols = _key_cols(pool)
        measured = compute_measured_cost(agg_dir, fn_name, key_cols)
        if measured.empty:
            print(
                f"  {fn_name}: no aggregated data, skipping calibration",
                file=sys.stderr,
            )
            continue
        n_launches = compute_n_launches(agg_dir, fn_name, key_cols)
        measured = measured.merge(n_launches, on=key_cols, how="left")
        fresh = (
            compute_fresh_predicted_cost(predicted_all, fn_name, key_cols)
            if predicted_all is not None
            else None
        )
        pool = augment_pool(pool, measured, key_cols, fresh)
        pool.to_csv(pool_out_dir / f"{fn_name}_pool.csv", index=False)

        out_path = out_dir / fn_name / f"{plot_name}.png"
        tasks.append(
            (
                f"{plot_name} ({fn_name})",
                plot_cost_calibration,
                (pool, fn_name, out_path),
            )
        )
    return tasks


def collect_launch_calibration_tasks(
    kernel_oracle_dir: pathlib.Path,
    agg_dir: pathlib.Path,
    out_dir: pathlib.Path,
    name_filter: Optional[str],
    fn_filter: Optional[str] = None,
):
    plot_name = "launch_calibration"
    if name_filter and name_filter not in plot_name:
        return []

    tasks = []
    for fn_name, pool_csv in find_function_pools(kernel_oracle_dir):
        if not _fn_matches(fn_name, fn_filter):
            continue
        pool = pd.read_csv(pool_csv)
        key_cols = _key_cols(pool)
        measured = compute_measured_launch_cost(agg_dir, fn_name, key_cols)
        if measured.empty:
            print(
                f"  {fn_name}: no aggregated data, skipping launch calibration",
                file=sys.stderr,
            )
            continue
        pool = pool.merge(measured, on=key_cols, how="left")

        out_path = out_dir / fn_name / f"{plot_name}.png"
        tasks.append(
            (
                f"{plot_name} ({fn_name})",
                plot_launch_calibration,
                (pool, fn_name, out_path),
            )
        )

    return tasks


def collect_breakdown_tasks(
    agg_dir: pathlib.Path,
    predicted_dir: pathlib.Path,
    out_dir: pathlib.Path,
    name_filter: Optional[str],
    fn_filter: Optional[str] = None,
):
    plot_name = "cost_breakdown"
    if name_filter and name_filter not in plot_name:
        return []

    predicted_csv = predicted_dir / "predicted_costs.csv"
    if not predicted_csv.exists():
        print(
            f"  {predicted_csv} not found, skipping cost_breakdown plots "
            "(run `doit agg_predictions:<prim>` first)",
            file=sys.stderr,
        )
        return []
    predicted_all = pd.read_csv(predicted_csv)

    tasks = []
    for fn_name in sorted(predicted_all["fn_name"].unique()):
        if not _fn_matches(fn_name, fn_filter):
            continue
        joined = build_breakdown_frame(agg_dir, predicted_all, fn_name)
        if joined.empty:
            print(
                f"  {fn_name}: no measured/predicted overlap, skipping breakdown plots",
                file=sys.stderr,
            )
            continue
        fn_out = out_dir / fn_name
        for suffix, func in [
            ("error_box", plot_breakdown_error_box),
            ("scatter", plot_breakdown_scatter),
            ("error_heatmap", plot_breakdown_error_heatmap),
            ("composition", plot_breakdown_composition),
        ]:
            tasks.append(
                (
                    f"{plot_name}_{suffix} ({fn_name})",
                    func,
                    (joined, fn_name, fn_out / f"{plot_name}_{suffix}.png"),
                )
            )
    return tasks


def collect_combined_tasks(
    oracle_dir: pathlib.Path,
    agg_dir: pathlib.Path,
    predicted_dir: pathlib.Path,
    out_dir: pathlib.Path,
    pool_out_dir: pathlib.Path,
    name_filter: Optional[str],
    fn_filter: Optional[str] = None,
):
    """cost_overview.png: plot_cost_calibration's panel stacked over
    plot_breakdown_scatter's panels in one 4x3 figure (see
    plot_calibration_and_breakdown) -- requires both --oracle (for the
    calibration pool) and --predicted-dir (for the breakdown), and only
    for fn_names present in both."""
    plot_name = "cost_overview"
    if name_filter and name_filter not in plot_name:
        return []

    predicted_csv = predicted_dir / "predicted_costs.csv"
    if not predicted_csv.exists():
        print(
            f"  {predicted_csv} not found, skipping {plot_name} plots",
            file=sys.stderr,
        )
        return []
    predicted_all = pd.read_csv(predicted_csv)

    tasks = []
    pool_out_dir.mkdir(parents=True, exist_ok=True)
    for fn_name, pool_csv in find_function_pools(oracle_dir):
        if not _fn_matches(fn_name, fn_filter):
            continue
        pool = pd.read_csv(pool_csv)
        key_cols = _key_cols(pool)
        measured = compute_measured_cost(agg_dir, fn_name, key_cols)
        if measured.empty:
            print(
                f"  {fn_name}: no aggregated data, skipping {plot_name}",
                file=sys.stderr,
            )
            continue
        n_launches = compute_n_launches(agg_dir, fn_name, key_cols)
        measured = measured.merge(n_launches, on=key_cols, how="left")
        fresh = compute_fresh_predicted_cost(predicted_all, fn_name, key_cols)
        pool = augment_pool(pool, measured, key_cols, fresh)

        joined = build_breakdown_frame(agg_dir, predicted_all, fn_name)
        if joined.empty:
            print(
                f"  {fn_name}: no measured/predicted overlap, skipping {plot_name}",
                file=sys.stderr,
            )
            continue

        out_path = out_dir / fn_name / f"{plot_name}.png"
        tasks.append(
            (
                f"{plot_name} ({fn_name})",
                plot_calibration_and_breakdown,
                (pool, joined, fn_name, out_path),
            )
        )
    return tasks


def collect_failure_tasks(
    failures_csv: pathlib.Path,
    out_dir: pathlib.Path,
    name_filter: Optional[str],
    fn_filter: Optional[str] = None,
):
    plot_name = "failures"
    if name_filter and name_filter not in plot_name:
        return []
    if not failures_csv.exists():
        print(
            f"  {failures_csv} not found, skipping failure plots "
            "(run `doit failures:<prim>` first)",
            file=sys.stderr,
        )
        return []
    df = pd.read_csv(failures_csv)
    if fn_filter:
        df = df[df["fn_name"].isin(fn_filter.split(","))]
    if df.empty:
        return []

    tasks = [
        (
            f"{plot_name}_by_signature",
            plot_failure_summary,
            (df, out_dir / f"{plot_name}_by_signature.png"),
        )
    ]
    for fn_name in sorted(df["fn_name"].unique()):
        tasks.append(
            (
                f"{plot_name}_scatter ({fn_name})",
                plot_failure_scatter,
                (df, fn_name, out_dir / fn_name / f"{plot_name}_scatter.png"),
            )
        )
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in-dir",
        default="aggregated",
        help="Directory containing aggregated CSVs (default: aggregated)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Root output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only generate plots whose name contains this substring, "
        "e.g. --filter dpus or --filter cost_calibration "
        "(default: all plots)",
    )
    parser.add_argument(
        "--fn-filter",
        default=None,
        help="Only generate plots for these fn_name(s) (comma-separated, "
        "exact match -- e.g. --fn-filter red_4MB or "
        "--fn-filter red_4MB,red_64MB). Unlike --filter, this is an exact "
        "match rather than substring, since fn_name substrings collide "
        "(e.g. '4MB' also matches 'red_64MB'). Default: every function "
        "found in the input data.",
    )
    parser.add_argument(
        "--oracle",
        default=None,
        help="Path to an oracle directory, containing one "
        "infer_{fn_name}/pool.csv subdir per problem "
        "(same layout dodo.py reads configs from). Enables the "
        "cost_calibration plot; omit to skip it.",
    )
    parser.add_argument(
        "--pool-out-dir",
        default="pool_measured",
        help="Where to write cost-model-augmented pool CSVs "
        "(default: pool_measured; only used with --oracle)",
    )
    parser.add_argument(
        "--kernel-oracle",
        default=None,
        help="Path to a second oracle directory predicting only the "
        "on-DPU kernel cost (no transfer/alloc/free), same "
        "infer_{fn_name}/pool.csv layout as --oracle (joined by "
        "param columns, not row order). Enables the launch_calibration plot "
        "(measured launch cost vs predicted kernel cost); "
        "omit to skip it.",
    )
    parser.add_argument(
        "--predicted-dir",
        default=None,
        help="Directory containing predicted_costs.csv (aggregate."
        "aggregate_predicted_costs' output, e.g. PATHS.predicted_dir(prim)). "
        "Enables the per-category cost_breakdown plots; omit to skip them.",
    )
    parser.add_argument(
        "--failures-csv",
        default=None,
        help="Path to a failures.csv (cinm_experiments.failures.collect_failures' "
        "output, e.g. PATHS.failures_csv(prim)). Enables the failures_by_signature "
        "and per-function failures_scatter plots; omit to skip them.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Parallel plotting processes (default: cpu count)",
    )
    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)

    tasks = collect_metric_tasks(in_dir, out_dir, args.filter, args.fn_filter)
    if args.oracle:
        tasks += collect_calibration_tasks(
            pathlib.Path(args.oracle),
            in_dir,
            out_dir,
            pathlib.Path(args.pool_out_dir),
            args.filter,
            pathlib.Path(args.predicted_dir) if args.predicted_dir else None,
            args.fn_filter,
        )
    if args.kernel_oracle:
        tasks += collect_launch_calibration_tasks(
            pathlib.Path(args.kernel_oracle),
            in_dir,
            out_dir,
            args.filter,
            args.fn_filter,
        )
    if args.predicted_dir:
        tasks += collect_breakdown_tasks(
            in_dir,
            pathlib.Path(args.predicted_dir),
            out_dir,
            args.filter,
            args.fn_filter,
        )
    if args.oracle and args.predicted_dir:
        tasks += collect_combined_tasks(
            pathlib.Path(args.oracle),
            in_dir,
            pathlib.Path(args.predicted_dir),
            out_dir,
            pathlib.Path(args.pool_out_dir),
            args.filter,
            args.fn_filter,
        )
    if args.failures_csv:
        tasks += collect_failure_tasks(
            pathlib.Path(args.failures_csv), out_dir, args.filter, args.fn_filter
        )

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

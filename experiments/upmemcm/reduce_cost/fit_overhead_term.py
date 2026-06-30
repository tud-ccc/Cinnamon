#!/usr/bin/env python3
"""
Fit an additive overhead term f(dpus; params) that, added to a kernel-only
cost model's predicted cost, best matches the measured launch (kernel
execution) cost.

For each problem (fn_name), and once pooled across all problems, computes
residual = measured_launch_cost - predicted_cost per config and fits several
candidate templates for residual ~ f(dpus) via weighted least squares.

Weighting: every config is first weighted by 1 / (mean measured_launch_cost
in its dpus group)^2, then divided by the group's sample count. This is
equivalent to computing a per-dpus-value MSE and then combining those
per-group MSEs with more weight on low-measured-time (typically high-dpus)
groups, rather than letting groups with more sampled configs dominate just
by virtue of their count.

Usage:
  python3 fit_overhead_term.py --kernel-oracle ../../data/prim_red_oracle_notransfer
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from plot_cost import compute_measured_launch_cost, find_function_pools

# ── Templates: name -> feature function (dpus array -> [n, k] feature matrix) ─

TEMPLATES = {
    "a (baseline)":               lambda d, c: np.zeros((len(d), 0)),
    "a + b*dpus":                 lambda d, c: np.column_stack([d]),
    "a + b*(dpus//64)":           lambda d, c: np.column_stack([d // 64]),
    "a + b*log2(dpus)":           lambda d, c: np.column_stack([np.log2(d)]),
    "a + b*dpus + c*(dpus//64)":  lambda d, c: np.column_stack([d, d // 64]),
    "a + b*dpus + c*log2(dpus)":  lambda d, c: np.column_stack([d, np.log2(d)]),

    "a + b*mramCols":             lambda d, c: np.column_stack([c]),
    "a + b*mramCols + c*dpus":    lambda d, c: np.column_stack([c, d]),
    "a + b*log2(mramCols)":       lambda d, c: np.column_stack([np.log2(c)]),
    "a + b*log2(mramCols) + c*log2(dpus)":       lambda d, c: np.column_stack([np.log2(c), np.log2(d)]),
}


def build_clean_pool(agg_dir: pathlib.Path, pool_csv: pathlib.Path, fn_name: str) -> pd.DataFrame:
    """measured_launch_cost/cost/dpus for fn_name, dropping unmeasured/timeout configs."""
    measured = compute_measured_launch_cost(agg_dir, fn_name)
    if measured.empty:
        return measured
    pool = pd.read_csv(pool_csv)
    pool = pool.merge(measured, left_index=True, right_on="config_id", how="left")
    pool = pool.drop(columns=["config_id"])
    data = pool.dropna(subset=["measured_launch_cost"])
    data = data[np.isfinite(data["cost"]) & np.isfinite(data["measured_launch_cost"])]
    data = data[(data["dpus"] >= 2) & (data["measured_launch_cost"] > 10e4)]
    return data


def group_weights(dpus: np.ndarray, measured_launch_cost_ms: np.ndarray) -> np.ndarray:
    """1 / (group mean measured_launch_cost)^2, spread evenly within each dpus group."""
    df = pd.DataFrame({"dpus": dpus, "measured_launch_cost_ms": measured_launch_cost_ms})
    group_mean = df.groupby("dpus")["measured_launch_cost_ms"].transform("mean")
    group_size = df.groupby("dpus")["measured_launch_cost_ms"].transform("size")
    return (1.0 / (group_mean ** 2 * group_size)).to_numpy()


def weighted_rmse(residual: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.average((residual - pred) ** 2, weights=weight)))


def fit_template(dpus: np.ndarray, mramCols: np.ndarray, residual: np.ndarray, weight: np.ndarray, feature_fn):
    """Weighted non-negative least squares: all coefficients, including the
    intercept, are constrained to be >= 0. Every feature column (dpus,
    mramCols, log2 of either, dpus//64) is itself non-negative, so this keeps
    f(dpus) non-negative and monotonically non-decreasing, avoiding the
    blow-up-to-negative-outlier failure mode an unconstrained negative slope
    can produce at the extremes of the range.
    """
    X = feature_fn(dpus, mramCols)
    X_full = np.column_stack([np.ones(len(residual)), X])
    sqrt_w = np.sqrt(weight)
    coef_full, _ = nnls(X_full * sqrt_w[:, None], residual * sqrt_w)
    pred = X_full @ coef_full
    return {"intercept": float(coef_full[0]), "coef": list(coef_full[1:]), "pred": pred}


def fit_all_templates(dpus: np.ndarray, mramCols: np.ndarray, residual: np.ndarray, weight: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, feature_fn in TEMPLATES.items():
        fit = fit_template(dpus, mramCols, residual, weight, feature_fn)
        rmse = weighted_rmse(residual, fit["pred"], weight)
        rows.append({
            "template": name,
            "weighted_rmse_ms": rmse,
            "intercept": fit["intercept"],
            "coef": fit["coef"],
        })
    return pd.DataFrame(rows).sort_values("weighted_rmse_ms").reset_index(drop=True)


def make_fit_plot(dpus: np.ndarray, mramCols: np.ndarray, residual: np.ndarray, weight: np.ndarray,
                   best_name: str, out_path: pathlib.Path, title: str):
    fit = fit_template(dpus, mramCols, residual, weight, TEMPLATES[best_name])
    order = np.argsort(dpus)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(dpus, residual, s=10, alpha=0.4, color="darkorange", label="residual (measured - predicted)")
    ax.plot(dpus[order], fit["pred"][order], color="crimson", linewidth=1.5,
            label=f"fit: {best_name}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of DPUs")
    ax.set_ylabel("residual (ms)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-oracle", required=True,
                         help="Path to a kernel-only oracle directory (infer_{fn_name}/pool.csv "
                              "layout, cost in ms, no transfer/alloc/free cost)")
    parser.add_argument("--in-dir", default="aggregated",
                         help="Directory with aggregated CSVs (default: aggregated)")
    parser.add_argument("--plots-dir", default="plots",
                         help="Where to write overhead_fit.png per problem (default: plots)")
    args = parser.parse_args()

    agg_dir = pathlib.Path(args.in_dir)
    plots_dir = pathlib.Path(args.plots_dir)

    all_dpus, all_mramCols, all_residual_ms, all_measured_cost_ms = [], [], [], []

    for fn_name, pool_csv in find_function_pools(pathlib.Path(args.kernel_oracle)):
        data = build_clean_pool(agg_dir, pool_csv, fn_name)
        if data.empty:
            print(f"  {fn_name}: no usable data, skipping", file=sys.stderr)
            continue

        dpus = data["dpus"].to_numpy(dtype=float)
        mramCols = data["mramCol"].to_numpy(dtype=float)
        measured_cost_ms = data["measured_launch_cost"].to_numpy() / 1e6
        residual_ms = measured_cost_ms - data["cost"].to_numpy()
        weight = group_weights(dpus, measured_cost_ms)

        results = fit_all_templates(dpus, mramCols, residual_ms, weight)
        print(f"\n=== {fn_name} (n={len(data)}) ===")
        print(results.to_string(index=False))

        best = results.iloc[0]["template"]
        make_fit_plot(dpus, mramCols, residual_ms, weight, best,
                      plots_dir / fn_name / "overhead_fit.png",
                      f"{fn_name}: residual + best fit ({best})")

        all_dpus.append(dpus)
        all_mramCols.append(mramCols)
        all_residual_ms.append(residual_ms)
        all_measured_cost_ms.append(measured_cost_ms)

    if not all_dpus:
        print("No data found for any problem.", file=sys.stderr)
        sys.exit(1)

    dpus_pooled = np.concatenate(all_dpus)
    mramCols_pooled = np.concatenate(all_mramCols)
    residual_pooled = np.concatenate(all_residual_ms)
    measured_cost_pooled_ms = np.concatenate(all_measured_cost_ms)
    weight_pooled = group_weights(dpus_pooled, measured_cost_pooled_ms)

    results_pooled = fit_all_templates(dpus_pooled, mramCols_pooled, residual_pooled, weight_pooled)
    print(f"\n=== ALL PROBLEMS POOLED (n={len(dpus_pooled)}) ===")
    print(results_pooled.to_string(index=False))

    best_pooled = results_pooled.iloc[0]["template"]
    make_fit_plot(dpus_pooled, mramCols_pooled, residual_pooled, weight_pooled, best_pooled,
                  plots_dir / "overhead_fit_pooled.png",
                  f"All problems pooled: residual + best fit ({best_pooled})")


if __name__ == "__main__":
    main()

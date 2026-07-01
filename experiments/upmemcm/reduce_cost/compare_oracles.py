#!/usr/bin/env python3
"""Compare two oracle cost predictions against measured values.

Generates a 2-panel scatter plot (same style as fit_overhead_term.py):
  left  — oracle A predicted vs measured cost
  right — oracle B predicted vs measured cost

Oracle pool.csv files are expected under <oracle-dir>/infer_<fn_name>/pool.csv
and must have a `cost` column in milliseconds (row index = config_id).

Usage:
  python3 compare_oracles.py \\
      --oracle-a ../../data/oracle_without_transfer \\
      --oracle-b ../../data/oracle_with_transfer \\
      --label-a "no transfer" --label-b "with transfer" \\
      --measured full \\
      --in-dir aggregated
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
from scipy.stats import spearmanr

from plot_cost import find_function_pools, compute_measured_cost


# ── Measured-cost loaders ──────────────────────────────────────────────────────

def _measured_launch(agg_dir: pathlib.Path, fn_name: str) -> pd.DataFrame:
    """[config_id, measured_ms] — mean per-launch cost (ns→ms)."""
    launch = pd.read_csv(agg_dir / "launch.csv")
    launch = launch[launch["fn_name"] == fn_name]
    per_iter = launch.groupby(["config_id", "iteration"])["elapsed_ns"].mean()
    df = per_iter.groupby("config_id").mean().rename("measured_ms").reset_index()
    df["measured_ms"] /= 1e6
    return df


def _measured_full(agg_dir: pathlib.Path, fn_name: str) -> pd.DataFrame:
    """[config_id, measured_ms] — full loop cost (ns→ms)."""
    mc = compute_measured_cost(agg_dir, fn_name)
    if mc.empty:
        return pd.DataFrame()
    mc = mc.rename(columns={"measured_cost": "measured_ms"})
    mc["measured_ms"] /= 1e6
    return mc


# ── Data alignment ─────────────────────────────────────────────────────────────

def _load_pool(oracle_dir: pathlib.Path, fn_name: str) -> pd.DataFrame:
    pool_csv = oracle_dir / f"infer_{fn_name}" / "pool.csv"
    if not pool_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(pool_csv)
    df["config_id"] = df.index
    return df


# ── Derived oracle B ───────────────────────────────────────────────────────────

def compute_cost_b(pool_a: pd.DataFrame, agg_dir: pathlib.Path, fn_name: str) -> pd.Series:
    """Add measured gather+scatter transfer cost to the oracle-A prediction.

    For each config, computes the per-iteration sum of gather and scatter
    elapsed_ns, averages over iterations, converts to ms, and adds it to
    pool_a["cost"] (the oracle-A kernel cost prediction).
    """
    rows = []
    for fname in ("gather.csv", "scatter.csv"):
        path = agg_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Transfer CSV not found: {path}")
        df = pd.read_csv(path)
        df = df[df["fn_name"] == fn_name][["config_id", "iteration", "elapsed_ns"]]
        rows.append(df)

    transfer = pd.concat(rows)
    # sum gather + scatter per (config_id, iteration), then average over iterations
    per_iter = transfer.groupby(["config_id", "iteration"])["elapsed_ns"].sum()
    mean_transfer_ns = per_iter.groupby("config_id").mean()
    mean_transfer_ms = mean_transfer_ns / 1e6

    transfer_ms = pool_a["config_id"].map(mean_transfer_ms)
    return pool_a["cost"] + transfer_ms


# ── Data alignment ─────────────────────────────────────────────────────────────

def build_comparison_data(agg_dir, oracle_a, oracle_b, fn_name, measured_mode):
    """Return DataFrame with [dpus, cost_a, cost_b, measured_ms], NaN/inf dropped.

    If oracle_b is None, cost_b is derived from oracle_a's pool via compute_cost_b().
    """
    pool_a = _load_pool(oracle_a, fn_name)
    if pool_a.empty:
        return pd.DataFrame()

    if oracle_b is not None:
        pool_b = _load_pool(oracle_b, fn_name)
        if pool_b.empty:
            return pd.DataFrame()
        cost_b_series = pool_b.set_index("config_id")["cost"].rename("cost_b")
        df = pool_a.rename(columns={"cost": "cost_a"})[["config_id", "dpus", "mramCol", "cost_a"]]
        df = df.merge(cost_b_series, on="config_id", how="inner")
    else:
        df = pool_a.copy()
        df["cost_b"] = compute_cost_b(pool_a, agg_dir, fn_name)
        df = df.rename(columns={"cost": "cost_a"})[["config_id", "dpus", "mramCol", "cost_a", "cost_b"]]

    if measured_mode == "launch":
        meas = _measured_launch(agg_dir, fn_name)
    else:
        meas = _measured_full(agg_dir, fn_name)
    if meas.empty:
        return pd.DataFrame()

    df = df.merge(meas, on="config_id", how="inner")
    df = df.dropna(subset=["cost_a", "cost_b", "measured_ms"])
    df = df[np.isfinite(df["cost_a"]) & np.isfinite(df["cost_b"]) & np.isfinite(df["measured_ms"])]
    return df.reset_index(drop=True)


# ── Plot ───────────────────────────────────────────────────────────────────────

def _spearman_topk(pred, measured, top_q):
    k = max(2, int(np.ceil(top_q * len(pred))))
    idx = np.argsort(pred)[:k]
    rho, _ = spearmanr(pred[idx], measured[idx])
    return float(rho) if np.isfinite(rho) else 0.0


def _false_positives(predicted_ms, measured_ms):
    """Configs ranked better than the true optimum — rank of argmin(measured)."""
    true_best = np.argmin(measured_ms)
    return int(np.sum(predicted_ms < predicted_ms[true_best]))


def make_comparison_plot(dpus, cost_a, cost_b, measured_ms,
                          label_a, label_b, top_quantile, out_path, title):
    rho_a  = _spearman_topk(cost_a, measured_ms, top_quantile)
    rho_b  = _spearman_topk(cost_b, measured_ms, top_quantile)
    w      = 1.0 / (measured_ms ** 2); w /= w.sum()
    rmse_a = float(np.sqrt(np.average((cost_a - measured_ms) ** 2, weights=w)))
    rmse_b = float(np.sqrt(np.average((cost_b - measured_ms) ** 2, weights=w)))
    k      = max(2, int(np.ceil(top_quantile * len(measured_ms))))

    k_min     = int(np.floor(np.log2(dpus.min())))
    k_max     = int(np.ceil(np.log2(dpus.max())))
    dpu_ticks = [2 ** i for i in range(k_min, k_max + 1)]
    dpu_norm  = LogNorm(vmin=dpus.min(), vmax=dpus.max())

    pad = 1.15
    all_x = np.concatenate([cost_a, cost_b])
    lo = min(all_x.min(), measured_ms.min()) / pad
    hi = max(all_x.max(), measured_ms.max()) * pad

    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    sc = None
    for ax, x, subtitle, rho, rmse in [
        (ax0, cost_a, label_a, rho_a, rmse_a),
        (ax1, cost_b, label_b, rho_b, rmse_b),
    ]:
        sc = ax.scatter(x, measured_ms, s=10, alpha=0.7, c=dpus, cmap="viridis", norm=dpu_norm)
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("predicted cost (ms)")
        ax.set_ylabel("measured cost (ms)")
        ax.set_title(subtitle)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        fp = _false_positives(x, measured_ms)
        ax.text(0.03, 0.97,
                f"Spearman ρ@top-{top_quantile:.0%} (n={k}): {rho:.3f}\n"
                f"false positives (rank of true best): {fp}\n"
                f"wRMSE (1/cost²): {rmse:.3f} ms",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    cbar = fig.colorbar(sc, cax=cax, label="Number of DPUs", ticks=dpu_ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle-a", required=True,
                   help="First oracle directory (infer_<fn>/pool.csv layout)")
    p.add_argument("--oracle-b", default=None,
                   help="Second oracle directory (same layout); "
                        "if omitted, cost_b is derived from oracle A via compute_cost_b()")
    p.add_argument("--label-a", default="oracle A", help="Legend label for oracle A")
    p.add_argument("--label-b", default="oracle B", help="Legend label for oracle B")
    p.add_argument("--in-dir",  default="aggregated",
                   help="Aggregated CSVs directory (default: aggregated)")
    p.add_argument("--plots-dir", default="plots",
                   help="Output directory for plots (default: plots)")
    p.add_argument("--measured", choices=["launch", "full"], default="full",
                   help="Which measured cost to compare against: "
                        "'launch' = mean per-launch kernel cost, "
                        "'full' = total loop cost minus alloc/free (default: full)")
    p.add_argument("--top-quantile", type=float, default=0.20,
                   help="Top fraction used for Spearman ρ metric (default: 0.20)")
    p.add_argument("--plot-name", default="oracle_comparison",
                   help="Base filename for output plots (without .png); per-function plots "
                        "go to <plots-dir>/<fn>/<name>.png, pooled to <plots-dir>/<name>_pooled.png "
                        "(default: oracle_comparison)")
    args = p.parse_args()

    oracle_a  = pathlib.Path(args.oracle_a)
    oracle_b  = pathlib.Path(args.oracle_b) if args.oracle_b else None
    agg_dir   = pathlib.Path(args.in_dir)
    plots_dir = pathlib.Path(args.plots_dir)
    top_q     = args.top_quantile

    fn_pool_pairs = list(find_function_pools(oracle_a))
    if not fn_pool_pairs:
        print("No function pools found in oracle A.", file=sys.stderr)
        sys.exit(1)

    all_dpus, all_cost_a, all_cost_b, all_measured = [], [], [], []

    for fn_name, _ in fn_pool_pairs:
        df = build_comparison_data(agg_dir, oracle_a, oracle_b, fn_name, args.measured)
        if df.empty:
            print(f"  {fn_name}: no usable data, skipping", file=sys.stderr)
            continue

        dpus     = df["dpus"].to_numpy(dtype=float)
        cost_a   = df["cost_a"].to_numpy()
        cost_b   = df["cost_b"].to_numpy()
        measured = df["measured_ms"].to_numpy()

        make_comparison_plot(
            dpus, cost_a, cost_b, measured,
            args.label_a, args.label_b, top_q,
            plots_dir / fn_name / f"{args.plot_name}.png",
            f"{fn_name}: {args.label_a} vs {args.label_b}",
        )
        print(f"  {fn_name}: n={len(df)}")

        all_dpus.append(dpus); all_cost_a.append(cost_a)
        all_cost_b.append(cost_b); all_measured.append(measured)

    if not all_dpus:
        print("No data for any function.", file=sys.stderr)
        sys.exit(1)

    make_comparison_plot(
        np.concatenate(all_dpus),
        np.concatenate(all_cost_a),
        np.concatenate(all_cost_b),
        np.concatenate(all_measured),
        args.label_a, args.label_b, top_q,
        plots_dir / f"{args.plot_name}_pooled.png",
        f"All problems pooled: {args.label_a} vs {args.label_b}",
    )
    print(f"  pooled: n={sum(len(d) for d in all_dpus)}")


if __name__ == "__main__":
    main()

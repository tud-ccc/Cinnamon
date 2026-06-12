#!/usr/bin/env python3
"""
Top-k% recall and best-cost-found curves for Bayesian optimisation runs.

Usage:
    python plot_recall.py oracle.csv bo1.csv [bo2.csv ...] [--pcts 2 5 10 15] [--out-dir DIR]

oracle.csv  — exhaustive search pool.csv (ground truth costs for all configs)
bo*.csv     — one or more BO pool.csv files covering the same config space
              (multiple CSVs = multiple seeds; shown as mean±σ band per threshold)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META_COLS = {"visited", "valid", "cost", "mu", "sigma", "acq", "eval_iter"}


def apply_scale(costs, scale):
    if scale == "linear":
        return costs
    if scale == "log2":
        return np.log2(costs)
    if scale == "ln":
        return np.log(costs)
    if scale == "sqrt":
        return np.sqrt(costs)
    if scale == "cbrt":
        return np.cbrt(costs)
    return np.log10(costs)  # "log10" and default


def scale_label(scale):
    return {"linear": "linear", "log2": "log₂", "log10": "log₁₀",
            "ln": "ln", "sqrt": "√", "cbrt": "∛"}.get(scale, scale)


def dim_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def assert_same_space(oracle_df, bo_df, dims, path):
    bo_dims = dim_cols(bo_df)
    assert bo_dims == dims, (
        f"{path}: dimension columns differ from oracle\n"
        f"  oracle: {dims}\n  BO:     {bo_dims}"
    )
    for d in dims:
        ov = set(oracle_df[d].unique())
        bv = set(bo_df[d].unique())
        assert ov == bv, (
            f"{path}: dimension '{d}' has different unique values\n"
            f"  oracle: {sorted(ov)}\n  BO:     {sorted(bv)}"
        )


def make_keys(df, dims):
    return [tuple(row) for row in df[dims].itertuples(index=False)]


def compute_curves(bo_df, dims, topk_keys, max_iter):
    """Return (iters, recall, best_cost) step-function arrays over [0, max_iter]."""
    obs = bo_df[bo_df["eval_iter"].notna() & bo_df["cost"].notna()].copy()
    obs["eval_iter"] = obs["eval_iter"].astype(int)
    obs = obs.sort_values("eval_iter")

    keys = make_keys(obs, dims)
    iter_vals = obs["eval_iter"].tolist()
    cost_vals  = obs["cost"].tolist()

    iters = np.arange(0, max_iter + 1)
    recall    = np.zeros(len(iters))
    best_cost = np.full(len(iters), np.nan)

    ev_idx = 0
    found_topk = 0
    running_best = np.inf

    for n in iters:
        while ev_idx < len(iter_vals) and iter_vals[ev_idx] <= n:
            key  = keys[ev_idx]
            cost = cost_vals[ev_idx]
            if key in topk_keys:
                found_topk += 1
            running_best = min(running_best, cost)
            ev_idx += 1
        recall[n]    = found_topk / len(topk_keys)
        best_cost[n] = running_best if np.isfinite(running_best) else np.nan

    return iters, recall, best_cost


def plot_curves(ax, iters, curves, names, ylabel, title):
    cmap = plt.cm.tab10
    label_individually = len(curves) <= 8
    for i, (c, name) in enumerate(zip(curves, names)):
        ax.plot(iters, c, color=cmap(i / max(len(curves), 1)),
                lw=0.9, alpha=0.5 if len(curves) > 1 else 1.0,
                label=name if label_individually else None)
    if len(curves) > 1:
        mean = np.nanmean(curves, axis=0)
        std  = np.nanstd(curves,  axis=0)
        ax.plot(iters, mean, color="black", lw=2, label="mean", zorder=5)
        ax.fill_between(iters, mean - std, mean + std,
                        color="black", alpha=0.15, label="±1σ")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right" if "recall" in ylabel.lower() else "upper right")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oracle", help="Exhaustive search pool.csv")
    ap.add_argument("bo_pools", nargs="+", metavar="bo.csv",
                    help="BO pool.csv files (one per seed/run)")
    ap.add_argument("--pcts", type=float, nargs="+", default=[2, 5, 10, 15],
                    help="Top-k%% thresholds to plot (default: 2 5 10 15)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: first BO pool directory)")
    ap.add_argument("--objective-scale", default="log10",
                    help="Cost transform used during surrogate training "
                         "(linear, log2, log10, ln, sqrt, cbrt)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.bo_pools[0]).parent
    scale = args.objective_scale
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load oracle ───────────────────────────────────────────────────────────
    oracle_df  = pd.read_csv(args.oracle)
    dims       = dim_cols(oracle_df)
    oracle_obs = oracle_df
    if "valid" in oracle_df.columns:
        oracle_obs = oracle_df[oracle_df["valid"] == 1]
    oracle_obs = oracle_obs[oracle_obs["cost"].notna()]
    oracle_best = oracle_obs["cost"].min()
    n_oracle = len(oracle_obs)
    print(f"Oracle: {n_oracle} observed configs, best cost = {oracle_best:.3g}")

    # ── Load BO pools, assert same space ─────────────────────────────────────
    max_iter = 0
    bo_data  = []
    for path in args.bo_pools:
        df = pd.read_csv(path)
        try:
            assert_same_space(oracle_df, df, dims, path)
        except AssertionError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        if "eval_iter" not in df.columns:
            print(f"WARNING: {path} has no eval_iter column — skipping", file=sys.stderr)
            continue
        max_iter = max(max_iter, int(df["eval_iter"].dropna().max()))
        bo_data.append((Path(path).stem, df))

    if not bo_data:
        print("No valid BO pool CSVs.", file=sys.stderr)
        sys.exit(1)

    print(f"BO runs: {len(bo_data)}, max iteration: {max_iter}")

    # ── Compute curves for each percentage threshold ──────────────────────────
    pct_results = []  # (pct, k, recalls[n_seeds, n_iters])
    all_best = None
    iters = None

    for pi, pct in enumerate(sorted(args.pcts)):
        k = max(1, int(np.ceil(pct / 100 * n_oracle)))
        topk_df   = oracle_obs.nsmallest(k, "cost")
        topk_keys = set(make_keys(topk_df, dims))
        print(f"  top {pct}%: k={k}, cost range "
              f"[{topk_df['cost'].min():.3g}, {topk_df['cost'].max():.3g}]")

        seed_recalls = []
        seed_bests   = []
        for name, df in bo_data:
            it, recall, best_cost = compute_curves(df, dims, topk_keys, max_iter)
            seed_recalls.append(recall)
            seed_bests.append(best_cost)
            if pi == 0:
                print(f"    {name}: final recall = {recall[-1]:.1%}, "
                      f"best found = {np.nanmin(best_cost):.3g}")

        pct_results.append((pct, k, np.array(seed_recalls)))
        if pi == 0:
            all_best = np.array(seed_bests)
            iters = it

    names = [name for name, _ in bo_data]

    # ── Plot 1: multi-threshold recall ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = plt.cm.tab10
    for i, (pct, k, seed_recalls) in enumerate(pct_results):
        color = cmap(i / max(len(pct_results), 1))
        label = f"top {pct}% (k={k})"
        if seed_recalls.shape[0] == 1:
            ax.plot(iters, seed_recalls[0], color=color, lw=1.5, label=label)
        else:
            mean = np.nanmean(seed_recalls, axis=0)
            std  = np.nanstd(seed_recalls,  axis=0)
            ax.plot(iters, mean, color=color, lw=1.5, label=label)
            ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Recall  (fraction of threshold found)")
    ax.set_title("Top-k% recall over BO iterations")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "recall_pcts.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # ── Plot 2: best cost found so far ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    scaled_best = apply_scale(all_best, scale)
    sl = scale_label(scale)
    ylabel = f"Best cost found so far  ({sl})" if scale != "linear" else "Best cost found so far"
    plot_curves(ax, iters, scaled_best, names,
                ylabel=ylabel,
                title="Best cost found over BO iterations")
    ax.axhline(apply_scale(oracle_best, scale), color="red", lw=1, ls="--",
               label=f"Oracle best ({oracle_best:.3g})")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    p = out_dir / "best_cost_found.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()

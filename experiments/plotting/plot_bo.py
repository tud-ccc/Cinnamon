#!/usr/bin/env python3
"""
Visualise Bayesian optimisation runs: heatmaps, surrogate calibration,
recall curves, and best-cost-found plots.

Usage — standalone pools (heatmaps and calibration only):
    python plot_bo.py pool.csv [pool2.csv ...] [options]

Usage — with oracle (adds recall / best-cost plots and fills README stats):
    python plot_bo.py --oracle oracle.csv bo1.csv [bo2.csv ...]
                      [--oracle oracle2.csv bo3.csv ...] [options]
"""
import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from tqdm import tqdm


META_COLS = {"visited", "valid", "cost", "eval_iter", "eval_time_ms", "mu", "sigma", "acq"}
_SENTINEL_ITER = 2**63
_SEED_TEMPLATE_PATH    = Path(__file__).parent / "README_seed_synopsis.md"
_PROBLEM_TEMPLATE_PATH = Path(__file__).parent / "README_problem_synopsis.md"


# ── Scale helpers ──────────────────────────────────────────────────────────────
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
    return np.log10(costs)


def scale_label(scale):
    return {"linear": "linear", "log2": "log₂", "log10": "log₁₀",
            "ln": "ln", "sqrt": "√", "cbrt": "∛"}.get(scale, scale)


# ── Pool CSV loading ───────────────────────────────────────────────────────────
def _load_pool_csv(path, inf_margin=0.10):
    """Read a pool CSV and replace +inf costs with max_finite * (1 + margin)."""
    df = pd.read_csv(path)
    if "cost" in df.columns:
        cost = df["cost"]
        finite_max = cost[np.isfinite(cost)].max()
        if pd.notna(finite_max):
            df["cost"] = cost.replace(np.inf, finite_max * (1 + inf_margin))
    return df


# ── Dimension utilities ────────────────────────────────────────────────────────
def _dim_cols(df):
    return [c for c in df.columns if c not in META_COLS]


# ── Search-space validation ────────────────────────────────────────────────────
def assert_same_space(oracle_df, bo_df, dims, path):
    bo_dims = _dim_cols(bo_df)
    assert bo_dims == dims, (
        f"{path}: dimension columns differ from oracle\n"
        f"  oracle: {dims}\n  BO:     {bo_dims}"
    )
    # Both CSVs now only contain valid rows, so the BO pool is a subset of the
    # oracle's valid space — use subset check instead of equality.
    oracle_valid = oracle_df[oracle_df["valid"] == 1] if "valid" in oracle_df.columns else oracle_df
    for d in dims:
        ov = set(oracle_valid[d].unique())
        bv = set(bo_df[d].unique())
        assert bv.issubset(ov), (
            f"{path}: dimension '{d}' has values not present in oracle\n"
            f"  extra in BO: {sorted(bv - ov)}\n  oracle: {sorted(ov)}"
        )


def make_keys(df, dims):
    return [tuple(row) for row in df[dims].itertuples(index=False)]


# ── Recall / best-cost computation ────────────────────────────────────────────
def compute_curves(bo_df, dims, topk_keys, max_iter):
    """Return (iters, recall, best_cost) step-function arrays over [0, max_iter]."""
    obs = bo_df[
        bo_df["eval_iter"].notna()
        & (bo_df["eval_iter"] < _SENTINEL_ITER)
        & bo_df["cost"].notna()
    ].copy()
    obs["eval_iter"] = obs["eval_iter"].astype(int)
    obs = obs.sort_values("eval_iter")

    keys = make_keys(obs, dims)
    iter_vals = obs["eval_iter"].tolist()
    cost_vals = obs["cost"].tolist()

    iters = np.arange(0, max_iter + 1)
    recall = np.zeros(len(iters))
    best_cost = np.full(len(iters), np.nan)

    ev_idx, found_topk, running_best = 0, 0, np.inf
    for n in iters:
        while ev_idx < len(iter_vals) and iter_vals[ev_idx] <= n:
            if keys[ev_idx] in topk_keys:
                found_topk += 1
            running_best = min(running_best, cost_vals[ev_idx])
            ev_idx += 1
        recall[n] = found_topk / len(topk_keys)
        best_cost[n] = running_best if np.isfinite(running_best) else np.nan

    return iters, recall, best_cost


def _plot_curves_on_ax(ax, iters, curves, names, ylabel, title):
    cmap = plt.cm.tab10
    label_individually = len(curves) <= 8
    for i, (c, name) in enumerate(zip(curves, names)):
        ax.plot(iters, c, color=cmap(i / max(len(curves), 1)),
                lw=0.9, alpha=0.5 if len(curves) > 1 else 1.0,
                label=name if label_individually else None)
    if len(curves) > 1:
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)
        ax.plot(iters, mean, color="black", lw=2, label="mean", zorder=5)
        ax.fill_between(iters, np.maximum(mean - std, 1), mean + std, color="black", alpha=0.08, label="±1σ")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right" if "recall" in ylabel.lower() else "upper right")


# ── Cell status ────────────────────────────────────────────────────────────────
def cell_status(valid, attempted, has_value):
    """Return (invalid, failed) boolean masks.

    Works with both numpy arrays (build_rgba) and pandas Series (generate_readme).
    - invalid:  constraint-violated (valid == 0)
    - failed:   eval_iter was set (actually attempted), valid, but produced no cost
    """
    invalid = valid == 0
    failed = ~has_value & ~invalid & (attempted > 0)
    return invalid, failed


# ── Heatmap plotting ───────────────────────────────────────────────────────────
def build_rgba(subset, x, y, x_vals, y_vals, val_col, norm, cmap, *, white_unvisited=True):
    """Return an RGBA image for one facet slice. x = pivot index (rows), y = pivot columns."""
    piv_val      = subset.pivot_table(index=x, columns=y, values=val_col,     aggfunc="mean")
    piv_attempted = subset.pivot_table(index=x, columns=y, values="attempted", aggfunc="max")
    piv_valid    = subset.pivot_table(index=x, columns=y, values="valid",     aggfunc="max")

    vals      = piv_val.reindex(index=x_vals, columns=y_vals).to_numpy(dtype=float)
    attempted = np.nan_to_num(piv_attempted.reindex(index=x_vals, columns=y_vals).to_numpy(dtype=float))
    valid     = np.nan_to_num(piv_valid.reindex(index=x_vals, columns=y_vals).to_numpy(dtype=float))

    no_data = np.isnan(vals)
    invalid, failed = cell_status(valid, attempted, ~no_data)

    fill = np.nanmedian(vals) if not np.all(no_data) else 0.0
    safe = np.where(no_data, fill, vals)

    img = cmap(norm(safe))
    img[no_data] = [1.0, 1.0, 1.0, 1.0]    # white — unsampled
    img[invalid] = [0.80, 0.80, 0.80, 1.0]  # light gray — constraint violated
    img[failed]  = [0.55, 0.55, 0.55, 1.0]  # mid gray — attempted but failed

    return img


def make_facet_plot(df, x, y, f, out_dir, metric, title, label, norm, cmap, *, white_unvisited=True):
    x_vals     = sorted(df[x].unique())
    y_vals     = sorted(df[y].unique())
    facet_vals = sorted(df[f].unique())

    # imshow: rows = x_vals (vertical), cols = y_vals (horizontal)
    extent = [0.5, len(y_vals) + 0.5, 0.5, len(x_vals) + 0.5]
    ncols = 2
    nrows = int(np.ceil(len(facet_vals) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    flat = axes.flatten()
    for ax in flat[len(facet_vals):]:
        ax.set_visible(False)

    for idx, (ax, T) in enumerate(zip(flat, facet_vals)):
        subset = df[df[f] == T]
        img = build_rgba(subset, x, y, x_vals, y_vals, metric, norm, cmap,
                         white_unvisited=white_unvisited)
        ax.imshow(img, origin="lower", aspect="auto", extent=extent)

        in_first_col = (idx % ncols == 0)
        in_bottom    = (idx >= (nrows - 1) * ncols)

        ax.set_xticks(range(1, len(y_vals) + 1))
        ax.set_yticks(range(1, len(x_vals) + 1))
        ax.set_xticklabels(y_vals if in_bottom else [], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(x_vals if in_first_col else [], fontsize=7)
        ax.set_xlim(0, len(y_vals) + 0.5)
        ax.set_ylim(0, len(x_vals) + 0.5)
        if in_first_col:
            ax.set_ylabel(x)
        if in_bottom:
            ax.set_xlabel(y)
        ax.set_title(f"{f} = {T}", fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=label)

    out_path = os.path.join(out_dir, f"pool_{metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_facet_plot_tasks(csv_path, scale, x, y, f):
    df = _load_pool_csv(csv_path)
    xyf = [x, y, f]
    agg = df.groupby(xyf, as_index=False).agg(
        valid=("valid", "max"),
        attempted=("eval_iter", lambda s: s.notna().any()),
        cost=("cost", "min"),
        mu=("mu", "min"),
        sigma=("sigma", "mean"),
        acq=("acq", "min"),
    )
    # Bring cost into the same scale space as mu/sigma/acq so all four heatmaps
    # use consistent units.
    agg["cost"] = apply_scale(agg["cost"], scale)

    base = dict(df=agg, x=x, y=y, f=f)
    sl = scale_label(scale)

    cost_vals = agg["cost"].dropna()
    tasks = [base | dict(
        metric="cost",
        title=f"Min observed cost  ({sl})  [white = unsampled, gray = failed]",
        label=f"Cost ({sl}, lower is better)",
        norm=mcolors.Normalize(vmin=cost_vals.min(), vmax=cost_vals.max()),
        cmap=plt.cm.viridis_r,
    )]
    for metric, title, label, cmap in [
        ("mu",    f"Min surrogate μ  ({sl} scale)",
         f"μ — predicted {sl}(cost)  (lower is better)", plt.cm.viridis_r),
        ("sigma", f"Mean surrogate σ  ({sl} scale)",
         f"σ — uncertainty in {sl}(cost)  (lower = more certain)", plt.cm.plasma),
        ("acq",   f"Best acquisition score  ({sl} scale)",
         f"UCB acquisition  μ − κσ  ({sl} scale)", plt.cm.plasma_r),
    ]:
        vals = agg[metric].dropna()
        tasks.append(base | dict(
            metric=metric, title=title, label=label, cmap=cmap,
            norm=mcolors.Normalize(vmin=vals.min(), vmax=vals.max()),
            white_unvisited=False,
        ))
    return tasks


# ── Validation / training RMSE + MAPE ─────────────────────────────────────────
def _plot_validation_metric(val, err_col, agg_fn, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    overall = val.groupby("x", sort=True)[err_col].agg(agg_fn)
    ax.plot(overall.index, overall.values, color="black", lw=2, label="all", zorder=5)
    x_min, x_max = int(val["x"].min()), int(val["x"].max())
    locator = plt.MaxNLocator(integer=True)
    auto_ticks = [int(t) for t in locator.tick_values(x_min, x_max) if t >= x_min]
    ax.set_xticks(sorted(set([x_min] + auto_ticks)))
    ax.set_xlim(left=x_min - 5)
    if "tasklets" in val.columns:
        tasklet_vals = sorted(val["tasklets"].unique())
        if len(tasklet_vals) > 1:
            cmap = plt.cm.tab10
            for i, T in enumerate(tasklet_vals):
                by_x = val[val["tasklets"] == T].groupby("x", sort=True)[err_col].agg(agg_fn)
                ax.plot(by_x.index, by_x.values,
                        color=cmap(i / len(tasklet_vals)), lw=1, alpha=0.7, label=f"T={T}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_validation_rmse(val_csv_path, out_dir, dataset, scale):
    val = pd.read_csv(val_csv_path)
    val["scaled_true"] = apply_scale(val["cost"], scale)
    val["sq_err"] = (val["mu"] - val["scaled_true"]) ** 2
    val["x"] = val["iter"]
    return _plot_validation_metric(
        val,
        err_col="sq_err",
        agg_fn=lambda s: np.sqrt(s.mean()),
        xlabel="Evaluations",
        ylabel=f"RMSE  ({scale_label(scale)} cost units)",
        title=f"Surrogate {dataset} RMSE\n",
        out_path=os.path.join(out_dir, f"pool_{dataset}_rmse.png"),
    )


def plot_validation_mape(val_csv_path, out_dir, dataset, scale):
    val = pd.read_csv(val_csv_path)
    scaled_cost = apply_scale(val["cost"], scale)
    val["pct_err"] = 100 * np.abs(val["mu"] - scaled_cost) / np.abs(scaled_cost)
    val["x"] = val["iter"]
    return _plot_validation_metric(
        val,
        err_col="pct_err",
        agg_fn="mean",
        xlabel="Evaluations",
        ylabel="MAPE (%)",
        title=(f"Surrogate {dataset} MAPE (cost in {scale_label(scale)} space)\n"
               "mean |predicted − true| / true × 100"),
        out_path=os.path.join(out_dir, f"pool_{dataset}_mape.png"),
    )


def _plot_eval_time_vs_cost(seed_csv_paths, out_dir, scale, names=None):
    """Scatter: evaluation wall-clock time (s) vs cost, coloured by dpus value."""
    frames = []
    for path in seed_csv_paths:
        df = _load_pool_csv(path)
        if "eval_time_ms" not in df.columns or "cost" not in df.columns:
            continue
        obs = df[df["cost"].notna() & df["eval_time_ms"].notna()].copy()
        if not obs.empty:
            frames.append(obs)

    if not frames:
        return None

    all_data = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    if "dpus" in all_data.columns:
        dpu_vals = sorted(all_data["dpus"].dropna().unique())
        cmap = plt.cm.tab20
        for i, d in enumerate(dpu_vals):
            subset = all_data[all_data["dpus"] == d]
            cost_scaled = apply_scale(subset["cost"] , scale)
            time_s = subset["eval_time_ms"] / 1000.0
            ax.scatter(cost_scaled, time_s, color=cmap(i % 20), s=14,
                       alpha=0.55, linewidths=0, label=f"dpus={int(d)}")
        ax.legend(title="dpus", fontsize=8, loc="upper right",
                  title_fontsize=8, ncol=max(1, len(dpu_vals) // 8))
    else:
        cost_scaled = apply_scale(all_data["cost"], scale)
        time_s = all_data["eval_time_ms"] / 1000.0
        ax.scatter(cost_scaled, time_s, s=14, alpha=0.55, linewidths=0)

    sl = scale_label(scale)
    ax.set_xlabel(f"Cost  ({sl})")
    ax.set_ylabel("Evaluation time (s)")
    ax.set_title(f"Evaluation time vs cost — {len(seed_csv_paths)} seed(s)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "agg_eval_time_vs_cost.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_eval_time_explainability(seed_csv_paths, out_dir):
    """Box plots of eval_time_ms per discrete parameter value + Spearman correlation bar chart.

    Combines all seeds into one dataset. Produces two output files:
    - agg_eval_time_corr.png   — horizontal bar chart of |ρ| per dimension
    - agg_eval_time_boxes.png  — one box plot per dimension, time vs param value
    """
    from scipy.stats import spearmanr

    frames = []
    for path in seed_csv_paths:
        df = _load_pool_csv(path)
        if "eval_time_ms" in df.columns:
            frames.append(df[df["eval_time_ms"].notna() & df["cost"].notna()])
    if not frames:
        return []

    data = pd.concat(frames, ignore_index=True)
    dims = _dim_cols(data)
    time_s = data["eval_time_ms"] / 1000.0
    out_paths = []
    out_dir = Path(out_dir)

    # ── Spearman correlation bar chart ──────────────────────────────────────────
    corrs = {}
    for d in dims:
        col = data[d].dropna()
        valid = col.index.intersection(time_s.dropna().index)
        if len(valid) < 5:
            continue
        rho, _ = spearmanr(col.loc[valid], time_s.loc[valid])
        corrs[d] = rho

    if corrs:
        sorted_dims = sorted(corrs, key=lambda d: abs(corrs[d]), reverse=True)
        rho_vals = [corrs[d] for d in sorted_dims]

        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(sorted_dims))))
        colors = ["#d62728" if r > 0 else "#1f77b4" for r in rho_vals]
        ax.barh(sorted_dims, rho_vals, color=colors, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Spearman ρ  (positive = larger value → slower)")
        ax.set_title(f"Eval-time correlation with parameters — {len(frames)} seed(s)")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        p = out_dir / "agg_eval_time_corr.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(str(p))

    # ── Box plots per dimension ─────────────────────────────────────────────────
    if dims:
        ncols = min(3, len(dims))
        nrows = int(np.ceil(len(dims) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        flat = axes.flatten()
        for ax in flat[len(dims):]:
            ax.set_visible(False)

        for ax, d in zip(flat, dims):
            groups = data.groupby(d)["eval_time_ms"].apply(
                lambda s: (s / 1000.0).dropna().values
            )
            labels = [str(k) for k in groups.index]
            ax.boxplot(groups.values, labels=labels, showfliers=False,
                       medianprops=dict(color="#d62728", lw=1.5))
            rho_str = f"  ρ={corrs[d]:.2f}" if d in corrs else ""
            ax.set_title(f"{d}{rho_str}", fontsize=9)
            ax.set_xlabel(d, fontsize=8)
            ax.set_ylabel("Eval time (s)", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"Eval time distribution by parameter — {len(frames)} seed(s)",
                     fontsize=10)
        plt.tight_layout()
        p = out_dir / "agg_eval_time_boxes.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(str(p))

    return out_paths


# ── Aggregate learning curves (multi-seed, problem-level) ─────────────────────
def _plot_aggregate_learning_curves(seed_csv_paths, out_dir, scale):
    """RMSE and MAPE aggregated across seeds: mean + IQR band, no per-tasklet split."""
    out_dir = Path(out_dir)
    sl = scale_label(scale)
    out_paths = []

    for dataset in ("validation", "training"):
        rmse_curves, mape_curves = [], []
        for seed_csv in seed_csv_paths:
            metric_csv = Path(seed_csv).parent / f"{dataset}.csv"
            if not metric_csv.exists():
                continue
            val = pd.read_csv(metric_csv)
            if not {"iter", "mu", "cost"}.issubset(val.columns):
                continue
            scaled_true = apply_scale(val["cost"], scale)
            val["sq_err"]  = (val["mu"] - scaled_true) ** 2
            val["pct_err"] = 100 * np.abs(val["mu"] - scaled_true) / np.abs(scaled_true)
            rmse_curves.append(val.groupby("iter")["sq_err"].agg(lambda s: np.sqrt(s.mean())))
            mape_curves.append(val.groupby("iter")["pct_err"].mean())

        if not rmse_curves:
            continue

        specs = [
            (rmse_curves, f"RMSE  ({sl} cost units)",
             f"Surrogate {dataset} RMSE — {len(rmse_curves)} seeds",
             f"agg_{dataset}_rmse.png"),
            (mape_curves, "MAPE (%)",
             f"Surrogate {dataset} MAPE — {len(mape_curves)} seeds",
             f"agg_{dataset}_mape.png"),
        ]
        for curves, ylabel, title, filename in specs:
            all_iters = sorted(set().union(*[set(c.index) for c in curves]))
            mat = np.full((len(curves), len(all_iters)), np.nan)
            iter_idx = {it: j for j, it in enumerate(all_iters)}
            for i, c in enumerate(curves):
                for it, v in c.items():
                    mat[i, iter_idx[it]] = v

            iters = np.array(all_iters)
            mean = np.nanmean(mat, axis=0)
            q25  = np.nanpercentile(mat, 25, axis=0)
            q75  = np.nanpercentile(mat, 75, axis=0)

            fig, ax = plt.subplots(figsize=(8, 4))
            cmap_t = plt.cm.tab10
            for i, row in enumerate(mat):
                ok = ~np.isnan(row)
                if ok.any():
                    ax.plot(iters[ok], row[ok],
                            color=cmap_t(i / max(len(mat), 1)), lw=0.8, alpha=0.4)
            ok = ~np.isnan(mean)
            ax.plot(iters[ok], mean[ok], color="black", lw=2, label="mean", zorder=5)
            ax.fill_between(iters[ok], q25[ok], q75[ok],
                            color="black", alpha=0.15, label="IQR (25–75%)")
            x_min = int(iters[0])
            locator = plt.MaxNLocator(integer=True)
            auto_ticks = [int(t) for t in locator.tick_values(x_min, int(iters[-1])) if t >= x_min]
            ax.set_xticks(sorted(set([x_min] + auto_ticks)))
            ax.set_xlim(left=x_min - 5)
            ax.set_xlabel("Evaluations")
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.set_title(title)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = out_dir / filename
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(str(p))

    return out_paths


# ── Aggregate timing plot ──────────────────────────────────────────────────────
def _plot_aggregate_timings(seed_csv_paths, out_dir):
    """Wall-clock time per evaluation aggregated across seeds: mean + IQR band."""
    out_dir = Path(out_dir)
    curves = []
    for seed_csv in seed_csv_paths:
        timing_csv = Path(seed_csv).parent / "timings.csv"
        if not timing_csv.exists():
            continue
        df = pd.read_csv(timing_csv)
        if not {"iter", "elapsed_ms"}.issubset(df.columns):
            continue
        curves.append(df.set_index("iter")["elapsed_ms"] / 1000.0)  # → seconds

    if not curves:
        return []

    all_iters = sorted(set().union(*[set(c.index) for c in curves]))
    mat = np.full((len(curves), len(all_iters)), np.nan)
    iter_idx = {it: j for j, it in enumerate(all_iters)}
    for i, c in enumerate(curves):
        for it, v in c.items():
            if it in iter_idx:
                mat[i, iter_idx[it]] = v

    iters = np.array(all_iters)
    mean = np.nanmean(mat, axis=0)
    q25  = np.nanpercentile(mat, 25, axis=0)
    q75  = np.nanpercentile(mat, 75, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap_t = plt.cm.tab10
    for i, row in enumerate(mat):
        ok = ~np.isnan(row)
        if ok.any():
            ax.plot(iters[ok], row[ok],
                    color=cmap_t(i / max(len(mat), 1)), lw=0.8, alpha=0.4)
    ok = ~np.isnan(mean)
    ax.plot(iters[ok], mean[ok], color="black", lw=2, label="mean", zorder=5)
    ax.fill_between(iters[ok], q25[ok], q75[ok],
                    color="black", alpha=0.15, label="IQR (25–75%)")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Cumulated elapsed time (s)")
    ax.set_title(f"Wall-clock time throughout evaluations — {len(curves)} seeds")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "agg_timings.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [str(p)]


# ── Sigma calibration plots (multi-seed aware) ────────────────────────────────
def _compute_min_dist(df, dims):
    """Min L1 distance in discrete grid-step indices to the nearest visited config."""
    from scipy.spatial import KDTree
    index_maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in dims}
    coords = np.column_stack([df[c].map(index_maps[c]).values for c in dims]).astype(float)
    visited_mask = df["cost"].notna().values
    visited_coords = coords[visited_mask]
    if len(visited_coords) == 0:
        return np.full(len(df), np.nan)
    tree = KDTree(visited_coords)
    dists, _ = tree.query(coords, k=1, p=1, workers=-1)
    return dists


def _plot_sigma_calibration(pool_csvs, out_dir, scale, names=None):
    """Compute and save pool_sigma_vs_dist.png and pool_sigma_scatter.png.

    pool_csvs: list of paths to pool.csv files (one per seed).
    Single-seed: shows IQR band and n= annotations. Multi-seed: shows per-seed
    thin lines plus a cross-seed mean±1σ band.
    """
    dfs = [_load_pool_csv(p) for p in pool_csvs]
    if names is None:
        names = [Path(p).parent.name for p in pool_csvs]

    dims = _dim_cols(dfs[0])
    sl = scale_label(scale)
    out_paths = []
    out_dir = str(out_dir)

    # Compute per-seed stats
    seed_data = []
    for df in dfs:
        if "sigma" not in df.columns:
            continue
        df = df.copy()
        df["min_dist"] = _compute_min_dist(df, dims)
        sub = df[df["sigma"].notna() & (df["valid"] == 1)]
        if sub.empty:
            continue
        dist_vals = sorted(sub["min_dist"].dropna().unique())
        xs, medians, q25s, q75s, counts = [], [], [], [], []
        for d in dist_vals:
            g = sub[sub["min_dist"] == d]["sigma"].values
            if len(g) == 0:
                continue
            xs.append(d)
            medians.append(np.median(g))
            q25s.append(np.percentile(g, 25))
            q75s.append(np.percentile(g, 75))
            counts.append(len(g))
        if not xs:
            continue
        seed_data.append({
            "xs": np.array(xs), "medians": np.array(medians),
            "q25s": np.array(q25s), "q75s": np.array(q75s),
            "counts": counts, "sub": sub,
        })

    if not seed_data:
        return []

    n = len(seed_data)
    cmap_t = plt.cm.tab10

    # ── sigma_vs_dist ──
    fig, ax = plt.subplots(figsize=(10, 4))
    if n == 1:
        d = seed_data[0]
        ax.plot(d["xs"], d["medians"], color="steelblue", lw=2, marker="o", ms=4, label="median σ")
        ax.fill_between(d["xs"], d["q25s"], d["q75s"], alpha=0.25, color="steelblue", label="IQR")
        for dist, m, cnt in zip(d["xs"], d["medians"], d["counts"]):
            ax.annotate(f"n={cnt}", (dist, m), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6, color="gray")
        ax.set_xticks(d["xs"].astype(int))
    else:
        for i, d in enumerate(seed_data):
            ax.plot(d["xs"], d["medians"], color=cmap_t(i / n), lw=1, alpha=0.6)
        all_xs = sorted({x for d in seed_data for x in d["xs"].tolist()})
        mat = np.full((n, len(all_xs)), np.nan)
        for i, d in enumerate(seed_data):
            for j, xv in enumerate(all_xs):
                idx = np.where(d["xs"] == xv)[0]
                if len(idx):
                    mat[i, j] = d["medians"][idx[0]]
        mean_m = np.nanmean(mat, axis=0)
        std_m  = np.nanstd(mat,  axis=0)
        xs_arr = np.array(all_xs)
        valid  = ~np.isnan(mean_m)
        ax.plot(xs_arr[valid], mean_m[valid], color="black", lw=2, label="mean", zorder=5)
        ax.fill_between(xs_arr[valid], (mean_m - std_m)[valid], (mean_m + std_m)[valid],
                        color="black", alpha=0.15, label="±1σ")
        ax.set_xticks(xs_arr[valid].astype(int))

    ax.set_xlabel("Min grid-step distance to nearest observation")
    ax.set_ylabel(f"σ (surrogate uncertainty, {sl} units)")
    ax.set_title(f"Surrogate σ vs. distance from observations  ({n} seed{'s' if n != 1 else ''})\n"
                 "Well-calibrated: σ increases monotonically with distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "pool_sigma_vs_dist.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p)

    # ── sigma_scatter ──
    if all("mu" in d["sub"].columns for d in seed_data):
        alpha = max(0.15, 0.4 / n)
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = None
        for d in seed_data:
            sub = d["sub"]
            sc = ax.scatter(sub["min_dist"], sub["sigma"], c=sub["mu"], cmap="viridis_r",
                            alpha=alpha, s=8, linewidths=0)
        if sc is not None:
            fig.colorbar(sc, ax=ax, label=f"μ  (predicted {sl} cost — lower is better)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Min grid-step distance to nearest observation")
        ax.set_ylabel(f"σ (surrogate uncertainty, {sl} units)")
        ax.set_title(f"Calibration scatter: σ vs. distance  ({n} seed{'s' if n != 1 else ''})\n"
                     "Bottom-right = overconfident far from data  ·  Top-left = underfit near data")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, "pool_sigma_scatter.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p)

    return out_paths


# ── Random-search baseline ────────────────────────────────────────────────────
def _expected_random_best(oracle_costs, max_iter):
    """Expected best cost after k uniform random draws without replacement.

    Uses the order-statistics formula:
        E[min_k] = c[0] + Σ_j diff_j · C(N-j, k) / C(N, k)
    with the recursion  r_j(k) = r_j(k-1) · (N-j-k+1) / (N-k+1).
    """
    c = np.sort(oracle_costs.astype(float))
    N = len(c)
    if N == 0 or max_iter == 0:
        return np.full(max_iter + 1, np.nan)

    diffs = np.diff(c)           # shape (N-1,)
    j_arr = np.arange(1, N)      # j = 1 … N-1

    r = np.ones(N - 1)           # r_j(k=0) = 1 for all j
    expected = np.full(max_iter + 1, np.nan)

    for k in range(1, max_iter + 1):
        denom = float(N - k + 1)
        if denom <= 0:            # k ≥ N: sampled everything
            expected[k:] = c[0]
            break
        numer = (N - j_arr - k + 1).astype(float)
        mask = numer > 0
        r[mask] *= numer[mask] / denom
        r[~mask] = 0.0
        expected[k] = c[0] + float(np.dot(diffs, r))

    return expected


# ── Oracle curves (recall + best-cost) ────────────────────────────────────────
def _plot_oracle_curves(oracle_csv, bo_csvs, out_dir, pcts, scale):
    """Compute and save recall_pcts.png and best_cost_found.png."""
    oracle_df = pd.read_csv(oracle_csv)
    dims = _dim_cols(oracle_df)
    oracle_obs = oracle_df[oracle_df["valid"] == 1] if "valid" in oracle_df.columns else oracle_df
    oracle_obs = oracle_obs[oracle_obs["cost"].notna()]
    oracle_best = oracle_obs["cost"].min()
    n_oracle = len(oracle_obs)

    max_iter = 0
    bo_data = []
    for path in bo_csvs:
        df = _load_pool_csv(path)
        valid_iters = df["eval_iter"].dropna() if "eval_iter" in df.columns else pd.Series([], dtype=float)
        valid_iters = valid_iters[valid_iters < _SENTINEL_ITER]
        if not valid_iters.empty:
            max_iter = max(max_iter, int(valid_iters.max()))
        bo_data.append((Path(path).stem, df))

    names = [n for n, _ in bo_data]
    pct_results = []
    all_best = None
    iters = None

    for pi, pct in enumerate(sorted(pcts)):
        k = max(1, int(np.ceil(pct / 100 * n_oracle)))
        topk_keys = set(make_keys(oracle_obs.nsmallest(k, "cost"), dims))
        seed_recalls, seed_bests = [], []
        for _, df in bo_data:
            it, recall, best = compute_curves(df, dims, topk_keys, max_iter)
            seed_recalls.append(recall)
            seed_bests.append(best)
        pct_results.append((pct, k, np.array(seed_recalls)))
        if pi == 0:
            all_best = np.array(seed_bests)
            iters = it

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    # recall
    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = plt.cm.tab10
    for i, (pct, k, seed_recalls) in enumerate(pct_results):
        color = cmap(i / max(len(pct_results), 1))
        label = f"top {pct}% (k={k})"
        if seed_recalls.shape[0] == 1:
            ax.plot(iters, seed_recalls[0], color=color, lw=1.5, label=label)
        else:
            mean = np.nanmean(seed_recalls, axis=0)
            std  = np.nanstd(seed_recalls, axis=0)
            ax.plot(iters, mean, color=color, lw=1.5, label=label)
            ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Recall  (fraction of threshold found)")
    ax.set_title(f"Top-k% recall over evaluations  ({len(bo_data)} seeds)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "recall_pcts.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(str(p))

    # best cost found — normalised so oracle best = 1
    random_baseline = _expected_random_best(oracle_obs["cost"].values, max_iter)
    fig, ax = plt.subplots(figsize=(9, 4))
    _plot_curves_on_ax(ax, iters, all_best / oracle_best, names,
                       ylabel="Best cost found  (relative to oracle, log scale)",
                       title=f"Best cost found over evaluations  ({len(bo_data)} seeds)")
    ax.set_yscale("log")
    ax.plot(iters, random_baseline / oracle_best, color="gray", lw=2, ls=":",
            label="random search", zorder=4)
    ax.axhline(1.0, color="red", lw=1, ls="--",
               label=f"Oracle best ({oracle_best:.3g})")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    p = out_dir / "best_cost_found.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(str(p))

    # first-hit curve: evaluations needed to reach a quality threshold
    # all_best: (n_seeds, max_iter+1) — running minimum cost per seed per iter
    thresholds = np.logspace(np.log10(0.05), np.log10(100), 200)  # 0.05 % … 5 %
    targets = oracle_best * (1 + thresholds / 100)
    hit_iters = np.full((len(bo_data), len(thresholds)), float(max_iter + 1))
    for ti, target in enumerate(targets):
        reached = all_best <= target           # (n_seeds, max_iter+1)
        ever    = np.any(reached, axis=1)      # (n_seeds,)
        first   = np.argmax(reached, axis=1)   # first True per seed (0 if never)
        hit_iters[:, ti] = np.where(ever, first, max_iter + 1)

    mean_hits  = np.mean(hit_iters,   axis=0)
    worst_hits = np.max(hit_iters,    axis=0)
    best_hits  = np.min(hit_iters,    axis=0)
    # q25_hits   = np.percentile(hit_iters, 25, axis=0)
    # q75_hits   = np.percentile(hit_iters, 75, axis=0)

    # random baseline: first k where expected_random_best[k] <= target
    random_hit_iters = np.full(len(thresholds), float(max_iter + 1))
    for ti, target in enumerate(targets):
        reached = random_baseline <= target
        if np.any(reached):
            random_hit_iters[ti] = float(np.argmax(reached))

    fig, ax = plt.subplots(figsize=(9, 4))
    # ax.fill_betweenx(thresholds, q25_hits, q75_hits, color="gray", alpha=0.2, label="IQR (25–75%)")
    ax.plot(mean_hits,  thresholds, color="black", lw=2,   label="mean across seeds")
    ax.plot(worst_hits, thresholds, color="firebrick",  lw=1.5, ls="--", label="worst seed")
    ax.plot(best_hits,  thresholds, color="seagreen",   lw=1.5, ls="--", label="best seed")
    ax.plot(random_hit_iters, thresholds, color="gray", lw=1.5, ls=":", label="random search")
    ax.set_xlabel("Evaluations to first reach threshold")
    ax.set_ylabel("Quality threshold  (% above oracle best)")
    ax.set_title(f"First-hit cost: evaluations needed per quality level  ({len(bo_data)} seeds)")
    ax.set_yscale("log")
    ax.set_ylim(thresholds[-1], thresholds[0])  # inverted: better (lower %) at top
    ax.set_xlim(0, max_iter * 1.05)
    tick_vals = [v for v in [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
                 if thresholds[0] <= v <= thresholds[-1]]
    ax.set_yticks(tick_vals)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"+{v:g}%"))
    ax.set_ylabel("Quality threshold (% above oracle best) — better ↑")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = out_dir / "first_hit_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(str(p))

    return out_paths


# ── Oracle group ───────────────────────────────────────────────────────────────
@dataclass
class OracleGroup:
    oracle_csv:  str | None   # None when no oracle is available
    oracle_best: float | None # None when no oracle is available
    seeds:       list         # [(name, csv_path), ...]
    out_dir:     Path         # common parent of all seed directories


def _load_oracle_group(oracle_path, bo_paths):
    oracle_df  = None
    dims       = None
    oracle_best = None
    if oracle_path:
        oracle_df = pd.read_csv(oracle_path)
        dims = _dim_cols(oracle_df)
        oracle_obs = oracle_df[oracle_df["valid"] == 1] if "valid" in oracle_df.columns else oracle_df
        oracle_obs = oracle_obs[oracle_obs["cost"].notna()]
        if oracle_obs.empty:
            print(f"ERROR: oracle {oracle_path} has no valid evaluated configs", file=sys.stderr)
            return None
        oracle_best = float(oracle_obs["cost"].min())

    seeds = []
    for path in bo_paths:
        df = _load_pool_csv(path)
        try:
            if oracle_path:
                assert_same_space(oracle_df, df, dims, path)
        except AssertionError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue
        if "eval_iter" not in df.columns:
            print(f"WARNING: {path} has no eval_iter column — skipping", file=sys.stderr)
            continue
        seeds.append((Path(path).stem, str(path)))

    if not seeds:
        print(f"ERROR: no valid BO pools for oracle {oracle_path}", file=sys.stderr)
        return None

    return OracleGroup(
        oracle_csv=str(oracle_path) if oracle_path else None,
        oracle_best=oracle_best,
        seeds=seeds,
        out_dir=Path(seeds[0][1]).parent.parent,
    )


# ── README generation ──────────────────────────────────────────────────────────
def generate_seed_readme(csv_path, scale, ax_x, ax_y, ax_f, oracle_group=None):
    if not _SEED_TEMPLATE_PATH.exists():
        return None
    template = _SEED_TEMPLATE_PATH.read_text()

    df = _load_pool_csv(csv_path)
    seed_dir     = Path(csv_path).parent
    seed_name    = seed_dir.name
    problem_name = seed_dir.parent.name
    n_total  = len(df)
    n_valid  = int(df["valid"].sum()) if "valid" in df.columns else "?"
    n_obs    = int(df["cost"].notna().sum())
    _valid     = df["valid"]    if "valid"    in df.columns else pd.Series(1, index=df.index)
    _attempted = df["eval_iter"].notna() if "eval_iter" in df.columns else pd.Series(False, index=df.index)
    _, failed_mask = cell_status(_valid, _attempted, df["cost"].notna())
    n_failed = int(failed_mask.sum())

    valid_iters = df["eval_iter"].dropna() if "eval_iter" in df.columns else pd.Series([], dtype=float)
    valid_iters = valid_iters[valid_iters < _SENTINEL_ITER]
    n_iters = int(valid_iters.max()) if not valid_iters.empty else "?"

    best_cost = df["cost"].min()
    best_str  = f"{best_cost:.4g}" if pd.notna(best_cost) else "?"

    if oracle_group is not None and oracle_group.oracle_best is not None and pd.notna(best_cost):
        oracle_best_str = f"{oracle_group.oracle_best:.4g}"
        gap_pct_str = f"{100 * (best_cost / oracle_group.oracle_best - 1):.1f}"
    else:
        oracle_best_str = "—"
        gap_pct_str = "—"

    subs = {
        "problem_name": problem_name,
        "seed_name":    seed_name,
        "n_total":      n_total,
        "n_valid":      n_valid,
        "n_obs":        n_obs,
        "n_failed":     n_failed,
        "n_iters":      n_iters,
        "best_cost":    best_str,
        "oracle_best":  oracle_best_str,
        "gap_pct":      gap_pct_str,
        "scale":        scale,
        "ax_x":         ax_x,
        "ax_y":         ax_y,
        "ax_f":         ax_f,
    }
    for key, val in subs.items():
        template = template.replace(f"${{{key}}}", str(val))

    out = seed_dir / "README_bo_synopsis.md"
    out.write_text(template)
    return str(out)


def generate_problem_readme(oracle_group, seed_csv_paths, scale, ax_x, ax_y, ax_f):
    if not _PROBLEM_TEMPLATE_PATH.exists():
        return None
    template = _PROBLEM_TEMPLATE_PATH.read_text()

    problem_name = oracle_group.out_dir.name
    n_seeds = len(seed_csv_paths)

    # Derive search space size from space.json sidecar (written by C++ BananasSearch dumper).
    # pool.csv only contains valid rows, so n_valid and n_total come from the JSON.
    import json as _json
    _space_json = Path(seed_csv_paths[0]).parent / "space.json"
    if _space_json.exists():
        with open(_space_json) as _f:
            _meta = _json.load(_f)
        n_total = _meta.get("total_size", "?")
        n_valid = _meta.get("n_valid", "?")
    else:
        first_df = _load_pool_csv(seed_csv_paths[0])
        n_valid = len(first_df)
        n_total = "?"

    max_iters = []
    bests = []
    for p in seed_csv_paths:
        df = _load_pool_csv(p)
        iters = df["eval_iter"].dropna() if "eval_iter" in df.columns else pd.Series([], dtype=float)
        iters = iters[iters < _SENTINEL_ITER]
        if not iters.empty:
            max_iters.append(int(iters.max()))
        bc = df["cost"].min()
        if pd.notna(bc):
            bests.append(float(bc))

    n_iters = max(max_iters) if max_iters else "?"

    if bests and pd.notna(oracle_group.oracle_best):
        ob = oracle_group.oracle_best
        mean_best = float(np.mean(bests))
        best_best = float(min(bests))
        gap_mean = f"{100 * (mean_best / ob - 1):.1f}"
        gap_best = f"{100 * (best_best / ob - 1):.1f}"
        oracle_best_str   = f"{ob:.4g}"
        best_cost_mean_str = f"{mean_best:.4g}"
        best_cost_best_str = f"{best_best:.4g}"
    else:
        oracle_best_str = "—"
        best_cost_mean_str = "—"
        best_cost_best_str = "—"
        gap_mean = "—"
        gap_best = "—"

    subs = {
        "problem_name":    problem_name,
        "n_total":         n_total,
        "n_valid":         n_valid,
        "n_seeds":         n_seeds,
        "n_iters":         n_iters,
        "oracle_best":     oracle_best_str,
        "best_cost_mean":  best_cost_mean_str,
        "best_cost_best":  best_cost_best_str,
        "gap_pct_mean":    gap_mean,
        "gap_pct_best":    gap_best,
        "scale":           scale,
        "ax_x":            ax_x,
        "ax_y":            ax_y,
        "ax_f":            ax_f,
    }
    for key, val in subs.items():
        template = template.replace(f"${{{key}}}", str(val))

    out = oracle_group.out_dir / "README_bo_synopsis.md"
    out.write_text(template)
    return str(out)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pool_csvs", nargs="*", metavar="pool.csv",
                    help="Standalone BO pool CSVs (heatmaps and calibration only, no oracle)")
    ap.add_argument("--oracle", nargs="+", action="append", metavar="CSV",
                    help="Oracle CSV followed by one or more BO pool CSVs (repeatable). "
                         "First is the exhaustive oracle, the rest are BO seeds.")
    ap.add_argument("--objective-scale", default="log10",
                    help="Cost transform matching surrogate training "
                         "(linear, log2, log10, ln, sqrt, cbrt; default: log10)")
    ap.add_argument("--axes", default="wramCol,wramRow,tasklets", metavar="X,Y,FACET",
                    help="Comma-separated pivot x,y,facet column names for heatmaps "
                         "(default: wramCol,wramRow,tasklets)")
    ap.add_argument("--pcts", type=float, nargs="+", default=[2, 5, 10, 15],
                    help="Top-k%% recall thresholds (default: 2 5 10 15)")
    ap.add_argument("--no-per-seed", action="store_true",
                    help="Skip per-seed heatmaps and learning-curve plots")
    ap.add_argument("--skip-existing-seed-plots", action="store_true",
                    help="Skip per-seed plots when pool_cost.png already exists "
                         "(problem-level aggregate plots are always regenerated)")
    ap.add_argument("--out-dir", dest="out_dir", required=True, metavar="DIR",
                    help="Output directory for aggregate (problem-level) plots and README")
    ap.add_argument("--plots", nargs="+", default=None, metavar="NAME",
                    help="Only generate plots whose tag contains one of these substrings "
                         "(e.g. --plots eval_time sigma_calibration cost)")
    args = ap.parse_args()

    scale = args.objective_scale
    ax_x, ax_y, ax_f = args.axes.split(",", 2)

    oracle_groups = []
    if args.oracle:
        for group_paths in args.oracle:
            if len(group_paths) < 2:
                print("ERROR: --oracle needs at least 2 files (oracle + BO pool)", file=sys.stderr)
                sys.exit(1)
            group = _load_oracle_group(group_paths[0], group_paths[1:])
            if group is not None:
                oracle_groups.append(group)

    # Group standalone pool CSVs by their grandparent directory (the problem-level
    # dir that contains multiple seed_* subdirs). This enables aggregate plots even
    # when no exhaustive oracle is available.
    seed_groups = []
    if args.pool_csvs:
        by_problem: dict[Path, list[str]] = {}
        for csv_path in args.pool_csvs:
            by_problem.setdefault(Path(csv_path).parent.parent, []).append(csv_path)
        for paths in by_problem.values():
            group = _load_oracle_group(None, sorted(paths))
            if group is not None:
                seed_groups.append(group)

    all_groups = oracle_groups + seed_groups
    if not all_groups:
        ap.print_help()
        sys.exit(0)

    out_dir = Path(args.out_dir)
    for group in all_groups:
        group.out_dir = out_dir

    all_futures = {}  # future → tag string
    _plot_filter = args.plots  # None = all; list[str] = substring allowlist

    with ProcessPoolExecutor() as executor:

        def sub(tag, fn, /, *a, **kw):
            """Submit fn(*a, **kw) only if tag matches the --plots filter."""
            if _plot_filter and not any(p in tag for p in _plot_filter):
                return
            all_futures[executor.submit(fn, *a, **kw)] = tag

        for group in all_groups:
            seed_csvs  = [p for _, p in group.seeds]
            seed_names = [n for n, _ in group.seeds]

            # ── Per-seed plots ──
            for name, csv_path in group.seeds:
                out_dir = str(Path(csv_path).parent)
                seed_is_new = not (args.skip_existing_seed_plots
                                   and os.path.exists(os.path.join(out_dir, "pool_cost.png")))

                if not args.no_per_seed and seed_is_new:
                    for kw in build_facet_plot_tasks(csv_path, scale, ax_x, ax_y, ax_f):
                        sub(f"{name}/{kw['metric']}", make_facet_plot, out_dir=out_dir, **kw)

                    val_path = Path(csv_path).parent / "validation.csv"
                    if val_path.exists():
                        for fn, tag in [(plot_validation_rmse, "validation_rmse"),
                                        (plot_validation_mape, "validation_mape")]:
                            sub(f"{name}/{tag}", fn, str(val_path), out_dir,
                                dataset="validation", scale=scale)

                    train_path = Path(csv_path).parent / "training.csv"
                    if train_path.exists():
                        for fn, tag in [(plot_validation_rmse, "training_rmse"),
                                        (plot_validation_mape, "training_mape")]:
                            sub(f"{name}/{tag}", fn, str(train_path), out_dir,
                                dataset="training", scale=scale)

                readme_is_new = not (args.skip_existing_seed_plots
                                     and (Path(csv_path).parent / "README_bo_synopsis.md").exists())
                if not args.no_per_seed and readme_is_new:
                    sub(f"{name}/seed_readme", generate_seed_readme,
                        csv_path, scale, ax_x, ax_y, ax_f, group)

            # ── Aggregate plots ──
            agg_dir = str(group.out_dir)

            # Oracle-specific plots (recall curves, best-cost vs oracle).
            if group.oracle_csv is not None:
                sub("recall+best_cost", _plot_oracle_curves,
                    group.oracle_csv, seed_csvs, agg_dir, args.pcts, scale)

            # Oracle-independent aggregate plots (generated for all groups).
            if len(seed_csvs) > 1:
                sub("sigma_calibration", _plot_sigma_calibration,
                    seed_csvs, agg_dir, scale, seed_names)
                sub("aggregate_learning_curves", _plot_aggregate_learning_curves,
                    seed_csvs, agg_dir, scale)
                sub("aggregate_timings", _plot_aggregate_timings,
                    seed_csvs, agg_dir)

            sub("eval_time_vs_cost", _plot_eval_time_vs_cost,
                seed_csvs, agg_dir, scale, seed_names)
            sub("eval_time_explainability", _plot_eval_time_explainability,
                seed_csvs, agg_dir)
            sub("problem_readme", generate_problem_readme,
                group, seed_csvs, scale, ax_x, ax_y, ax_f)

        for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="plots"):
            tag = all_futures[future]
            ex = future.exception()
            if ex:
                tqdm.write(f"Error in {tag}:")
                traceback.print_exception(ex)
            else:
                result = future.result()
                if isinstance(result, list):
                    for p in result:
                        if p:
                            tqdm.write(f"Saved: {p}")
                elif result:
                    tqdm.write(f"Saved: {result}")



if __name__ == "__main__":
    main()

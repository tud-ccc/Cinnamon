#!/usr/bin/env python3
"""
Visualise pool.csv as 2D heatmaps.
Facets: one subplot per unique tasklets value.
Axes:   tile_ (y) × tile_1 (x).
Outputs one figure per metric: cost, mu, sigma, acq.

Usage:
    python plot_pool.py [pool.csv] [out_dir]
Defaults: pool.csv in cwd, output next to the CSV.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


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


DPU = 1
RANK = 1

# ── Image builder ─────────────────────────────────────────────────────────────
def build_rgba(subset, tile0_vals, tile1_vals, val_col, norm, cmap, *, white_unvisited=True):
    """Return an RGBA image (H=tile_, W=tile_1) for one tasklets slice."""
    piv_val  = subset.pivot_table(index="tile_",  columns="tile_1", values=val_col,   aggfunc="mean")
    piv_vis  = subset.pivot_table(index="tile_",  columns="tile_1", values="visited", aggfunc="max")
    piv_cost = subset.pivot_table(index="tile_",  columns="tile_1", values="cost",    aggfunc="min")

    vals = piv_val.reindex(index=tile0_vals, columns=tile1_vals).to_numpy(dtype=float)
    vis  = piv_vis.reindex(index=tile0_vals, columns=tile1_vals).to_numpy(dtype=float)
    cost = piv_cost.reindex(index=tile0_vals, columns=tile1_vals).to_numpy(dtype=float)

    unvisited = np.isnan(vis) | (vis == 0)
    failed    = (~unvisited) & np.isnan(cost)
    no_data   = np.isnan(vals)

    fill = np.nanmedian(vals) if not np.all(no_data) else 0.0
    safe = np.where(no_data, fill, vals)

    img = cmap(norm(safe))
    img[no_data] = [1.0, 1.0, 1.0, 1.0]           # white — metric has no value
    if white_unvisited:
        img[unvisited] = [1.0, 1.0, 1.0, 1.0]     # white — not sampled (cost plot)
    img[failed] = [0.55, 0.55, 0.55, 1.0]          # gray  — evaluated but cost failed

    return img


# ── Constraint mask ────────────────────────────────────────────────────────────
# Returns True for invalid cells (should be greyed out).
# m = tile_, k = tile_1, T = tasklets.  Edit this formula as needed.
WRAM_LIMIT = 65536 / 4

def constraint_violated(m, k, T):
    return T * k * m / (RANK * DPU) + k + T * m / (RANK * DPU) > WRAM_LIMIT

def draw_constraint(ax, T, tile0_vals, tile1_vals):
    extent = [0.5, len(tile1_vals) + 0.5, 0.5, len(tile0_vals) + 0.5]
    m_grid, k_grid = np.meshgrid(tile0_vals, tile1_vals, indexing="ij")
    mask = constraint_violated(m_grid, k_grid, T)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = [0.75, 0.75, 0.75, 0.6]
    ax.imshow(overlay, origin="lower", aspect="auto", extent=extent, zorder=3)


# ── Figure factory ─────────────────────────────────────────────────────────────
def make_figure(agg, tile0_vals, tile1_vals, tasklet_vals, out_dir,
                metric, title, label, norm, cmap, *, white_unvisited=True):
    extent = [0.5, len(tile1_vals) + 0.5, 0.5, len(tile0_vals) + 0.5]
    ncols = 2
    nrows = int(np.ceil(len(tasklet_vals) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.5 * nrows),
                              squeeze=False)
    flat = axes.flatten()

    for ax in flat[len(tasklet_vals):]:
        ax.set_visible(False)

    for idx, (ax, T) in enumerate(zip(flat, tasklet_vals)):
        subset = agg[agg["tasklets"] == T]
        img = build_rgba(subset, tile0_vals, tile1_vals, metric, norm, cmap,
                         white_unvisited=white_unvisited)

        ax.imshow(img, origin="lower", aspect="auto", extent=extent)
        draw_constraint(ax, T, tile0_vals, tile1_vals)

        in_first_col = (idx % ncols == 0)
        in_bottom    = (idx >= (nrows - 1) * ncols)

        xtick_pos = range(1, len(tile1_vals) + 1)
        ytick_pos = range(1, len(tile0_vals) + 1)
        ax.set_xticks(xtick_pos)
        ax.set_yticks(ytick_pos)
        ax.set_xticklabels(tile1_vals if in_bottom else [], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tile0_vals if in_first_col else [], fontsize=7)
        ax.set_xlim(0, len(tile1_vals) + 0.5)
        ax.set_ylim(0, len(tile0_vals) + 0.5)
        if in_first_col:
            ax.set_ylabel("tile_")
        if in_bottom:
            ax.set_xlabel("tile_1")
        ax.set_title(f"tasklets = {T}", fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    # Add colorbar in its own axes on the right so it doesn't steal space
    # from the subplots or push the suptitle off the figure.
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=label)

    out_path = os.path.join(out_dir, f"pool_{metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path



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
                        color=cmap(i / len(tasklet_vals)),
                        lw=1, alpha=0.7, label=f"T={T}")
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
    """Plot surrogate RMSE on held-out validation points vs evaluations."""
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
    val["pct_err"] = 100 * np.abs(val["mu"] - scaled_cost) / scaled_cost
    val["x"] = val["iter"]
    return _plot_validation_metric(
        val,
        err_col="pct_err",
        agg_fn="mean",
        xlabel="Evaluations",
        ylabel="MAPE (%)",
        title=f"Surrogate {dataset} MAPE (cost in {scale_label(scale)} space)\n"
        "mean |predicted − true| / true × 100",
        out_path=os.path.join(out_dir, f"pool_{dataset}_mape.png"),
    )


META_COLS = {"visited", "valid", "cost", "eval_iter", "mu", "sigma", "acq"}

def _dim_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def _compute_min_dist(df, dims):
    """Min L1 distance in discrete grid-step indices to the nearest visited config."""
    index_maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in dims}
    coords = np.column_stack([df[c].map(index_maps[c]).values for c in dims]).astype(int)
    visited_mask = df["visited"].fillna(0).astype(bool).values
    visited_coords = coords[visited_mask]
    if len(visited_coords) == 0:
        return np.full(len(df), np.nan)
    # (N, V, D) → L1 sum → (N, V) → min → (N,)
    diffs = np.abs(coords[:, None, :] - visited_coords[None, :, :])
    return diffs.sum(axis=2).min(axis=1).astype(float)


def plot_sigma_vs_distance(df, out_dir, scale="log10"):
    """Median ± IQR of surrogate σ at each grid-step distance from the nearest observation."""
    dims = _dim_cols(df)
    if "sigma" not in df.columns:
        return None
    df = df.copy()
    df["min_dist"] = _compute_min_dist(df, dims)
    sub = df[df["sigma"].notna() & (df["valid"] == 1)]
    if sub.empty:
        return None

    dist_vals = sorted(sub["min_dist"].dropna().unique())
    medians, q25s, q75s, counts = [], [], [], []
    kept_dists = []
    for d in dist_vals:
        g = sub[sub["min_dist"] == d]["sigma"].values
        if len(g) == 0:
            continue
        kept_dists.append(d)
        medians.append(np.median(g))
        q25s.append(np.percentile(g, 25))
        q75s.append(np.percentile(g, 75))
        counts.append(len(g))

    xs = np.array(kept_dists)
    medians, q25s, q75s = map(np.array, (medians, q25s, q75s))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, medians, color="steelblue", lw=2, marker="o", ms=4, label="median σ")
    ax.fill_between(xs, q25s, q75s, alpha=0.25, color="steelblue", label="IQR")
    for d, m, n in zip(xs, medians, counts):
        ax.annotate(f"n={n}", (d, m), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6, color="gray")
    ax.set_xticks(xs.astype(int))
    ax.set_xlabel("Min grid-step distance to nearest observation")
    ax.set_ylabel(f"σ (surrogate uncertainty, {scale_label(scale)} units)")
    ax.set_title("Surrogate σ vs. distance from observations\n"
                 "Well-calibrated: σ increases monotonically with distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pool_sigma_vs_dist.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sigma_scatter(df, out_dir, scale="log10"):
    """Scatter of (min_dist, σ) coloured by μ — reveals over/under-confident regions."""
    dims = _dim_cols(df)
    if "sigma" not in df.columns or "mu" not in df.columns:
        return None
    df = df.copy()
    df["min_dist"] = _compute_min_dist(df, dims)
    sub = df[df["sigma"].notna() & df["mu"].notna() & (df["valid"] == 1)]
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        sub["min_dist"], sub["sigma"],
        c=sub["mu"], cmap="viridis_r",
        alpha=0.4, s=8, linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label=f"μ  (predicted {scale_label(scale)} cost — lower is better)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Min grid-step distance to nearest observation")
    ax.set_ylabel(f"σ (surrogate uncertainty, {scale_label(scale)} units)")
    ax.set_title("Calibration scatter: σ vs. distance from observations\n"
                 "Bottom-right = overconfident far from data  ·  Top-left = underfit near data")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pool_sigma_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_tasks(csv_path, scale="log10"):
    """Load one CSV and return (data_tuple, list_of_figure_kwargs)."""
    out_dir = str(Path(csv_path).parent)

    df = pd.read_csv(csv_path)
    df = df[df["dpus"] == DPU]
    # df = df[df['ranks'] == RANK]

    group_cols = ["tile_", "tile_1", "tasklets", "dpus"]
    agg = df.groupby(group_cols, as_index=False).agg(
        visited=("visited", "max"),
        cost=("cost", "min"),
        mu=("mu", "mean"),
        sigma=("sigma", "mean"),
        acq=("acq", "min"),
    )

    tile0_vals   = sorted(agg["tile_"].unique())
    tile1_vals   = sorted(agg["tile_1"].unique())
    tasklet_vals = sorted(agg["tasklets"].unique())
    data = (agg, tile0_vals, tile1_vals, tasklet_vals, out_dir)

    cost_vals = agg["cost"].dropna()
    tasks = [
        dict(metric="cost",
             title="Observed cost  [white = unsampled, gray = failed]",
             label="Simulated cost (log scale, lower is better)",
             norm=mcolors.LogNorm(vmin=cost_vals.min(), vmax=cost_vals.max()),
             cmap=plt.cm.viridis_r),
    ]
    sl = scale_label(scale)
    for metric, title, label, cmap in [
        ("mu",    f"Surrogate μ  ({sl} scale)",
         f"μ — predicted {sl}(cost)  (lower is better)", plt.cm.viridis_r),
        ("sigma", f"Surrogate σ  ({sl} scale)",
         f"σ — uncertainty in {sl}(cost)  (lower = more certain)", plt.cm.plasma),
        ("acq",   "Acquisition score  (lower = higher priority)",
         f"UCB acquisition  μ − κσ  ({sl} scale)", plt.cm.plasma_r),
    ]:
        vals = agg[metric].dropna()
        tasks.append(dict(
            metric=metric, title=title, label=label, cmap=cmap,
            norm=mcolors.Normalize(vmin=vals.min(), vmax=vals.max()),
            white_unvisited=False,
        ))
    return data, tasks


if __name__ == "__main__":
    import argparse
    import traceback

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_paths", nargs="*", default=["pool.csv"], metavar="pool.csv")
    ap.add_argument("--objective-scale", default="log10",
                    help="Cost transform used during surrogate training "
                         "(linear, log2, log10, ln, sqrt, cbrt)")
    args = ap.parse_args()
    scale = args.objective_scale
    csv_paths = args.csv_paths

    all_futures = {}
    with ProcessPoolExecutor() as executor:
        for csv_path in csv_paths:
            data, tasks = build_tasks(csv_path, scale)
            out_dir = str(Path(csv_path).parent)
            for kw in tasks:
                f = executor.submit(make_figure, *data, **kw)
                all_futures[f] = (csv_path, kw["metric"])

            val_path = Path(csv_path).parent / "validation.csv"
            if val_path.exists():
                for fn, tag in [(plot_validation_rmse, "validation_rmse"),
                                (plot_validation_mape, "validation_mape")]:
                    f = executor.submit(fn, str(val_path), out_dir, dataset="validation", scale=scale)
                    all_futures[f] = (csv_path, tag)

            train_rmse_path = Path(csv_path).parent / "training.csv"
            if train_rmse_path.exists():
                for fn, tag in [(plot_validation_rmse, "training_rmse"),
                                (plot_validation_mape, "training_mape")]:
                    f = executor.submit(fn, str(val_path), out_dir, dataset="training", scale=scale)
                    all_futures[f] = (csv_path, tag)

            raw_df = pd.read_csv(csv_path)
            raw_df = raw_df[raw_df["dpus"] == DPU]
            if "sigma" in raw_df.columns:
                for fn, tag in [(plot_sigma_vs_distance, "sigma_vs_dist"),
                                (plot_sigma_scatter,     "sigma_scatter")]:
                    f = executor.submit(fn, raw_df, out_dir, scale)
                    all_futures[f] = (csv_path, tag)

        for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="plots"):
            csv_path, metric = all_futures[future]
            ex = future.exception()
            if ex:
                tqdm.write(f"Error plotting {metric} for {csv_path}:")
                traceback.print_exception(ex)
            else:
                tqdm.write(f"Saved: {future.result()}")

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


def cell_status(valid, attempted, has_value):
    """Return (invalid, failed) boolean masks.

    Works with both numpy arrays (build_rgba) and pandas Series (generate_readme).
    - invalid:  constraint-violated (valid == 0)
    - failed:   actually attempted (eval_iter was set), valid, but produced no value
    """
    invalid = valid == 0
    failed  = ~has_value & ~invalid & (attempted > 0)
    return invalid, failed


# ── Image builder ─────────────────────────────────────────────────────────────
def build_rgba(
    subset, x, y, x_vals, y_vals, val_col, norm, cmap, *, white_unvisited=True
):
    """Return an RGBA image (H=tile_, W=tile_1) for one tasklets slice."""
    piv_val      = subset.pivot_table(index=x, columns=y, values=val_col,    aggfunc="mean")
    piv_attempted = subset.pivot_table(index=x, columns=y, values="attempted", aggfunc="max")
    piv_valid    = subset.pivot_table(index=x, columns=y, values="valid",    aggfunc="max")

    vals = piv_val.reindex(index=x_vals, columns=y_vals).to_numpy(dtype=float)
    attempted = np.nan_to_num(
        piv_attempted.reindex(index=x_vals, columns=y_vals).to_numpy(dtype=float)
    )
    valid = np.nan_to_num(
        piv_valid.reindex(index=x_vals, columns=y_vals).to_numpy(dtype=float)
    )

    no_data = np.isnan(vals)
    invalid, failed = cell_status(valid, attempted, ~no_data)

    img = cmap(norm(vals))
    img[no_data] = [1.0, 1.0, 1.0, 1.0]  # white — metric has no value
    img[invalid] = [0.80, 0.80, 0.80, 1.0]  # light gray — constraint violated
    img[failed]  = [0.55, 0.55, 0.55, 1.0]  # mid gray — evaluated but cost failed

    return img



# ── Figure factory ─────────────────────────────────────────────────────────────
def make_facet_plot(
    df, x, y, f, out_dir, metric, title, label, norm, cmap, *, white_unvisited=True
):
    tile0_vals   = sorted(df[x].unique())
    tile1_vals   = sorted(df[y].unique())
    facet_vals = sorted(df[f].unique())

    extent = [0.5, len(tile1_vals) + 0.5, 0.5, len(tile0_vals) + 0.5]
    ncols = 2
    nrows = int(np.ceil(len(facet_vals) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.5 * nrows),
                              squeeze=False)
    flat = axes.flatten()

    for ax in flat[len(facet_vals):]:
        ax.set_visible(False)

    for idx, (ax, T) in enumerate(zip(flat, facet_vals)):
        subset = df[df["tasklets"] == T]
        img = build_rgba(subset, x, y, tile0_vals, tile1_vals, metric, norm, cmap,
                         white_unvisited=white_unvisited)

        ax.imshow(img, origin="lower", aspect="auto", extent=extent)

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
    from scipy.spatial import KDTree
    index_maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in dims}
    coords = np.column_stack([df[c].map(index_maps[c]).values for c in dims]).astype(float)
    visited_mask = df["cost"].notna().values
    visited_coords = coords[visited_mask]
    if len(visited_coords) == 0:
        return np.full(len(df), np.nan)
    # KDTree query with p=1 (Manhattan) is O((N+V) log V) in memory O(V·D).
    tree = KDTree(visited_coords)
    dists, _ = tree.query(coords, k=1, p=1, workers=-1)
    return dists


def plot_sigma_vs_distance(df, out_dir, scale):
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


def plot_sigma_scatter(df, out_dir, scale):
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
    ax.set_title("Calibration scatter: σ vs. distance from observations")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pool_sigma_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_facet_plot_tasks(csv_path, scale, x, y, f):
    """Load one CSV and return (data_tuple, list_of_figure_kwargs)."""

    df = pd.read_csv(csv_path)

    xyf_axes = [x, y, f]
    agg = df.groupby(xyf_axes, as_index=False).agg(
        valid=("valid", "max"),
        attempted=("eval_iter", lambda x: x.notna().any()),
        cost=("cost", "min"),
        mu=("mu", "min"),
        sigma=("sigma", "mean"),
        acq=("acq", "min"),
    )

    base_parms = dict(df=agg, x=x, y=y, f=f)

    cost_vals = agg["cost"].dropna()
    tasks = [
        base_parms
        | dict(
            metric="cost",
            title="Min observed cost  [white = unsampled, gray = failed]",
            label="Simulated cost (log scale, lower is better)",
            norm=mcolors.LogNorm(vmin=cost_vals.min(), vmax=cost_vals.max()),
            cmap=plt.cm.viridis_r,
        ),
    ]
    sl = scale_label(scale)
    for metric, title, label, cmap in [
        ("mu",    f"Min surrogate μ  ({sl} scale)",
         f"μ — predicted {sl}(cost)  (lower is better)", plt.cm.viridis_r),
        ("sigma", f"Mean surrogate σ  ({sl} scale)",
         f"σ — uncertainty in {sl}(cost)  (lower = more certain)", plt.cm.plasma),
        ("acq",   "Best acquisition score  (lower = higher priority)",
         f"UCB acquisition  μ − κσ  ({sl} scale)", plt.cm.plasma_r),
    ]:
        vals = agg[metric].dropna()
        tasks.append(base_parms | dict(
            metric=metric, title=title, label=label, cmap=cmap,
            norm=mcolors.Normalize(vmin=vals.min(), vmax=vals.max()),
            white_unvisited=False,
        ))
    return tasks


_SENTINEL_ITER = 2**63
_TEMPLATE_PATH = Path(__file__).parent / "README_plot_synopsis.md"


def generate_readme(csv_path, scale, ax_x, ax_y, ax_f):
    if not _TEMPLATE_PATH.exists():
        return
    template = _TEMPLATE_PATH.read_text()

    df = pd.read_csv(csv_path)
    run_name = Path(csv_path).parent.name
    n_total  = len(df)
    n_valid  = int(df["valid"].sum()) if "valid" in df.columns else "?"
    n_obs    = int(df["cost"].notna().sum())
    _valid     = df["valid"]    if "valid"    in df.columns else pd.Series(1, index=df.index)
    _attempted = df["eval_iter"].notna() if "eval_iter" in df.columns else pd.Series(False, index=df.index)
    _, failed_mask = cell_status(_valid, _attempted, df["cost"].notna())
    n_failed = int(failed_mask.sum())

    valid_iters = df["eval_iter"].dropna() if "eval_iter" in df.columns else pd.Series([], dtype=float)
    valid_iters = valid_iters[valid_iters < _SENTINEL_ITER]
    n_iters  = int(valid_iters.max()) if not valid_iters.empty else "?"

    best_cost = df["cost"].min()
    best_str  = f"{best_cost:.4g}" if pd.notna(best_cost) else "?"

    subs = {
        "run_name":     run_name,
        "n_total":      n_total,
        "n_valid":      n_valid,
        "n_obs":        n_obs,
        "n_failed":     n_failed,
        "n_iters":      n_iters,
        "best_cost":    best_str,
        "oracle_best":  "—",
        "gap_pct":      "—",
        "scale":        scale,
        "ax_x":         ax_x,
        "ax_y":         ax_y,
        "ax_f":         ax_f,
    }
    for key, val in subs.items():
        template = template.replace(f"${{{key}}}", str(val))

    out = Path(csv_path).parent / "README_bo_synopsis.md"
    out.write_text(template)
    return str(out)


if __name__ == "__main__":
    import argparse
    import traceback

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_paths", nargs="*", default=["pool.csv"], metavar="pool.csv")
    ap.add_argument("--objective-scale", default="log10",
                    help="Cost transform used during surrogate training "
                         "(linear, log2, log10, ln, sqrt, cbrt)")
    ap.add_argument("--axes", default="tile_1,tile_,tasklets", metavar="X,Y,FACET",
                    help="Comma-separated x,y,facet column names for the heatmap "
                         "(default: tile_1,tile_,tasklets)")
    args = ap.parse_args()
    scale = args.objective_scale
    csv_paths = args.csv_paths
    ax_x, ax_y, ax_f = args.axes.split(",", 2)

    all_futures = {}
    with ProcessPoolExecutor() as executor:
        for csv_path in csv_paths:
            tasks = build_facet_plot_tasks(csv_path, scale, x=ax_x, y=ax_y, f=ax_f)
            out_dir = str(Path(csv_path).parent)
            for kw in tasks:
                f = executor.submit(make_facet_plot, out_dir=out_dir, **kw)
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

    for csv_path in csv_paths:
        out = generate_readme(csv_path, scale, ax_x, ax_y, ax_f)
        if out:
            print(f"Saved: {out}")

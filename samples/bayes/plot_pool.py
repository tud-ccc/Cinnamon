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
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# ── Args ──────────────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "pool.csv"
out_dir  = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(csv_path))
os.makedirs(out_dir, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)

# Aggregate over any extra dims (ranks, dpus, …) that are not the 3 we plot.
group_cols = ["tile_", "tile_1", "tasklets"]
agg = df.groupby(group_cols, as_index=False).agg(
    visited=("visited", "max"),
    cost=("cost", "min"),   # best observed cost across hardware configs
    mu=("mu", "mean"),
    sigma=("sigma", "mean"),
    acq=("acq", "min"),     # lowest (most promising) acquisition value
)

tile0_vals   = sorted(agg["tile_"].unique())
tile1_vals   = sorted(agg["tile_1"].unique())
tasklet_vals = sorted(agg["tasklets"].unique())

# ── Image builder ─────────────────────────────────────────────────────────────
def build_rgba(subset, val_col, norm, cmap, *, white_unvisited=True):
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
constraint_violated = lambda m, k, T: T * k * m + k + T * m > WRAM_LIMIT

def draw_constraint(ax, T):
    m_grid, k_grid = np.meshgrid(tile0_vals, tile1_vals, indexing="ij")
    mask = constraint_violated(m_grid, k_grid, T)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = [0.75, 0.75, 0.75, 0.6]
    ax.imshow(overlay, origin="lower", aspect="auto", zorder=3)


# ── Figure factory ─────────────────────────────────────────────────────────────
def make_figure(metric, title, label, norm, cmap, *, white_unvisited=True):
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
        img = build_rgba(subset, metric, norm, cmap,
                         white_unvisited=white_unvisited)

        ax.imshow(img, origin="lower", aspect="auto")
        draw_constraint(ax, T)
        ax.set_xticks(range(len(tile1_vals)))
        ax.set_yticks(range(len(tile0_vals)))

        in_first_col = (idx % ncols == 0)
        in_bottom    = (idx >= (nrows - 1) * ncols)

        ax.set_yticklabels(tile0_vals if in_first_col else [], fontsize=7)
        ax.set_xticklabels(
            tile1_vals if in_bottom else [],
            rotation=45, ha="right", fontsize=7,
        )
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
    print(f"Saved: {out_path}")


# ── Cost ──────────────────────────────────────────────────────────────────────
cost_vals = agg["cost"].dropna()
make_figure(
    "cost",
    title="Observed cost  [white = unsampled, gray = failed]",
    label="Simulated cost (log scale, lower is better)",
    norm=mcolors.LogNorm(vmin=cost_vals.min(), vmax=cost_vals.max()),
    cmap=plt.cm.viridis_r,
)

# ── Surrogate outputs (log10 space) ───────────────────────────────────────────
for metric, title, label, cmap in [
    ("mu",    "Surrogate μ  (log₁₀ scale)",
     "μ — predicted log₁₀(cost)  (lower is better)", plt.cm.viridis_r),
    ("sigma", "Surrogate σ  (log₁₀ scale)",
     "σ — uncertainty in log₁₀(cost)  (lower = more certain)", plt.cm.plasma),
    ("acq",   "Acquisition score  (lower = higher priority)",
     "UCB acquisition  μ − κσ  (log₁₀ scale)", plt.cm.plasma_r),
]:
    vals = agg[metric].dropna()
    make_figure(
        metric,
        title=title,
        label=label,
        norm=mcolors.Normalize(vmin=vals.min(), vmax=vals.max()),
        cmap=cmap,
        white_unvisited=False,  # show all configs with valid values; only gray out failed
    )

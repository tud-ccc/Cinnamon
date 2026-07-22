#!/usr/bin/env python3
"""
Analyze scatter_bench.cpp's results.csv (columns: num_dpus, blocks_per_dpu,
block_size, iter, ns). All latencies are converted to milliseconds up front
and reported/plotted in ms throughout.

Produces:
  - plots/latency_vs_block_size.png          (lines per num_dpus, fixed blocks_per_dpu)
  - plots/latency_vs_bytes_per_dpu.png        (points per num_dpus, every blocks_per_dpu/
                                               block_size split -- reveals whether blocks_per_dpu
                                               matters on its own, not just via bytes_per_dpu)
  - plots/latency_3d.html                     (interactive 3D: block_size x blocks_per_dpu x
                                               latency, colored by num_dpus -- open in a browser)
  - plots/latency_vs_num_dpus.png             (lines per block_size, fixed blocks_per_dpu)
  - plots/heatmap_blocks_vs_size.png          (blocks_per_dpu x block_size, fixed num_dpus)
  - plots/latency_vs_total_bytes.png          (one small panel per num_dpus value)
  - plots/latency_vs_bytes_per_dpu_faceted.png (one small panel per num_dpus value)
  - plots/regression_fit.png                  (measured vs. predicted, one subplot per template)
  - plots/regression_fit_best.png             (measured vs. predicted, best template only)
  - a regression table (stdout): several feature-transform templates fit by
    OLS and compared by RMSE/R^2, e.g. raw dims vs. log2(block_size) vs.
    log2(everything) vs. total_bytes -- same idea as reduce_cost's
    fit_overhead_term.py TEMPLATES dict, simplified to plain OLS. Every
    template's predictions are floored at the empirical minimum measured
    latency (see --floor) before scoring/plotting, since no call can ever be
    faster than that fixed per-call overhead. Also included: a "hinge" model
    (see --hinge-threshold) with one slope shared across all DPU counts on
    max(0, block_size - threshold) but a separate intercept per DPU count --
    matches the observed shape (flat below the threshold, linear above it
    with a slope that doesn't depend on DPU count).

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
    "quadratic (dpus, blocks, size)":
        lambda d: np.column_stack([
            d.num_dpus, d.blocks_per_dpu, d.block_size,
            d.num_dpus ** 2, d.blocks_per_dpu ** 2, d.block_size ** 2,
            d.num_dpus * d.blocks_per_dpu, d.num_dpus * d.block_size,
            d.blocks_per_dpu * d.block_size,
        ]),
    "log2(size)":
        lambda d: np.column_stack([d.num_dpus, d.blocks_per_dpu, np.log2(d.block_size)]),
    # "log2(dpus), log2(size)":
    #     lambda d: np.column_stack([np.log2(d.num_dpus), d.blocks_per_dpu, np.log2(d.block_size)]),
    "log2(dpus, blocks, size)":
        lambda d: np.column_stack([np.log2(d.num_dpus), np.log2(d.blocks_per_dpu), np.log2(d.block_size)]),
    "total_bytes":
        lambda d: np.column_stack([d.num_dpus * d.blocks_per_dpu * d.block_size]),
    "log2(total_bytes)":
        lambda d: np.column_stack([np.log2(d.num_dpus * d.blocks_per_dpu * d.block_size)]),
    # "log2(dpus), log2(bytes_per_dpu)":
    #     lambda d: np.column_stack([np.log2(d.num_dpus), np.log2(d.blocks_per_dpu * d.block_size)]),
}


def fit_ols(X: np.ndarray, y: np.ndarray, floor: float = 0.0) -> dict:
    """OLS with intercept, floored: a dpu_push_sg_xfer call can never
    complete faster than `floor` (the empirical minimum measured latency --
    fixed per-call overhead: rank dispatch, WRAM setup, etc.), but an
    unconstrained OLS fit happily predicts below it wherever the data itself
    is pinned at that floor (small transfers), which is exactly where it fit
    worst. So we fit the *excess* above the floor (y - floor) and rectify the
    fitted value back through the floor (floor + max(0, raw)) before scoring
    -- this can only pull predictions up to the floor in the region that
    used to undershoot it, never push them down elsewhere.
    """
    Xi = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Xi, y - floor, rcond=None)
    raw = Xi @ coef
    pred = floor + np.maximum(raw, 0.0)
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"pred": pred, "intercept": float(coef[0]), "coef": coef[1:],
             "rmse": rmse, "r2": r2, "floor": floor}


def fit_all_templates(data: pd.DataFrame, y: np.ndarray, floor: float = 0.0) -> tuple[pd.DataFrame, dict]:
    rows = []
    fits = {}
    for name, feature_fn in TEMPLATES.items():
        X = feature_fn(data)
        fit = fit_ols(X, y, floor=floor)
        fits[name] = fit
        rows.append({"template": name, "rmse_ms": fit["rmse"], "r2": fit["r2"]})
    table = pd.DataFrame(rows).sort_values("rmse_ms").reset_index(drop=True)
    return table, fits


DEFAULT_HINGE_THRESHOLD = 128.0  # bytes: below this, latency looks flat (dispatch-cost
                                  # dominated); above it, ~linear in block_size


def fit_hinge_fixed_effects(data: pd.DataFrame, y: np.ndarray, floor: float,
                            threshold: float = DEFAULT_HINGE_THRESHOLD) -> dict:
    """One shared slope on max(0, block_size - threshold), but a separate
    intercept per num_dpus value -- matches the observed shape (flat below
    the knee, then linear above it with a slope that doesn't depend on DPU
    count, i.e. per-DPU bandwidth isn't degraded by running more DPUs in
    parallel; what *does* vary per DPU count is the fixed dispatch overhead).

    Not expressible as a plain feature matrix for fit_ols (a per-group
    intercept needs one dummy column per num_dpus value, which is exactly
    collinear with the global intercept column fit_ols always prepends), so
    this uses the standard "fixed effects" / within-estimator trick instead:
    demean x and y by num_dpus group, fit the single shared slope on the
    demeaned data (group means contribute nothing to that regression), then
    recover each group's own intercept from its own mean afterwards.
    """
    x = np.maximum(data.block_size.to_numpy(dtype=float) - threshold, 0.0)
    y_excess = y - floor
    groups = data.num_dpus.to_numpy()

    tmp = pd.DataFrame({"g": groups, "x": x, "y": y_excess})
    x_tilde = tmp["x"] - tmp.groupby("g")["x"].transform("mean")
    y_tilde = tmp["y"] - tmp.groupby("g")["y"].transform("mean")
    denom = float((x_tilde ** 2).sum())
    slope = float((x_tilde * y_tilde).sum() / denom) if denom > 0 else 0.0

    group_mean_x = tmp.groupby("g")["x"].mean()
    group_mean_y = tmp.groupby("g")["y"].mean()
    dpu_intercepts = (group_mean_y - slope * group_mean_x).to_dict()

    raw = tmp["g"].map(dpu_intercepts).to_numpy() + slope * x
    pred = floor + np.maximum(raw, 0.0)
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"pred": pred, "intercept": float(np.mean(list(dpu_intercepts.values()))),
            "coef": np.array([slope]), "rmse": rmse, "r2": r2, "floor": floor,
            "dpu_intercepts": dpu_intercepts, "threshold": threshold}


# ── Plots ────────────────────────────────────────────────────────────────────

def _pow2_ticks(values) -> list[int]:
    """Powers of 2 spanning values' range, for colorbar/axis ticks."""
    lo, hi = float(np.min(values)), float(np.max(values))
    k_min = int(np.floor(np.log2(lo)))
    k_max = int(np.ceil(np.log2(hi)))
    return [2 ** k for k in range(k_min, k_max + 1)]


def _dpu_ticks(data: pd.DataFrame) -> list[int]:
    """Powers of 2 spanning data.num_dpus's range, for colorbar ticks."""
    return _pow2_ticks(data.num_dpus)



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
    cbar = fig.colorbar(sm, ax=ax, ticks=_dpu_ticks(sub), label="Number of DPUs")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_bytes_per_dpu(agg: pd.DataFrame, out_path: pathlib.Path):
    """Same style/colors as plot_latency_vs_block_size, but x = bytes_per_dpu
    (blocks_per_dpu * block_size), using every (blocks_per_dpu, block_size)
    split rather than one fixed blocks_per_dpu -- points, not connected
    lines, since several (blocks_per_dpu, block_size) pairs share the same
    bytes_per_dpu. If bytes_per_dpu alone explained latency, same-color
    points at the same x would collapse onto one curve; vertical spread at a
    fixed x means blocks_per_dpu matters on its own (e.g. a per-block
    dispatch cost), not just through the byte total it produces."""
    dpus = sorted(agg["num_dpus"].unique())
    norm = LogNorm(vmin=min(dpus), vmax=max(dpus))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in dpus:
        s = agg[agg["num_dpus"] == n]
        ax.scatter(s["bytes_per_dpu"], s["ms"], s=10, alpha=0.6, color=cmap(norm(n)))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("bytes per DPU (blocks_per_dpu × block_size)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Scatter latency vs. bytes per DPU (every blocks_per_dpu/block_size split)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, ticks=_dpu_ticks(agg), label="Number of DPUs")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_3d(agg: pd.DataFrame, out_path: pathlib.Path):
    """Interactive 3D scatter (block_size x blocks_per_dpu x latency,
    colored by num_dpus), written as a standalone HTML file via Plotly.

    Matplotlib's mplot3d does NOT reliably support log-scaled 3D axes --
    set_xscale/set_zscale exist but don't correctly rescale the 3D
    projection, which is what made the first version of this plot render
    with garbage axis scales. Plotly's WebGL 3D scene supports log axes
    directly and correctly, and as a bonus you get rotate/zoom/hover instead
    of a single fixed static angle -- exactly what's needed to tell apart
    points that land close together in a log-log-linear projection, and to
    check whether blocks_per_dpu shifts the point cloud independently of
    block_size (a separate sheet per blocks_per_dpu value) rather than
    everything collapsing onto one smooth surface.
    """
    import plotly.graph_objects as go

    dpu_ticks = _pow2_ticks(agg["num_dpus"])
    log_dpus = np.log2(agg["num_dpus"])

    fig = go.Figure(data=[go.Scatter3d(
        x=agg["block_size"], y=agg["blocks_per_dpu"], z=agg["ms"],
        mode="markers",
        marker=dict(
            size=4, opacity=0.7,
            color=log_dpus, colorscale="Viridis",
            colorbar=dict(title="Number of DPUs",
                          tickvals=np.log2(dpu_ticks),
                          ticktext=[f"{t:g}" for t in dpu_ticks]),
        ),
        customdata=agg[["num_dpus", "blocks_per_dpu", "block_size", "ms"]],
        hovertemplate=(
            "num_dpus=%{customdata[0]}<br>"
            "blocks_per_dpu=%{customdata[1]}<br>"
            "block_size=%{customdata[2]} bytes<br>"
            "latency=%{customdata[3]:.4g} ms<extra></extra>"
        ),
    )])
    fig.update_layout(
        title="Scatter latency: block size × blocks per DPU × latency",
        scene=dict(
            xaxis=dict(title="block size (bytes)", type="log"),
            yaxis=dict(title="blocks per DPU"),
            zaxis=dict(title="latency (ms)", type="log"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


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


def plot_latency_vs_x_faceted(agg: pd.DataFrame, x_col: str, x_label: str,
                              title: str, out_path: pathlib.Path):
    """One small subplot per num_dpus value (not colored by it, faceted on
    it), latency vs. x_col, points colored by block_size. Every subplot
    shares the same x/y limits, so this shows whether x_col alone collapses
    latency onto (roughly) the same curve regardless of dpu count, or
    whether the curves still shift from panel to panel."""
    dpu_values = sorted(agg["num_dpus"].unique())
    n = len(dpu_values)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols

    size_norm = LogNorm(vmin=agg["block_size"].min(), vmax=agg["block_size"].max())
    cmap = plt.get_cmap("plasma")

    pad = 1.15
    x_lo, x_hi = agg[x_col].min() / pad, agg[x_col].max() * pad
    y_lo, y_hi = agg["ms"].min() / pad, agg["ms"].max() * pad

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows),
                             squeeze=False, constrained_layout=True)

    sc = None
    for i, n_dpu in enumerate(dpu_values):
        ax = axes[divmod(i, ncols)[0]][divmod(i, ncols)[1]]
        sub = agg[agg["num_dpus"] == n_dpu]
        sc = ax.scatter(sub[x_col], sub["ms"], s=6, alpha=0.6,
                        c=sub["block_size"], cmap=cmap, norm=size_norm)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"dpus={n_dpu}", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[divmod(j, ncols)[0]][divmod(j, ncols)[1]].axis("off")

    fig.supxlabel(x_label)
    fig.supylabel("latency (ms)")
    fig.suptitle(title)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label="block size (bytes)")
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

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

    dpu_ticks = _dpu_ticks(data)

    # Shared axis limits across every subplot for direct comparability. Some
    # templates (plain linear OLS on raw dims, mostly) extrapolate to
    # negative "predicted latency" for some rows -- meaningless on a log
    # scale, so only positive predictions inform the shared range; those
    # points just won't render on any subplot, matching their invalidity.
    all_pred = np.concatenate([fits[nm]["pred"] for nm in names])
    positive = np.concatenate([all_pred[all_pred > 0], y[y > 0]])
    pad = 1.15
    lo = positive.min() / pad
    hi = positive.max() * pad

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


def plot_best_regression_fit(data: pd.DataFrame, y: np.ndarray, fit: dict,
                             name: str, out_path: pathlib.Path):
    """Same square measured-vs-predicted style as plot_all_regression_fits,
    but just the single best template, as its own standalone plot."""
    pred = fit["pred"]
    positive = np.concatenate([pred[pred > 0], y[y > 0]])
    pad = 1.15
    lo = positive.min() / pad
    hi = positive.max() * pad

    norm = LogNorm(vmin=data.num_dpus.min(), vmax=data.num_dpus.max())
    dpu_ticks = _dpu_ticks(data)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pred, y, s=8, alpha=0.5, c=data.num_dpus, cmap="viridis", norm=norm)
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("predicted latency (ms)")
    ax.set_ylabel("measured latency (ms)")
    ax.set_title(f"Best fit: {name}  (R²={fit['r2']:.3f}, RMSE={fit['rmse']:.4g} ms)",
                color="red", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    cbar = fig.colorbar(sc, ax=ax, ticks=dpu_ticks, label="Number of DPUs")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
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
    parser.add_argument("--floor", type=float, default=None,
                        help="minimum measurable latency (ms), used to floor every regression's "
                             "predictions -- a dpu_push_sg_xfer call can never be faster than "
                             "this fixed per-call overhead (default: the minimum ms value in the data)")
    parser.add_argument("--hinge-threshold", type=float, default=DEFAULT_HINGE_THRESHOLD,
                        help="block_size (bytes) knee for the hinge template: latency is treated "
                             f"as flat below it, linear above it (default: {DEFAULT_HINGE_THRESHOLD:g})")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"{args.csv}: no rows (empty or header-only)")
    df["ms"] = df["ns"] / 1e6

    agg = (df.groupby(["num_dpus", "blocks_per_dpu", "block_size"])["ms"]
             .mean().reset_index())
    agg["total_bytes"] = agg.num_dpus * agg.blocks_per_dpu * agg.block_size
    agg["bytes_per_dpu"] = agg.blocks_per_dpu * agg.block_size

    blocks_per_dpu = args.blocks_per_dpu or int(agg["blocks_per_dpu"].max())
    num_dpus = args.num_dpus or int(agg["num_dpus"].max())
    out_dir = pathlib.Path(args.out_dir)

    plot_latency_vs_block_size(agg, blocks_per_dpu, out_dir / "latency_vs_block_size.png")
    plot_latency_vs_bytes_per_dpu(agg, out_dir / "latency_vs_bytes_per_dpu.png")
    plot_latency_3d(agg, out_dir / "latency_3d.png")
    plot_latency_vs_num_dpus(agg, blocks_per_dpu, out_dir / "latency_vs_num_dpus.png")
    plot_heatmap(agg, num_dpus, out_dir / "heatmap_blocks_vs_size.png")
    plot_latency_vs_x_faceted(agg, "total_bytes", "total bytes (num_dpus × blocks_per_dpu × block_size)",
                              "Scatter latency vs. total bytes, one panel per DPU count",
                              out_dir / "latency_vs_total_bytes.png")
    plot_latency_vs_x_faceted(agg, "bytes_per_dpu", "bytes per DPU (blocks_per_dpu × block_size)",
                              "Scatter latency vs. bytes per DPU, one panel per DPU count",
                              out_dir / "latency_vs_bytes_per_dpu_faceted.png")

    y = agg["ms"].to_numpy(dtype=float)
    floor = args.floor if args.floor is not None else float(y.min())
    print(f"Floor (min measured latency, used to clamp every template's predictions): {floor:.4g} ms")
    table, fits = fit_all_templates(agg, y, floor=floor)

    hinge_name = f"hinge (shared slope, per-dpu intercept, knee={args.hinge_threshold:g}B)"
    fits[hinge_name] = fit_hinge_fixed_effects(agg, y, floor, threshold=args.hinge_threshold)
    table = pd.concat([table, pd.DataFrame([{
        "template": hinge_name,
        "rmse_ms": fits[hinge_name]["rmse"],
        "r2": fits[hinge_name]["r2"],
    }])], ignore_index=True).sort_values("rmse_ms").reset_index(drop=True)

    print(f"\n=== Regression templates (n={len(agg)} aggregated configs) ===")
    print(table.to_string(index=False))

    best_name = table.iloc[0]["template"]
    plot_all_regression_fits(agg, y, fits, table, best_name, out_dir / "regression_fit.png")
    plot_best_regression_fit(agg, y, fits[best_name], best_name, out_dir / "regression_fit_best.png")
    print(f"\nBest fit: {best_name}")
    best_fit = fits[best_name]
    if "dpu_intercepts" in best_fit:
        print(f"  shared slope (ms per byte above {best_fit['threshold']:g}B) = {best_fit['coef'][0]:.4g}")
        print("  per-dpu-count intercept (excess above floor, ms):")
        for n_dpu, val in sorted(best_fit["dpu_intercepts"].items()):
            print(f"    dpus={n_dpu:<5d} {val:.4g}")
    else:
        print(f"  intercept (excess above floor) = {best_fit['intercept']:.4g}")
        print(f"  coef = {[f'{c:.4g}' for c in best_fit['coef']]}")
    print(f"\nPlots written to {out_dir}/")


if __name__ == "__main__":
    main()

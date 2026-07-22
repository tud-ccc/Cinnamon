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
                                               *measured* latency, colored by num_dpus)
  - plots/latency_3d_predicted_best.html      (same 3D view, z = best template's prediction)
  - plots/latency_3d_predicted_hinge.html     (same 3D view, z = hinge template's prediction --
                                               open in a browser)
  - plots/latency_vs_num_dpus.png             (lines per block_size, fixed blocks_per_dpu)
  - plots/heatmap_blocks_vs_size.png          (blocks_per_dpu x block_size, fixed num_dpus)
  - plots/latency_vs_total_bytes.png          (one small panel per num_dpus value)
  - plots/latency_vs_bytes_per_dpu_faceted.png (one small panel per num_dpus value)
  - plots/regression_fit.png                  (measured vs. predicted, one subplot per template)
  - plots/regression_fit_best.png             (measured vs. predicted, best template only)
  - a regression table (stdout): several feature-transform templates fit by
    weighted least squares (weight = 1/measured_ms^2) and ranked by
    relative_rmse (RMS of (pred-measured)/measured), not flat ms RMSE --
    latency spans ~0.18ms to tens of ms here, so a 1ms error is huge at the
    low end and negligible at the high end; flat RMSE (and an unweighted
    fit) would be dominated by the handful of largest-latency configs and
    say nothing about how well small transfers are predicted. ms RMSE and R^2
    (on absolute error) are still reported alongside for reference. Same idea
    as reduce_cost's fit_overhead_term.py TEMPLATES dict (which also uses a
    cost-weighted fit for the same reason), simplified to weighted OLS. Every
    template's predictions are floored at the empirical minimum measured
    latency (see --floor) before scoring/plotting, since no call can ever be
    faster than that fixed per-call overhead. Also included: a "hinge" model
    (see --hinge-threshold) with one slope shared across all DPU counts on
    max(0, block_size - threshold) but a separate intercept per DPU count --
    matches the observed shape (flat below the threshold, linear above it
    with a slope that doesn't depend on DPU count); and a "log2(dpus) +
    quadratic(blocks, size)" template matching the observation that each DPU
    count's curve looks like the same t(blocks, size) shifted by a fixed
    a*log2(dpus) offset. --dpu-split additionally fits every template
    separately on num_dpus <= threshold and num_dpus > threshold, to compare
    which template wins in each regime (e.g. if a single template can't fit
    both very low and very high DPU counts well); it also adds a "hybrid"
    template (--hybrid-low-template/--hybrid-high-template) that uses the
    low-regime template's own fit below the split and the high-regime
    template's own fit above it, competing alongside every other
    whole-dataset template for best_name (and so also appearing in
    regression_fit.png/regression_fit_best.png if it wins).

Usage:
  python3 analyze.py results.csv
  python3 analyze.py results.csv --out-dir plots --blocks-per-dpu 24 --num-dpus 2048
  python3 analyze.py results.csv --dpu-split 32
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
    "linear (dpus, blocks, size)": lambda d: np.column_stack(
        [d.num_dpus, d.blocks_per_dpu, d.block_size]
    ),
    "threeway": lambda d: np.column_stack(
        [
            d.num_dpus,
            d.blocks_per_dpu,
            d.block_size,
            d.num_dpus * d.blocks_per_dpu,
            d.num_dpus * d.block_size,
            d.blocks_per_dpu * d.block_size,
            d.blocks_per_dpu * d.block_size * d.num_dpus,
        ]
    ),
    "quadratic": lambda d: np.column_stack(
        [
            d.num_dpus,
            d.blocks_per_dpu,
            d.block_size,
            d.num_dpus**2,
            d.blocks_per_dpu**2,
            d.block_size**2,
            d.num_dpus * d.blocks_per_dpu,
            d.num_dpus * d.block_size,
            d.blocks_per_dpu * d.block_size,
        ]
    ),
    "log2(size)": lambda d: np.column_stack(
        [d.num_dpus, d.blocks_per_dpu, np.log2(d.block_size)]
    ),
    "log2(dpus), log2(size)": lambda d: np.column_stack(
        [np.log2(d.num_dpus), d.blocks_per_dpu, np.log2(d.block_size)]
    ),
    "log2(dpus, blocks, size)": lambda d: np.column_stack(
        [np.log2(d.num_dpus), np.log2(d.blocks_per_dpu), np.log2(d.block_size)]
    ),
    "total_bytes": lambda d: np.column_stack(
        [d.num_dpus * d.blocks_per_dpu * d.block_size]
    ),
    "log2(total_bytes)": lambda d: np.column_stack(
        [np.log2(d.num_dpus * d.blocks_per_dpu * d.block_size)]
    ),
    "compound2": lambda d: np.column_stack(
        [
            np.log2(d.num_dpus),
            # d.blocks_per_dpu,
            d.blocks_per_dpu * d.block_size,
            # d.num_dpus * d.blocks_per_dpu * d.block_size,
        ]
    ),
    "compound": lambda d: np.column_stack(
        [
            np.log2(d.num_dpus),
            d.blocks_per_dpu,
            d.num_dpus * d.blocks_per_dpu * d.block_size,
        ]
    ),
    # Matches the "each dpu count is the same curve, shifted by a*log2(dpus)"
    # observation on the *measured* data: a single log2(dpus) offset term,
    # plus a full 2D quadratic (incl. the blocks_per_dpu*block_size cross
    # term) for the dpus-independent part t(blocks, size).
    "log2(dpus) + quadratic(blocks, size)": lambda d: np.column_stack(
        [
            np.log2(d.num_dpus),
            d.blocks_per_dpu,
            d.block_size,
            d.blocks_per_dpu**2,
            d.block_size**2,
            d.blocks_per_dpu * d.block_size,
        ]
    ),
}


def relative_rmse(resid: np.ndarray, y: np.ndarray) -> float:
    """RMS of (resid/y): unlike a flat ms RMSE, a 1ms error is scored the
    same whether the true latency is 0.2ms (500% off) or 30ms (3% off).
    This weights every config by how significant its error actually is,
    which flat RMSE does not -- a template can look great on ms-RMSE purely
    by nailing the handful of huge high-byte configs while being way off
    (relatively) on every small-transfer one, since ms-RMSE is dominated by
    the largest latencies in the dataset."""
    return float(np.sqrt(np.mean((resid / y) ** 2)))


def fit_ols(
    X: np.ndarray, y: np.ndarray, floor: float = 0.0, weighted: bool = True
) -> dict:
    """OLS with intercept, floored: a dpu_push_sg_xfer call can never
    complete faster than `floor` (the empirical minimum measured latency --
    fixed per-call overhead: rank dispatch, WRAM setup, etc.), but an
    unconstrained OLS fit happily predicts below it wherever the data itself
    is pinned at that floor (small transfers), which is exactly where it fit
    worst. So we fit the *excess* above the floor (y - floor) and rectify the
    fitted value back through the floor (floor + max(0, raw)) before scoring
    -- this can only pull predictions up to the floor in the region that
    used to undershoot it, never push them down elsewhere.

    weighted=True (default) fits via weighted least squares with weight =
    1/y^2 -- i.e. it directly optimizes relative error (see relative_rmse),
    not just reports it after an absolute-error fit. Without this, the fit
    itself (not just the RMSE-based ranking) would be dominated by the
    largest latencies, the same problem relative_rmse is meant to fix.
    """
    Xi = np.column_stack([np.ones(len(y)), X])
    target = y - floor
    if weighted:
        sqrt_w = 1.0 / y
        coef, *_ = np.linalg.lstsq(
            Xi * sqrt_w[:, None], target * sqrt_w, rcond=None
        )
    else:
        coef, *_ = np.linalg.lstsq(Xi, target, rcond=None)
    raw = Xi @ coef
    pred = floor + np.maximum(raw, 0.0)
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    rel_rmse = relative_rmse(resid, y)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "pred": pred,
        "intercept": float(coef[0]),
        "coef": coef[1:],
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "r2": r2,
        "floor": floor,
    }


def fit_all_templates(
    data: pd.DataFrame, y: np.ndarray, floor: float = 0.0
) -> tuple[pd.DataFrame, dict]:
    rows = []
    fits = {}
    for name, feature_fn in TEMPLATES.items():
        X = feature_fn(data)
        fit = fit_ols(X, y, floor=floor)
        fits[name] = fit
        rows.append(
            {
                "template": name,
                "rel_rmse": fit["rel_rmse"],
                "rmse_ms": fit["rmse"],
                "r2": fit["r2"],
            }
        )
    table = pd.DataFrame(rows).sort_values("rel_rmse").reset_index(drop=True)
    return table, fits


DEFAULT_HINGE_THRESHOLD = 128.0  # bytes: below this, latency looks flat (dispatch-cost
# dominated); above it, ~linear in block_size


def fit_hinge_fixed_effects(
    data: pd.DataFrame,
    y: np.ndarray,
    floor: float,
    threshold: float = DEFAULT_HINGE_THRESHOLD,
) -> dict:
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
    recover each group's own intercept from its own mean afterwards. As in
    fit_ols, this is weighted by 1/y^2 throughout (weighted group means,
    weighted slope) so it optimizes relative error too, not absolute ms
    error -- otherwise this template would be on a different, easier-to-win
    footing than every fit_ols-based template when ranked by relative_rmse.
    """
    x = np.maximum(data.block_size.to_numpy(dtype=float) - threshold, 0.0)
    y_excess = y - floor
    groups = data.num_dpus.to_numpy()
    w = 1.0 / y**2

    tmp = pd.DataFrame({"g": groups, "x": x, "y": y_excess, "w": w})
    tmp["wx"] = tmp["w"] * tmp["x"]
    tmp["wy"] = tmp["w"] * tmp["y"]
    group_w_sum = tmp.groupby("g")["w"].transform("sum")
    x_mean_w = tmp.groupby("g")["wx"].transform("sum") / group_w_sum
    y_mean_w = tmp.groupby("g")["wy"].transform("sum") / group_w_sum
    x_tilde = tmp["x"] - x_mean_w
    y_tilde = tmp["y"] - y_mean_w
    denom = float((tmp["w"] * x_tilde**2).sum())
    slope = float((tmp["w"] * x_tilde * y_tilde).sum() / denom) if denom > 0 else 0.0

    group_mean_x = tmp.groupby("g").apply(
        lambda d: np.average(d["x"], weights=d["w"]), include_groups=False
    )
    group_mean_y = tmp.groupby("g").apply(
        lambda d: np.average(d["y"], weights=d["w"]), include_groups=False
    )
    dpu_intercepts = (group_mean_y - slope * group_mean_x).to_dict()

    raw = tmp["g"].map(dpu_intercepts).to_numpy() + slope * x
    pred = floor + np.maximum(raw, 0.0)
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    rel_rmse = relative_rmse(resid, y)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "pred": pred,
        "intercept": float(np.mean(list(dpu_intercepts.values()))),
        "coef": np.array([slope]),
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "r2": r2,
        "floor": floor,
        "dpu_intercepts": dpu_intercepts,
        "threshold": threshold,
    }


# ── Plots ────────────────────────────────────────────────────────────────────


def _pow2_ticks(values) -> list[int]:
    """Powers of 2 spanning values' range, for colorbar/axis ticks."""
    lo, hi = float(np.min(values)), float(np.max(values))
    k_min = int(np.floor(np.log2(lo)))
    k_max = int(np.ceil(np.log2(hi)))
    return [2**k for k in range(k_min, k_max + 1)]


def _dpu_ticks(data: pd.DataFrame) -> list[int]:
    """Powers of 2 spanning data.num_dpus's range, for colorbar ticks."""
    return _pow2_ticks(data.num_dpus)


def plot_latency_vs_block_size(
    agg: pd.DataFrame, blocks_per_dpu: int, out_path: pathlib.Path
):
    sub = agg[agg["blocks_per_dpu"] == blocks_per_dpu]
    if sub.empty:
        return
    dpus = sorted(sub["num_dpus"].unique())
    norm = LogNorm(vmin=min(dpus), vmax=max(dpus))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in dpus:
        s = sub[sub["num_dpus"] == n].sort_values("block_size")
        ax.plot(
            s["block_size"],
            s["ms"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=cmap(norm(n)),
        )
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
    ax.set_title(
        "Scatter latency vs. bytes per DPU (every blocks_per_dpu/block_size split)"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, ticks=_dpu_ticks(agg), label="Number of DPUs")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_latency_3d_html(
    agg: pd.DataFrame, z: np.ndarray, z_name: str, title: str, out_path: pathlib.Path
):
    """Shared Plotly implementation: block_size x blocks_per_dpu x `z`,
    colored by num_dpus, written as a standalone interactive HTML file.

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

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=agg["block_size"],
                y=agg["blocks_per_dpu"],
                z=z,
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color=log_dpus,
                    colorscale="Viridis",
                    colorbar=dict(
                        title="Number of DPUs",
                        tickvals=np.log2(dpu_ticks),
                        ticktext=[f"{t:g}" for t in dpu_ticks],
                    ),
                ),
                customdata=np.column_stack(
                    [agg["num_dpus"], agg["blocks_per_dpu"], agg["block_size"], z]
                ),
                hovertemplate=(
                    "num_dpus=%{customdata[0]}<br>"
                    "blocks_per_dpu=%{customdata[1]}<br>"
                    "block_size=%{customdata[2]} bytes<br>"
                    f"{z_name}=" + "%{customdata[3]:.4g} ms<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="block size (bytes)", type="log"),
            yaxis=dict(title="blocks per DPU"),
            zaxis=dict(title=f"{z_name} (ms)", type="log"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def plot_latency_3d(agg: pd.DataFrame, out_path: pathlib.Path):
    """Interactive 3D scatter of *measured* latency."""
    _plot_latency_3d_html(
        agg,
        agg["ms"].to_numpy(),
        "measured latency",
        "Measured latency: block size × blocks per DPU × latency",
        out_path,
    )


def plot_predicted_latency_3d(
    agg: pd.DataFrame, pred: np.ndarray, template_name: str, out_path: pathlib.Path
):
    """Same 3D view as plot_latency_3d, but z = a regression template's
    *predicted* latency instead of the measured value. Comparing this
    against plot_latency_3d's measured point cloud shows what shape each
    model actually assumes -- e.g. the hinge model's surface is flat along
    blocks_per_dpu (it has no blocks_per_dpu term at all) and only bends
    along block_size, so if the measured cloud visibly fans out along
    blocks_per_dpu instead, that mismatch is exactly why it fits poorly."""
    _plot_latency_3d_html(
        agg,
        pred,
        "predicted latency",
        f"Predicted latency ({template_name}): block size × blocks per DPU × latency",
        out_path,
    )


def plot_latency_vs_num_dpus(
    agg: pd.DataFrame, blocks_per_dpu: int, out_path: pathlib.Path
):
    sub = agg[agg["blocks_per_dpu"] == blocks_per_dpu]
    if sub.empty:
        return
    sizes = sorted(sub["block_size"].unique())
    norm = LogNorm(vmin=min(sizes), vmax=max(sizes))
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(7, 5))
    for sz in sizes:
        s = sub[sub["block_size"] == sz].sort_values("num_dpus")
        ax.plot(
            s["num_dpus"],
            s["ms"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=cmap(norm(sz)),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of DPUs")
    ax.set_ylabel("latency (ms)")
    ax.set_title(
        f"Scatter latency vs. number of DPUs (blocks_per_dpu={blocks_per_dpu})"
    )
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
    im = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.values,
        norm=LogNorm(vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values)),
        cmap="viridis",
        shading="nearest",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block size (bytes)")
    ax.set_ylabel("blocks per DPU")
    ax.set_title(f"Scatter latency (ms), num_dpus={num_dpus}")
    fig.colorbar(im, ax=ax, label="latency (ms)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_x_faceted(
    agg: pd.DataFrame, x_col: str, x_label: str, title: str, out_path: pathlib.Path
):
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

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    sc = None
    for i, n_dpu in enumerate(dpu_values):
        ax = axes[divmod(i, ncols)[0]][divmod(i, ncols)[1]]
        sub = agg[agg["num_dpus"] == n_dpu]
        sc = ax.scatter(
            sub[x_col],
            sub["ms"],
            s=6,
            alpha=0.6,
            c=sub["block_size"],
            cmap=cmap,
            norm=size_norm,
        )
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
        cbar = fig.colorbar(
            sc, ax=axes.ravel().tolist(), shrink=0.8, label="block size (bytes)"
        )
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_regression_fits(
    data: pd.DataFrame,
    y: np.ndarray,
    fits: dict,
    table: pd.DataFrame,
    best_name: str,
    out_path: pathlib.Path,
):
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
    gs = fig.add_gridspec(
        nrows, ncols + 1, width_ratios=[1] * ncols + [0.06], wspace=0.1, hspace=0.15
    )

    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        fit = fits[name]
        ax.scatter(
            fit["pred"], y, s=8, alpha=0.5, c=data.num_dpus, cmap=cmap, norm=norm
        )
        ax.plot(
            [lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x"
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("predicted latency (ms)")
        ax.set_ylabel("measured latency (ms)")
        is_best = name == best_name
        ax.set_title(
            f"{name}\nrelRMSE={fit['rel_rmse'] * 100:.1f}%, R²={fit['r2']:.3f}, "
            f"RMSE={fit['rmse']:.3g}ms",
            color="red" if is_best else "black",
            fontweight="bold" if is_best else "normal",
        )
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


def plot_best_regression_fit(
    data: pd.DataFrame, y: np.ndarray, fit: dict, name: str, out_path: pathlib.Path
):
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
    ax.plot(
        [lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("predicted latency (ms)")
    ax.set_ylabel("measured latency (ms)")
    ax.set_title(
        f"Best fit: {name}\n"
        f"relRMSE={fit['rel_rmse'] * 100:.1f}%, R²={fit['r2']:.3f}, "
        f"RMSE={fit['rmse']:.3g}ms",
        color="red",
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    cbar = fig.colorbar(sc, ax=ax, ticks=dpu_ticks, label="Number of DPUs")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_and_report(
    agg: pd.DataFrame, floor: float, hinge_threshold: float, label: str
) -> tuple[pd.DataFrame, dict, str]:
    """Fit every TEMPLATES entry plus the hinge model on `agg`, print a
    labeled summary table, and return (table, fits, hinge_name) so callers
    can plot/inspect the winner. Split out of main() so it can be called
    once on the whole dataset and again on DPU-count subsets (--dpu-split)
    to compare which template wins in each regime."""
    y = agg["ms"].to_numpy(dtype=float)
    table, fits = fit_all_templates(agg, y, floor=floor)

    hinge_name = f"hinge (shared slope, per-dpu intercept, knee={hinge_threshold:g}B)"
    fits[hinge_name] = fit_hinge_fixed_effects(agg, y, floor, threshold=hinge_threshold)
    table = (
        pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        {
                            "template": hinge_name,
                            "rel_rmse": fits[hinge_name]["rel_rmse"],
                            "rmse_ms": fits[hinge_name]["rmse"],
                            "r2": fits[hinge_name]["r2"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        .sort_values("rel_rmse")
        .reset_index(drop=True)
    )

    dpus = agg["num_dpus"]
    print(
        f"\n=== {label} (n={len(agg)} configs, dpus {dpus.min():g}-{dpus.max():g}) ==="
    )
    print(table.to_string(index=False))
    return table, fits, hinge_name


def fit_regime_hybrid(
    agg: pd.DataFrame,
    y: np.ndarray,
    floor: float,
    split: float,
    low_template: str,
    high_template: str,
) -> dict:
    """Fit `low_template` on num_dpus <= split and `high_template` on
    num_dpus > split *separately* -- each regime only sees its own rows, so
    neither fit is diluted by the other regime the way a single whole-dataset
    fit is -- then stitch the two predictions together by regime and score
    the combined result against the whole dataset. This is what --dpu-split's
    two regime tables hinted at (different templates win in each regime):
    does picking the right template per regime actually beat every single
    template fit globally?
    """
    mask_low = (agg["num_dpus"] <= split).to_numpy()
    low_fit = fit_ols(TEMPLATES[low_template](agg[mask_low]), y[mask_low], floor=floor)
    high_fit = fit_ols(
        TEMPLATES[high_template](agg[~mask_low]), y[~mask_low], floor=floor
    )

    pred = np.empty_like(y)
    pred[mask_low] = low_fit["pred"]
    pred[~mask_low] = high_fit["pred"]

    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    rel_rmse = relative_rmse(resid, y)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "pred": pred,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "r2": r2,
        "floor": floor,
        "low_template": low_template,
        "high_template": high_template,
        "split": split,
        "low_fit": low_fit,
        "high_fit": high_fit,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", help="Path to results.csv from scatter_bench")
    parser.add_argument(
        "--out-dir", default="plots", help="Output directory for plots (default: plots)"
    )
    parser.add_argument(
        "--blocks-per-dpu",
        type=int,
        default=None,
        help="blocks_per_dpu to slice for the vs-block-size/vs-num-dpus plots "
        "(default: the largest value present)",
    )
    parser.add_argument(
        "--num-dpus",
        type=int,
        default=None,
        help="num_dpus to slice for the heatmap (default: the largest value present)",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="minimum measurable latency (ms), used to floor every regression's "
        "predictions -- a dpu_push_sg_xfer call can never be faster than "
        "this fixed per-call overhead (default: the minimum ms value in the data)",
    )
    parser.add_argument(
        "--hinge-threshold",
        type=float,
        default=DEFAULT_HINGE_THRESHOLD,
        help="block_size (bytes) knee for the hinge template: latency is treated "
        f"as flat below it, linear above it (default: {DEFAULT_HINGE_THRESHOLD:g})",
    )
    parser.add_argument(
        "--dpu-split",
        type=float,
        default=None,
        help="if given, also fit+report every template separately on num_dpus <= "
        "this threshold and num_dpus > this threshold (in addition to the "
        "whole-dataset fit), to compare which template wins in each regime. "
        "Also builds a 'hybrid' template (see --hybrid-low/high-template) "
        "that uses the low-regime fit below this threshold and the "
        "high-regime fit above it, competing alongside every other template "
        "(default: no split, whole-dataset fit only, no hybrid)",
    )
    parser.add_argument(
        "--hybrid-low-template",
        default="threeway",
        help="template (a TEMPLATES key) to fit on num_dpus <= --dpu-split for the "
        "hybrid model (default: %(default)r)",
    )
    parser.add_argument(
        "--hybrid-high-template",
        default="threeway",
        help="template (a TEMPLATES key) to fit on num_dpus > --dpu-split for the "
        "hybrid model (default: %(default)r)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"{args.csv}: no rows (empty or header-only)")
    df["ms"] = df["ns"] / 1e6

    agg = (
        df.groupby(["num_dpus", "blocks_per_dpu", "block_size"])["ms"]
        .mean()
        .reset_index()
    )
    agg["total_bytes"] = agg.num_dpus * agg.blocks_per_dpu * agg.block_size
    agg["bytes_per_dpu"] = agg.blocks_per_dpu * agg.block_size

    blocks_per_dpu = args.blocks_per_dpu or int(agg["blocks_per_dpu"].max())
    num_dpus = args.num_dpus or int(agg["num_dpus"].max())
    out_dir = pathlib.Path(args.out_dir)

    plot_latency_vs_block_size(
        agg, blocks_per_dpu, out_dir / "latency_vs_block_size.png"
    )
    plot_latency_vs_bytes_per_dpu(agg, out_dir / "latency_vs_bytes_per_dpu.png")
    plot_latency_3d(agg, out_dir / "latency_3d.html")
    plot_latency_vs_num_dpus(agg, blocks_per_dpu, out_dir / "latency_vs_num_dpus.png")
    plot_heatmap(agg, num_dpus, out_dir / "heatmap_blocks_vs_size.png")
    plot_latency_vs_x_faceted(
        agg,
        "total_bytes",
        "total bytes (num_dpus × blocks_per_dpu × block_size)",
        "Scatter latency vs. total bytes, one panel per DPU count",
        out_dir / "latency_vs_total_bytes.png",
    )
    plot_latency_vs_x_faceted(
        agg,
        "bytes_per_dpu",
        "bytes per DPU (blocks_per_dpu × block_size)",
        "Scatter latency vs. bytes per DPU, one panel per DPU count",
        out_dir / "latency_vs_bytes_per_dpu_faceted.png",
    )

    y = agg["ms"].to_numpy(dtype=float)
    floor = args.floor if args.floor is not None else float(y.min())
    print(
        f"Floor (min measured latency, used to clamp every template's predictions): {floor:.4g} ms"
    )

    table, fits, hinge_name = fit_and_report(
        agg, floor, args.hinge_threshold, "All DPU counts"
    )

    if args.dpu_split is not None:
        for _name in (args.hybrid_low_template, args.hybrid_high_template):
            if _name not in TEMPLATES:
                raise SystemExit(
                    f"--hybrid-low/high-template: {_name!r} is not a TEMPLATES key. "
                    f"Available: {list(TEMPLATES)}"
                )
        low = agg[agg["num_dpus"] <= args.dpu_split]
        high = agg[agg["num_dpus"] > args.dpu_split]
        if not low.empty:
            fit_and_report(
                low, floor, args.hinge_threshold, f"DPU count <= {args.dpu_split:g}"
            )
        if not high.empty:
            fit_and_report(
                high, floor, args.hinge_threshold, f"DPU count > {args.dpu_split:g}"
            )

        # Hybrid: args.hybrid_low_template fit only on num_dpus <= split, plus
        # args.hybrid_high_template fit only on num_dpus > split, stitched
        # together -- added to `fits`/`table` alongside every whole-dataset
        # template, so it competes for best_name and shows up in
        # regression_fit.png (and regression_fit_best.png if it wins).
        hybrid_name = (
            f"hybrid: {args.hybrid_low_template} (≤{args.dpu_split:g} dpus) / "
            f"{args.hybrid_high_template} (>{args.dpu_split:g} dpus)"
        )
        hybrid_fit = fit_regime_hybrid(
            agg,
            y,
            floor,
            args.dpu_split,
            args.hybrid_low_template,
            args.hybrid_high_template,
        )
        fits[hybrid_name] = hybrid_fit
        table = (
            pd.concat(
                [
                    table,
                    pd.DataFrame(
                        [
                            {
                                "template": hybrid_name,
                                "rel_rmse": hybrid_fit["rel_rmse"],
                                "rmse_ms": hybrid_fit["rmse"],
                                "r2": hybrid_fit["r2"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            .sort_values("rel_rmse")
            .reset_index(drop=True)
        )
        print(f"\n=== {hybrid_name} ===")
        print(
            f"  relRMSE = {hybrid_fit['rel_rmse'] * 100:.2f}%   "
            f"RMSE = {hybrid_fit['rmse']:.4g} ms   R² = {hybrid_fit['r2']:.4f}"
        )

    best_name = table.iloc[0]["template"]
    plot_all_regression_fits(
        agg, y, fits, table, best_name, out_dir / "regression_fit.png"
    )
    plot_best_regression_fit(
        agg, y, fits[best_name], best_name, out_dir / "regression_fit_best.png"
    )
    plot_predicted_latency_3d(
        agg,
        fits[best_name]["pred"],
        best_name,
        out_dir / "latency_3d_predicted_best.html",
    )
    plot_predicted_latency_3d(
        agg,
        fits[hinge_name]["pred"],
        hinge_name,
        out_dir / "latency_3d_predicted_hinge.html",
    )
    print(f"\nBest fit: {best_name}")
    best_fit = fits[best_name]
    print(
        f"  relRMSE = {best_fit['rel_rmse'] * 100:.2f}%   "
        f"RMSE = {best_fit['rmse']:.4g} ms   R² = {best_fit['r2']:.4f}"
    )
    if "low_template" in best_fit:
        print(
            f"  {best_fit['low_template']} (<= {best_fit['split']:g} dpus): "
            f"intercept = {best_fit['low_fit']['intercept']:.4g}, "
            f"coef = {[f'{c:.4g}' for c in best_fit['low_fit']['coef']]}"
        )
        print(
            f"  {best_fit['high_template']} (> {best_fit['split']:g} dpus): "
            f"intercept = {best_fit['high_fit']['intercept']:.4g}, "
            f"coef = {[f'{c:.4g}' for c in best_fit['high_fit']['coef']]}"
        )
    elif "dpu_intercepts" in best_fit:
        print(
            f"  shared slope (ms per byte above {best_fit['threshold']:g}B) = {best_fit['coef'][0]:.4g}"
        )
        print("  per-dpu-count intercept (excess above floor, ms):")
        for n_dpu, val in sorted(best_fit["dpu_intercepts"].items()):
            print(f"    dpus={n_dpu:<5d} {val:.4g}")
    else:
        print(f"  intercept (excess above floor) = {best_fit['intercept']:.4g}")
        print(f"  coef = {[f'{c:.4g}' for c in best_fit['coef']]}")
    print(f"\nPlots written to {out_dir}/")


if __name__ == "__main__":
    main()

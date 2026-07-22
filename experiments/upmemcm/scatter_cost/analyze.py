#!/usr/bin/env python3
"""
Analyze scatter_bench.cpp-style results.csv files (columns: <dims...>, iter,
ns) for a UPMEM scatter/gather/broadcast cost-model study. All latencies are
converted to milliseconds up front and reported/plotted in ms throughout.

This file is split into two parts:

  - A *generic engine* (Dim, ScatterProblem, Template, build_templates,
    fit_ols, fit_regime_hybrid, fit_and_report, and every plot_* function):
    none of it hardcodes dimension names or a fixed dimension count. It's
    driven entirely by a ScatterProblem describing
    which DataFrame columns are the sweep's dimensions. This scatter variant
    happens to have 3 (num_dpus, blocks_per_dpu, block_size); a
    one-transfer-per-DPU variant (num_dpus, num_bytes) or a broadcast/gather
    study would define its own ScatterProblem and reuse every function below
    unchanged. Every template is a Template(key, name, features): `key` is
    the short identifier used as every dict/table key (and for
    fit_regime_hybrid's low_template/high_template); `name` is the longer
    display label used in printed tables and plot titles. (Candidate for
    lifting into cinm_experiments once a second experiment is actually
    written against it.)
  - scatter_cost-*specific* configuration at the bottom (PROBLEM,
    EXTRA_TEMPLATES, CLI flags/defaults, main()) wiring the engine to this
    experiment's actual 3 dimensions and its own hand-found extra templates.

Every regression template is fit by weighted least squares (weight =
1/measured_ms^2) and ranked by relative_rmse (RMS of
(pred-measured)/measured), not flat ms RMSE -- latency here spans ~0.18ms to
tens of ms, so a 1ms error is huge at the low end and negligible at the high
end; flat RMSE (and an unweighted fit) would be dominated by the handful of
largest-latency configs and say nothing about how well small transfers are
predicted. ms RMSE and R^2 (on absolute error) are still reported alongside
for reference. Same idea as reduce_cost's fit_overhead_term.py TEMPLATES dict
(which also uses a cost-weighted fit for the same reason), simplified to
weighted OLS.

--dpu-split fits every template separately on group <= threshold and
group > threshold, to compare which template wins in each regime (e.g. if a
single template can't fit both very low and very high DPU counts well); it
also adds a "hybrid" template built from whichever templates actually won
each regime (no need to name them by hand), fit again on just their own
regime and stitched together, competing alongside every other whole-dataset
template for best_key (and so also appearing in
regression_fit.png/regression_fit_best.png if it wins).

Usage:
  python3 analyze.py results.csv
  python3 analyze.py results.csv --out-dir plots --blocks-per-dpu 24 --num-dpus 2048
  python3 analyze.py results.csv --dpu-split 32
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import operator
import pathlib
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Generic engine -- dimension-agnostic, no scatter_cost-specific column names
# ═══════════════════════════════════════════════════════════════════════════

# ── Problem definition ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Dim:
    """One independent variable of the sweep: a DataFrame column, its axis
    label, and whether it spans enough orders of magnitude to warrant a log2
    axis/tick scale (True for num_dpus/block_size/num_bytes-like columns;
    False for something like blocks_per_dpu, 1-24 and best shown linearly)."""

    col: str
    label: str
    log: bool = True


@dataclasses.dataclass(frozen=True)
class ScatterProblem:
    """Describes the dimensions of a scatter_bench-style sweep, so the fitting
    and plotting machinery below never needs to hardcode column names.

    `group` is the dimension used for color/facet/regime-split roles across
    every plot and fit -- num_dpus in every scatter variant so far, since
    it's always the dimension with by far the widest dynamic range and the
    natural thing to facet/color by. `shape` is the remaining dimension(s)
    describing the transfer itself (e.g. [blocks_per_dpu, block_size], or
    just [num_bytes] for a one-transfer-per-DPU variant).
    """

    group: Dim
    shape: list[Dim]

    @property
    def all_dims(self) -> list[Dim]:
        return [self.group] + self.shape


def _product(cols: list[pd.Series]) -> pd.Series:
    return functools.reduce(operator.mul, cols)


@dataclasses.dataclass(frozen=True)
class Template:
    """A regression feature-transform template. `key` is the short,
    CLI/dict-friendly identifier (e.g. "pairwise"); `name` is the longer
    human-readable label used in tables and plot titles (e.g. "pairwise
    (all dims)"); `features` is the DataFrame -> [n, k] design-matrix
    function, same contract fit_ols/fit_all_templates expect."""

    key: str
    name: str
    features: Callable[[pd.DataFrame], np.ndarray]


def add_derived_columns(agg: pd.DataFrame, problem: ScatterProblem) -> None:
    """Adds "total" (product of every dim) and "shape_product" (product of
    just the shape dims -- the dim itself when there's only one) in place."""
    agg["total"] = _product([agg[d.col] for d in problem.all_dims])
    agg["shape_product"] = _product([agg[d.col] for d in problem.shape])


# ── Regression templates ────────────────────────────────────────────────────


def _cols(*columns) -> np.ndarray:
    """Column-stack any number of 1D array-likes into a design matrix."""
    return np.column_stack(columns)


def _pairwise_cols(dims: list[Dim], df: pd.DataFrame) -> list:
    """[dim for each dim] + [dim_i * dim_j for i <= j] -- linear terms, plus
    every square and cross term exactly once (no duplicate cross terms)."""
    cols = [df[d.col] for d in dims]
    out = list(cols)
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            out.append(cols[i] * cols[j])
    out.append(_product(cols[i] for i in range(len(cols))))
    return out


def build_templates(
    problem: ScatterProblem, extra: dict[str, Template] | None = None
) -> dict[str, Template]:
    """Generates a family of Templates from `problem`'s dimensions, merged
    with any experiment-specific `extra` templates that don't generalize
    (e.g. a bespoke compound feature found by trial and error). Returns a
    dict keyed by each Template's short `key` (what every dict/CLI lookup
    uses), not its longer display `name`."""
    dims = problem.all_dims
    group, shape = problem.group, problem.shape

    generated = [
        Template("linear", "linear (all dims)", lambda df: _cols(*[df[d.col] for d in dims])),
        Template(
            "pairwise", "pairwise products", lambda df: _cols(*_pairwise_cols(dims, df))
        ),
        Template(
            "log2_all", "log2(all dims)", lambda df: _cols(*[np.log2(df[d.col]) for d in dims])
        ),
        # Matches the "each group value is the same curve, shifted by
        # a*log2(group)" shape: a single log2(group) offset, plus a full
        # pairwise (incl. cross terms) for the group-independent shape part.
        Template(
            "log2g_quad_shape",
            "log2(group) + pairwise(shape)",
            lambda df: _cols(np.log2(df[group.col]), *_pairwise_cols(shape, df)),
        ),
        Template(
            "total",
            "total (product of all dims)",
            lambda df: _cols(_product([df[d.col] for d in dims])),
        ),
        Template(
            "log2_total",
            "log2(total)",
            lambda df: _cols(np.log2(_product([df[d.col] for d in dims]))),
        ),
        Template(
            "log2g_shape_prod",
            "log2(group) + shape_product",
            lambda df: _cols(np.log2(df[group.col]), _product([df[d.col] for d in shape])),
        ),
    ]
    templates = {t.key: t for t in generated}
    if extra:
        templates.update(extra)
    return templates


def relative_rmse(resid: np.ndarray, y: np.ndarray) -> float:
    """RMS of (resid/y): unlike a flat ms RMSE, a 1ms error is scored the
    same whether the true latency is 0.2ms (500% off) or 30ms (3% off).
    This weights every config by how significant its error actually is,
    which flat RMSE does not -- a template can look great on ms-RMSE purely
    by nailing the handful of huge high-byte configs while being way off
    (relatively) on every small-transfer one, since ms-RMSE is dominated by
    the largest latencies in the dataset."""
    return float(np.sqrt(np.mean((resid / y) ** 2)))


def fit_ols(X: np.ndarray, y: np.ndarray, weighted: bool = True) -> dict:
    """OLS with intercept (the empirical minimum measured latency -- fixed
    per-call overhead like rank dispatch, WRAM setup, etc. -- is expected to
    show up in the fitted intercept on its own, not enforced separately).

    weighted=True (default) fits via weighted least squares with weight =
    1/y^2 -- i.e. it directly optimizes relative error (see relative_rmse),
    not just reports it after an absolute-error fit. Without this, the fit
    itself (not just the RMSE-based ranking) would be dominated by the
    largest latencies, the same problem relative_rmse is meant to fix.
    """
    Xi = np.column_stack([np.ones(len(y)), X])
    if weighted:
        sqrt_w = 1.0 / y
        coef, *_ = np.linalg.lstsq(Xi * sqrt_w[:, None], y * sqrt_w, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    pred = Xi @ coef
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
    }


def fit_all_templates(
    templates: dict[str, Template], data: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """Fits every Template in `templates`, keyed by its short `key` in the
    returned `fits` dict (each fit also carries its own "name" for display)."""
    rows = []
    fits = {}
    for key, tmpl in templates.items():
        X = tmpl.features(data)
        fit = fit_ols(X, y)
        fit["name"] = tmpl.name
        fits[key] = fit
        rows.append(
            {
                "key": key,
                "name": tmpl.name,
                "rel_rmse": fit["rel_rmse"],
                "rmse_ms": fit["rmse"],
                "r2": fit["r2"],
            }
        )
    table = pd.DataFrame(rows).sort_values("rel_rmse").reset_index(drop=True)
    return table, fits


def fit_and_report(
    templates: dict[str, Template],
    agg: pd.DataFrame,
    problem: ScatterProblem,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """Fit every template on `agg`, print a labeled summary table, and return
    (table, fits) so callers can plot/inspect the winner. Split out of
    main() so it can be called once on the whole dataset and again on
    group-value subsets (--dpu-split) to compare which template wins in each
    regime."""
    y = agg["ms"].to_numpy(dtype=float)
    table, fits = fit_all_templates(templates, agg, y)

    g = agg[problem.group.col]
    print(f"\n=== {label} (n={len(agg)} configs, {problem.group.col} {g.min():g}-{g.max():g}) ===")
    print(table.to_string(index=False))
    return table, fits


def fit_regime_hybrid(
    templates: dict[str, Template],
    agg: pd.DataFrame,
    y: np.ndarray,
    problem: ScatterProblem,
    split: float,
    low_template: str,
    high_template: str,
) -> dict:
    """Fit `low_template` on group <= split and `high_template` on
    group > split *separately* -- each regime only sees its own rows, so
    neither fit is diluted by the other regime the way a single whole-dataset
    fit is -- then stitch the two predictions together by regime and score
    the combined result against the whole dataset. This is what --dpu-split's
    two regime tables hint at (different templates win in each regime): does
    picking the right template per regime actually beat every single
    template fit globally? `low_template`/`high_template` are template keys
    (short identifiers), not display names.
    """
    mask_low = (agg[problem.group.col] <= split).to_numpy()
    low_fit = fit_ols(templates[low_template].features(agg[mask_low]), y[mask_low])
    high_fit = fit_ols(templates[high_template].features(agg[~mask_low]), y[~mask_low])

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
        "low_template": low_template,
        "high_template": high_template,
        "split": split,
        "low_fit": low_fit,
        "high_fit": high_fit,
    }


# ── Plots ────────────────────────────────────────────────────────────────────


def _pow2_ticks(values) -> list[int]:
    """Powers of 2 spanning values' range, for colorbar/axis ticks."""
    lo, hi = float(np.min(values)), float(np.max(values))
    k_min = int(np.floor(np.log2(lo)))
    k_max = int(np.ceil(np.log2(hi)))
    return [2**k for k in range(k_min, k_max + 1)]


def _norm_for(dim: Dim, values):
    """(matplotlib Normalize, tick list-or-None) for a dimension's colorbar/
    color-mapped axis, honoring its `log` flag."""
    if dim.log:
        return LogNorm(vmin=np.min(values), vmax=np.max(values)), _pow2_ticks(values)
    return Normalize(vmin=np.min(values), vmax=np.max(values)), None


def _add_colorbar(fig, mappable, ax, dim: Dim, values, **kwargs):
    ticks = _pow2_ticks(values) if dim.log else None
    cbar = fig.colorbar(mappable, ax=ax, ticks=ticks, label=dim.label, **kwargs)
    if dim.log:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    return cbar


def plot_latency_vs_dim(
    agg: pd.DataFrame,
    x: Dim,
    color: Dim,
    fixed: dict,
    out_path: pathlib.Path,
    cmap_name: str = "viridis",
) -> None:
    """Lines per `color` value, x = `x`, restricted to rows where every
    column in `fixed` equals its given value. Pass fixed={} when there's
    nothing left to hold fixed (e.g. only one shape dim total)."""
    sub = agg
    for col, val in fixed.items():
        sub = sub[sub[col] == val]
    if sub.empty:
        return
    norm, _ = _norm_for(color, sub[color.col])
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(sub[color.col].unique()):
        s = sub[sub[color.col] == c].sort_values(x.col)
        ax.plot(
            s[x.col], s["ms"], marker="o", markersize=3, linewidth=1.5, color=cmap(norm(c))
        )
    if x.log:
        ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(x.label)
    ax.set_ylabel("latency (ms)")
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
    ax.set_title(f"Scatter latency vs. {x.label}" + (f" ({fixed_str})" if fixed_str else ""))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    _add_colorbar(fig, sm, ax, color, sub[color.col])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_shape_product(
    agg: pd.DataFrame, problem: ScatterProblem, out_path: pathlib.Path
) -> None:
    """x = shape_product (every shape dim multiplied together -- the dim
    itself when there's only one), scatter points (not connected lines,
    since several shape-dim combinations can share the same product),
    colored by group. Vertical spread at a fixed x means the shape dims
    matter individually, not just through their product."""
    color = problem.group
    norm, _ = _norm_for(color, agg[color.col])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(agg[color.col].unique()):
        s = agg[agg[color.col] == c]
        ax.scatter(s["shape_product"], s["ms"], s=10, alpha=0.6, color=cmap(norm(c)))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    shape_label = " × ".join(d.label for d in problem.shape)
    ax.set_xlabel(shape_label)
    ax.set_ylabel("latency (ms)")
    suffix = " (every split)" if len(problem.shape) > 1 else ""
    ax.set_title(f"Scatter latency vs. {shape_label}{suffix}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    _add_colorbar(fig, sm, ax, color, agg[color.col])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(
    agg: pd.DataFrame, problem: ScatterProblem, group_value, out_path: pathlib.Path
) -> None:
    """Pivot of the first two shape dims -> latency, at one fixed group
    value. Needs (at least) 2 shape dims to make sense -- a no-op when
    there's only one (nothing to put on the second axis)."""
    if len(problem.shape) < 2:
        return
    x_dim, y_dim = problem.shape[0], problem.shape[1]
    sub = agg[agg[problem.group.col] == group_value]
    if sub.empty:
        return
    pivot = sub.pivot(index=y_dim.col, columns=x_dim.col, values="ms")
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
    if x_dim.log:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(x_dim.label)
    ax.set_ylabel(y_dim.label)
    ax.set_title(f"Scatter latency (ms), {problem.group.col}={group_value}")
    fig.colorbar(im, ax=ax, label="latency (ms)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_latency_3d_html(
    agg: pd.DataFrame,
    problem: ScatterProblem,
    z: np.ndarray,
    z_name: str,
    title: str,
    out_path: pathlib.Path,
) -> None:
    """Interactive 3D scatter, written as a standalone HTML file via Plotly.

    Matplotlib's mplot3d does NOT reliably support log-scaled 3D axes --
    set_xscale/set_zscale exist but don't correctly rescale the 3D
    projection. Plotly's WebGL 3D scene supports log axes directly and
    correctly, plus rotate/zoom/hover instead of a single fixed static angle.

    Axes: the first two shape dims with color=group when there are >= 2
    shape dims (this experiment's layout); otherwise x=group, y=the one
    shape dim, no color dimension (there's nothing left to spend it on) --
    either way, exactly 3 informative axes.
    """
    import plotly.graph_objects as go

    if len(problem.shape) >= 2:
        x_dim, y_dim, color_dim = problem.shape[0], problem.shape[1], problem.group
    else:
        x_dim, y_dim, color_dim = problem.group, problem.shape[0], None

    marker = dict(size=4, opacity=0.7)
    if color_dim is not None:
        color_vals = agg[color_dim.col]
        if color_dim.log:
            ticks = _pow2_ticks(color_vals)
            marker.update(
                color=np.log2(color_vals),
                colorscale="Viridis",
                colorbar=dict(
                    title=color_dim.label,
                    tickvals=np.log2(ticks),
                    ticktext=[f"{t:g}" for t in ticks],
                ),
            )
        else:
            marker.update(
                color=color_vals, colorscale="Viridis", colorbar=dict(title=color_dim.label)
            )
    else:
        marker.update(color=z, colorscale="Viridis", colorbar=dict(title=z_name))

    hover_dims = problem.all_dims
    customdata = np.column_stack([agg[d.col] for d in hover_dims] + [z])
    hover_lines = [f"{d.col}=%{{customdata[{i}]}}" for i, d in enumerate(hover_dims)]
    hover_lines.append(f"{z_name}=%{{customdata[{len(hover_dims)}]:.4g}} ms")
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=agg[x_dim.col],
                y=agg[y_dim.col],
                z=z,
                mode="markers",
                marker=marker,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title=x_dim.label, type="log" if x_dim.log else "linear"),
            yaxis=dict(title=y_dim.label, type="log" if y_dim.log else "linear"),
            zaxis=dict(title=f"{z_name} (ms)", type="log"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def plot_latency_3d(agg: pd.DataFrame, problem: ScatterProblem, out_path: pathlib.Path) -> None:
    """Interactive 3D scatter of *measured* latency."""
    title = "Measured latency: " + " × ".join(d.label for d in problem.all_dims)
    _plot_latency_3d_html(agg, problem, agg["ms"].to_numpy(), "measured latency", title, out_path)


def plot_predicted_latency_3d(
    agg: pd.DataFrame,
    problem: ScatterProblem,
    pred: np.ndarray,
    template_name: str,
    out_path: pathlib.Path,
) -> None:
    """Same 3D view as plot_latency_3d, but z = a regression template's
    *predicted* latency instead of the measured value. Comparing this
    against plot_latency_3d's measured point cloud shows what shape each
    model actually assumes -- e.g. a template with no term for some
    dimension predicts a surface that's completely flat along it, so if the
    measured cloud visibly fans out along that dimension instead, that
    mismatch is exactly why the template fits poorly."""
    title = f"Predicted latency ({template_name}): " + " × ".join(
        d.label for d in problem.all_dims
    )
    _plot_latency_3d_html(agg, problem, pred, "predicted latency", title, out_path)


def plot_latency_vs_x_faceted(
    agg: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    out_path: pathlib.Path,
    facet: Dim,
    color: Dim,
) -> None:
    """One small subplot per `facet` value (not colored by it, faceted on
    it), latency vs. x_col, points colored by `color`. Every subplot shares
    the same x/y limits, so this shows whether x_col alone collapses latency
    onto (roughly) the same curve regardless of `facet`, or whether the
    curves still shift from panel to panel."""
    facet_values = sorted(agg[facet.col].unique())
    n = len(facet_values)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols

    color_norm, _ = _norm_for(color, agg[color.col])
    cmap = plt.get_cmap("plasma")

    pad = 1.15
    x_lo, x_hi = agg[x_col].min() / pad, agg[x_col].max() * pad
    y_lo, y_hi = agg["ms"].min() / pad, agg["ms"].max() * pad

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), squeeze=False, constrained_layout=True
    )

    sc = None
    for i, v in enumerate(facet_values):
        ax = axes[divmod(i, ncols)[0]][divmod(i, ncols)[1]]
        sub = agg[agg[facet.col] == v]
        sc = ax.scatter(
            sub[x_col], sub["ms"], s=6, alpha=0.6, c=sub[color.col], cmap=cmap, norm=color_norm
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{facet.col}={v}", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[divmod(j, ncols)[0]][divmod(j, ncols)[1]].axis("off")

    fig.supxlabel(x_label)
    fig.supylabel("latency (ms)")
    fig.suptitle(title)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label=color.label)
        if color.log:
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_regression_fits(
    data: pd.DataFrame,
    y: np.ndarray,
    fits: dict,
    table: pd.DataFrame,
    best_key: str,
    problem: ScatterProblem,
    out_path: pathlib.Path,
) -> None:
    """One square measured-vs-predicted subplot per regression template, 2
    per row. The best template's title is red. Every subplot in a row shares
    the same color scale (group), so only one colorbar is drawn per row (at
    the row's right edge) instead of one per subplot."""
    keys = list(table["key"])
    ncols = 2
    nrows = (len(keys) + ncols - 1) // ncols

    color = problem.group
    norm, ticks = _norm_for(color, data[color.col])
    cmap = plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Shared axis limits across every subplot for direct comparability. Some
    # templates (plain linear OLS on raw dims, mostly) extrapolate to
    # negative "predicted latency" for some rows -- meaningless on a log
    # scale, so only positive predictions inform the shared range; those
    # points just won't render on any subplot, matching their invalidity.
    all_pred = np.concatenate([fits[k]["pred"] for k in keys])
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

    for i, key in enumerate(keys):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        fit = fits[key]
        ax.scatter(fit["pred"], y, s=8, alpha=0.5, c=data[color.col], cmap=cmap, norm=norm)
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("predicted latency (ms)")
        ax.set_ylabel("measured latency (ms)")
        is_best = key == best_key
        ax.set_title(
            f"{fit['name']}\nrelRMSE={fit['rel_rmse'] * 100:.1f}%, R²={fit['r2']:.3f}, "
            f"RMSE={fit['rmse']:.3g}ms",
            color="red" if is_best else "black",
            fontweight="bold" if is_best else "normal",
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=8)

    for r in range(nrows):
        cax = fig.add_subplot(gs[r, ncols])
        cbar = fig.colorbar(sm, cax=cax, ticks=ticks, label=color.label)
        if color.log:
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            cbar.ax.yaxis.set_minor_formatter(NullFormatter())

    fig.suptitle("Regression template comparison (best fit in red)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_best_regression_fit(
    data: pd.DataFrame,
    y: np.ndarray,
    fit: dict,
    problem: ScatterProblem,
    out_path: pathlib.Path,
) -> None:
    """Same square measured-vs-predicted style as plot_all_regression_fits,
    but just the single best template, as its own standalone plot. `fit`
    must carry a "name" key (every fit produced by fit_all_templates/
    fit_and_report does)."""
    pred = fit["pred"]
    positive = np.concatenate([pred[pred > 0], y[y > 0]])
    pad = 1.15
    lo = positive.min() / pad
    hi = positive.max() * pad

    color = problem.group
    norm, ticks = _norm_for(color, data[color.col])

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pred, y, s=8, alpha=0.5, c=data[color.col], cmap="viridis", norm=norm)
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("predicted latency (ms)")
    ax.set_ylabel("measured latency (ms)")
    ax.set_title(
        f"Best fit: {fit['name']}\n"
        f"relRMSE={fit['rel_rmse'] * 100:.1f}%, R²={fit['r2']:.3f}, RMSE={fit['rmse']:.3g}ms",
        color="red",
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    cbar = fig.colorbar(sc, ax=ax, ticks=ticks, label=color.label)
    if color.log:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# scatter_cost-specific configuration
# ═══════════════════════════════════════════════════════════════════════════

PROBLEM = ScatterProblem(
    group=Dim("num_dpus", "number of DPUs"),
    shape=[
        Dim("blocks_per_dpu", "blocks per DPU", log=False),
        Dim("block_size", "block size (bytes)"),
    ],
)

# Bespoke templates found by trial and error that don't fall out of
# build_templates' generic families -- kept as `extra` rather than forcing
# the generator to special-case them.
EXTRA_TEMPLATES: dict[str, Template] = {
    "compound": Template(
        "compound",
        "compound",
        lambda d: _cols(
            np.log2(d.num_dpus), d.blocks_per_dpu, d.num_dpus * d.blocks_per_dpu * d.block_size
        ),
    ),
}

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
        "--dpu-split",
        type=float,
        default=None,
        help="if given, also fit+report every template separately on num_dpus <= "
        "this threshold and num_dpus > this threshold (in addition to the "
        "whole-dataset fit), to compare which template wins in each regime. "
        "Also builds a 'hybrid' template out of whichever template won each "
        "regime, competing alongside every other template "
        "(default: no split, whole-dataset fit only, no hybrid)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"{args.csv}: no rows (empty or header-only)")
    df["ms"] = df["ns"] / 1e6

    agg = df.groupby([d.col for d in PROBLEM.all_dims])["ms"].mean().reset_index()
    add_derived_columns(agg, PROBLEM)
    templates = build_templates(PROBLEM, extra=EXTRA_TEMPLATES)

    blocks_dim, size_dim = PROBLEM.shape
    blocks_per_dpu = args.blocks_per_dpu or int(agg[blocks_dim.col].max())
    num_dpus = args.num_dpus or int(agg[PROBLEM.group.col].max())
    out_dir = pathlib.Path(args.out_dir)

    fixed_blocks = {blocks_dim.col: blocks_per_dpu}
    plot_latency_vs_dim(
        agg, size_dim, PROBLEM.group, fixed_blocks, out_dir / "latency_vs_block_size.png"
    )
    plot_latency_vs_dim(
        agg,
        PROBLEM.group,
        size_dim,
        fixed_blocks,
        out_dir / "latency_vs_num_dpus.png",
        cmap_name="plasma",
    )
    plot_latency_vs_shape_product(agg, PROBLEM, out_dir / "latency_vs_bytes_per_dpu.png")
    plot_latency_3d(agg, PROBLEM, out_dir / "latency_3d.html")
    plot_heatmap(agg, PROBLEM, num_dpus, out_dir / "heatmap_blocks_vs_size.png")
    total_label = " × ".join(d.label for d in PROBLEM.all_dims)
    shape_label = " × ".join(d.label for d in PROBLEM.shape)
    plot_latency_vs_x_faceted(
        agg,
        "total",
        total_label,
        "Scatter latency vs. total bytes, one panel per DPU count",
        out_dir / "latency_vs_total_bytes.png",
        facet=PROBLEM.group,
        color=size_dim,
    )
    plot_latency_vs_x_faceted(
        agg,
        "shape_product",
        shape_label,
        "Scatter latency vs. bytes per DPU, one panel per DPU count",
        out_dir / "latency_vs_bytes_per_dpu_faceted.png",
        facet=PROBLEM.group,
        color=size_dim,
    )

    y = agg["ms"].to_numpy(dtype=float)
    table, fits = fit_and_report(templates, agg, PROBLEM, "All DPU counts")

    if args.dpu_split is not None:
        low = agg[agg[PROBLEM.group.col] <= args.dpu_split]
        high = agg[agg[PROBLEM.group.col] > args.dpu_split]
        low_best_key = high_best_key = None
        if not low.empty:
            low_table, _ = fit_and_report(templates, low, PROBLEM, f"DPU count <= {args.dpu_split:g}")
            low_best_key = low_table.iloc[0]["key"]
        if not high.empty:
            high_table, _ = fit_and_report(templates, high, PROBLEM, f"DPU count > {args.dpu_split:g}")
            high_best_key = high_table.iloc[0]["key"]

        # Hybrid: whichever template won the low regime, fit only on
        # num_dpus <= split, plus whichever won the high regime, fit only on
        # num_dpus > split, stitched together -- added to `fits`/`table`
        # alongside every whole-dataset template, so it competes for
        # best_key and shows up in regression_fit.png (and
        # regression_fit_best.png if it wins). Only meaningful with both
        # regimes populated.
        if low_best_key is not None and high_best_key is not None:
            hybrid_key = "hybrid"
            hybrid_name = (
                f"hybrid: {templates[low_best_key].name} (≤{args.dpu_split:g} dpus) / "
                f"{templates[high_best_key].name} (>{args.dpu_split:g} dpus)"
            )
            hybrid_fit = fit_regime_hybrid(
                templates, agg, y, PROBLEM, args.dpu_split, low_best_key, high_best_key
            )
            hybrid_fit["name"] = hybrid_name
            fits[hybrid_key] = hybrid_fit
            table = (
                pd.concat(
                    [
                        table,
                        pd.DataFrame(
                            [
                                {
                                    "key": hybrid_key,
                                    "name": hybrid_name,
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

    best_key = table.iloc[0]["key"]
    plot_all_regression_fits(agg, y, fits, table, best_key, PROBLEM, out_dir / "regression_fit.png")
    plot_best_regression_fit(agg, y, fits[best_key], PROBLEM, out_dir / "regression_fit_best.png")
    plot_predicted_latency_3d(
        agg,
        PROBLEM,
        fits[best_key]["pred"],
        fits[best_key]["name"],
        out_dir / "latency_3d_predicted_best.html",
    )
    best_fit = fits[best_key]
    print(f"\nBest fit: {best_fit['name']} (key={best_key})")
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
    else:
        print(f"  intercept = {best_fit['intercept']:.4g}")
        print(f"  coef = {[f'{c:.4g}' for c in best_fit['coef']]}")
    print(f"\nPlots written to {out_dir}/")


if __name__ == "__main__":
    main()

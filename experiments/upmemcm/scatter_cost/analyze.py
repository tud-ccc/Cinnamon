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
    fit_regime_hybrid's regime_templates); `name` is the longer display
    label used in printed tables and plot titles. (Candidate for lifting
    into cinm_experiments once a second experiment is actually written
    against it.)
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

--split DIM BOUNDARY [BOUNDARY ...] fits every template separately on each
regime carved out of column DIM by those boundaries -- e.g. --split num_dpus
32 512 gives 3 regimes (<=32, 32-512, >512) -- to compare which template
wins in each (useful when a single template can't fit every regime well,
e.g. very low vs. very high DPU counts). Repeat --split for more dimensions
(e.g. --split block_size 1024 2048 --split num_dpus 64) to split on several
dimensions at once; a row's combined regime is then the cross product of
every dimension's own regime. It also adds a "hybrid" template built from
whichever templates actually won each combined regime (no need to name them
by hand), fit again on just their own regime and stitched together,
competing alongside every other whole-dataset template for best_key (and so
also appearing in regression_fit.png/regression_fit_best.png if it wins).

Usage:
  python3 analyze.py results.csv
  python3 analyze.py results.csv --out-dir plots --blocks-per-dpu 24 --num-dpus 2048
  python3 analyze.py results.csv --split num_dpus 32
  python3 analyze.py results.csv --split num_dpus 32 512
  python3 analyze.py results.csv --split block_size 1024 2048 --split num_dpus 64
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import itertools
import operator
import pathlib
import threading
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, NullFormatter
from tqdm import tqdm
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# Generic engine -- dimension-agnostic, no scatter_cost-specific column names
# ═══════════════════════════════════════════════════════════════════════════

# ── Problem definition ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Dim:
    """One independent variable (or measured value) of a sweep: a DataFrame
    column, its axis label, and display preferences.

    `log`: preferred log base for this dimension's axis/colorbar scale, or
    None for a linear scale. Byte/count-like quantities (num_dpus,
    block_size, num_bytes) are naturally binary and read better base-2;
    time-like quantities (latency) are naturally decimal and read better
    base-10; something with a small linear range (blocks_per_dpu, 1-24)
    wants no log scale at all (None).
    `decimal_labels`: whether tick/colorbar labels are written in plain
    decimal (e.g. "1024") rather than exponent form. Decimal reads better
    for log2 in practice; log10 is fine as decimal too, so this defaults to
    True regardless of `log`.
    `cmap`: preferred colormap name when this dimension is used as a color
    axis, or None to let the plotting function fall back to its own
    default (e.g. viridis).
    """

    col: str
    label: str
    log: int | None = 2
    decimal_labels: bool = True
    cmap: str | None = None


def _product(cols: list[pd.Series]) -> pd.Series:
    return functools.reduce(operator.mul, cols)


@dataclasses.dataclass
class ScatterProblem:
    """Describes the dimensions of a scatter_bench-style sweep, so the fitting
    and plotting machinery below never needs to hardcode column names.

    `group` is the dimension used for color/facet/regime-split roles across
    every plot and fit -- num_dpus in every scatter variant so far, since
    it's always the dimension with by far the widest dynamic range and the
    natural thing to facet/color by. `shape` is the remaining dimension(s)
    describing the transfer itself (e.g. [blocks_per_dpu, block_size], or
    just [num_bytes] for a one-transfer-per-DPU variant).

    `derived` holds Dim descriptors for columns computed by
    add_derived_columns ("total", "shape_product") -- empty until that's
    called. Not frozen (unlike Dim/Template) because add_derived_columns
    populates this dict after construction.
    """

    group: Dim
    shape: list[Dim]
    derived: dict[str, Dim] = dataclasses.field(default_factory=dict)

    @property
    def all_dims(self) -> list[Dim]:
        return [self.group] + self.shape

    def add_derived_columns(self, agg: pd.DataFrame, log: int | None = 2) -> None:
        """Adds "total" (product of every dim) and "shape_product" (product
        of just the shape dims -- the dim itself when there's only one) to
        `agg` in place, and registers Dim descriptors for them in
        `self.derived` so they can be used for plotting like any other
        dimension (e.g. plot_faceted_scatter/plot_vs_dim). Both are
        byte/count-like products, hence `log` (base-2 by default) rather
        than inferring a base from the constituent dims."""
        agg["total"] = _product([agg[d.col] for d in self.all_dims])
        agg["shape_product"] = _product([agg[d.col] for d in self.shape])
        self.derived = {
            "total": Dim("total", " × ".join(d.label for d in self.all_dims), log=log),
            "shape_product": Dim(
                "shape_product", " × ".join(d.label for d in self.shape), log=log
            ),
        }


@dataclasses.dataclass(frozen=True)
class Template:
    """A regression feature-transform template. `key` is the short,
    CLI/dict-friendly identifier (e.g. "pairwise"); `name` is the longer
    human-readable label used in tables and plot titles (e.g. "pairwise
    (all dims)"); `features` is the DataFrame -> [n, k] design-matrix
    function, same contract fit_ols/fit_all_templates expect.

    `fit` overrides the (X, y) -> fit-dict function used to fit this
    template's design matrix -- None (the default) means fit_ols, the
    weighted-least-squares fit every other template uses. Set it to swap in
    a different fitting procedure (e.g. fit_lasso) that shares fit_ols's
    return-dict contract (pred, intercept, coef, rmse, rel_rmse, r2) but
    fits differently -- without teaching fit_all_templates about every
    possible fitting procedure by name.

    `report`, if set, is called with this template's own fit-dict once
    after the whole-dataset fit, to print whatever extra info is specific
    to this template (e.g. LASSO's alpha/which terms got pruned) -- keeps
    that knowledge on the Template itself instead of main() reaching into
    `fits[some_hardcoded_key]` and knowing what that particular template's
    fit dict contains."""

    key: str
    name: str
    features: Callable[[pd.DataFrame], np.ndarray]
    fit: Callable[[np.ndarray, np.ndarray], dict] | None = None
    report: Callable[[dict], None] | None = None


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


def _pairwise_col_names(dims: list[Dim]) -> list[str]:
    """Names matching _interaction_cols' columns 1:1, for labeling
    coefficients (e.g. LASSO's) back by term instead of bare index."""
    names = [d.col for d in dims]
    for i in range(len(dims)):
        for j in range(i, len(dims)):
            if i == j:
                names.append(f"{dims[i].col}*{dims[i].col}")
            else:
                names.append(f"{dims[i].col}*{dims[j].col}")
    if len(dims) > 2:
        names.append("*".join(d.col for d in dims))
    return names


def _interaction_col_names(dims: list[Dim]) -> list[str]:
    """Names matching _interaction_cols' columns 1:1, for labeling
    coefficients (e.g. LASSO's) back by term instead of bare index."""
    names = [d.col for d in dims]
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            names.append(f"{dims[i].col}*{dims[j].col}")
    if len(dims) > 2:
        names.append("*".join(d.col for d in dims))
    return names


def _interaction_cols(dims: list[Dim], df: pd.DataFrame) -> list:
    """[dim for each dim] + [dim_i * dim_j for i < j] + [product of all
    dims] -- every main effect and every cross term, but unlike
    _pairwise_cols, no squared terms. This is the "a + bx + cy + dz + e*xy +
    f*xz + g*yz + h*xyz" interaction-model form: the full set of candidates
    a term-pruning procedure (manual backward elimination, or LASSO) picks
    a subset from."""
    cols = [df[d.col] for d in dims]
    out = list(cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            out.append(cols[i] * cols[j])
    if len(cols) > 2:
        out.append(_product(cols))
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

    generated = [
        Template(
            "linear", "linear (all dims)", lambda df: _cols(*[df[d.col] for d in dims])
        ),
        Template(
            "pairwise", "pairwise products", lambda df: _cols(*_pairwise_cols(dims, df))
        ),
       
        Template(
            "total",
            "total (product of all dims)",
            lambda df: _cols(_product([df[d.col] for d in dims])),
        ),
        # Template(
        #     "log2_total",
        #     "log2(total)",
        #     lambda df: _cols(np.log2(_product([df[d.col] for d in dims]))),
        # ),
        # Template(
        #     "log2g_shape_prod",
        #     "log2(group) + shape_product",
        #     lambda df: _cols(
        #         np.log2(df[group.col]), _product([df[d.col] for d in shape])
        #     ),
        # ),
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


def fit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    weighted: bool = True,
    cv: int = 5,
    random_state: int = 0,
) -> dict:
    from sklearn.linear_model import LassoCV
    """Standardize X (zero mean, unit variance per column), fit LassoCV --
    weighted by 1/y^2 the same way fit_ols's weighted OLS is, if
    weighted=True, so this template is scored on the same relative-error
    footing as every other one in the table -- then unwind the
    standardization so `intercept`/`coef` come back on the *original*
    feature scale (directly comparable to any other template's printed
    coef, and usable to predict on raw, unstandardized inputs).

    Coefficients LassoCV shrinks to exactly zero are the terms this
    technique prunes automatically, in place of manually dropping one term
    at a time and refitting.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std

    # sample_weight = 1/y^2, matching fit_ols's row-scaling-by-1/y trick:
    # scaling rows by sqrt(w) before an unweighted fit minimizes
    # sum(w * resid^2) with w = sqrt(w)^2 -- same weighting, expressed via
    # sklearn's native sample_weight instead, which (unlike row-scaling)
    # doesn't also drag the intercept into the penalty term.
    sample_weight = (1.0 / y) ** 2 if weighted else None

    model = LassoCV(cv=cv, fit_intercept=True, max_iter=100_000, random_state=random_state)
    model.fit(Xs, y, sample_weight=sample_weight)

    # Unwind standardization: y = intercept_s + Xs @ coef_s
    #                            = intercept_s + ((X - mean) / std) @ coef_s
    #                            = (intercept_s - (mean/std)@coef_s) + X @ (coef_s/std)
    coef = model.coef_ / std
    intercept = float(model.intercept_ - np.sum(mean / std * model.coef_))

    pred = intercept + X @ coef
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    rel_rmse = float(np.sqrt(np.mean((resid / y) ** 2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "pred": pred,
        "intercept": intercept,
        "coef": coef,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "r2": r2,
        "alpha": float(model.alpha_),
        "n_nonzero": int(np.sum(coef != 0)),
    }

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
        fit_fn = tmpl.fit or fit_ols
        fit = fit_fn(X, y)
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
    print(
        f"\n=== {label} (n={len(agg)} configs, {problem.group.col} {g.min():g}-{g.max():g}) ==="
    )
    print(table.to_string(index=False))
    return table, fits


def regime_masks(values: pd.Series, boundaries: list[float]) -> list[np.ndarray]:
    """Partitions `values` into len(boundaries)+1 boolean masks by a list of
    upper boundaries (need not be pre-sorted): (values <= boundaries[0]),
    (boundaries[0] < values <= boundaries[1]), ..., (values > boundaries[-1]).
    Generalizes a single low/high threshold to an arbitrary number of
    regimes along one dimension -- e.g. boundaries=[32, 512] on num_dpus
    gives 3 regimes (<=32, 32-512, >512) instead of just 2."""
    boundaries = sorted(boundaries)
    masks = []
    lo = -np.inf
    for b in boundaries:
        masks.append(((values > lo) & (values <= b)).to_numpy())
        lo = b
    masks.append((values > lo).to_numpy())
    return masks


def regime_labels(col: str, boundaries: list[float]) -> list[str]:
    """Display labels matching regime_masks' regimes 1:1, e.g.
    ["num_dpus <= 32", "32 < num_dpus <= 512", "num_dpus > 512"]."""
    boundaries = sorted(boundaries)
    labels = [f"{col} <= {boundaries[0]:g}"]
    for lo, hi in zip(boundaries, boundaries[1:]):
        labels.append(f"{lo:g} < {col} <= {hi:g}")
    labels.append(f"{col} > {boundaries[-1]:g}")
    return labels


def combined_regime_masks(
    agg: pd.DataFrame, splits: dict[str, list[float]]
) -> tuple[list[np.ndarray], list[str]]:
    """Cross product of one regime_masks/regime_labels split per (col,
    boundaries) pair in `splits` -- e.g. splits={"block_size": [1024, 2048],
    "num_dpus": [64]} carves the 3 block_size regimes x the 2 num_dpus
    regimes into 6 combined regimes, each the intersection of one regime
    from every split dimension. Order matches itertools.product's (first
    split's regimes vary slowest), consistently between the returned masks
    and labels. len(splits) == 1 degenerates to plain regime_masks/
    regime_labels on that one column."""
    per_dim = [
        list(zip(regime_masks(agg[col], boundaries), regime_labels(col, boundaries)))
        for col, boundaries in splits.items()
    ]
    masks, labels = [], []
    for combo in itertools.product(*per_dim):
        masks.append(functools.reduce(operator.and_, (m for m, _ in combo)))
        labels.append(" && ".join(label for _, label in combo))
    return masks, labels


def fit_regime_hybrid(
    templates: dict[str, Template],
    agg: pd.DataFrame,
    y: np.ndarray,
    splits: dict[str, list[float]],
    regime_templates: list[str],
) -> dict:
    """Fit each combined regime's own template (`regime_templates[i]`, a
    template key) on just that regime's own rows -- carved out of `agg` by
    `splits` via combined_regime_masks -- so no fit is diluted by rows from a
    different regime the way a single whole-dataset fit is -- then stitch
    the per-regime predictions back together and score the combined result
    against the whole dataset. This is what --split's per-regime tables hint
    at (different templates can win in different regimes): does picking the
    right template per regime actually beat every single whole-dataset
    template fit? `regime_templates` must have one entry per combined regime,
    in the same order combined_regime_masks produces.
    """
    masks, labels = combined_regime_masks(agg, splits)
    assert len(regime_templates) == len(masks), (
        "need one template per combined regime (product of each split's own "
        "len(boundaries)+1)"
    )
    regime_fits = [
        fit_ols(templates[key].features(agg[mask]), y[mask])
        for mask, key in zip(masks, regime_templates)
    ]

    pred = np.empty_like(y)
    for mask, fit in zip(masks, regime_fits):
        pred[mask] = fit["pred"]

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
        "splits": splits,
        "regime_labels": labels,
        "regime_templates": regime_templates,
        "regime_fits": regime_fits,
    }


# ── Plots ────────────────────────────────────────────────────────────────────


def _log_ticks(values, base: int) -> list[float]:
    """Powers of `base` spanning values' range, for colorbar/axis ticks."""
    lo, hi = float(np.min(values)), float(np.max(values))
    k_min = int(np.floor(np.log(lo) / np.log(base)))
    k_max = int(np.ceil(np.log(hi) / np.log(base)))
    return [base**k for k in range(k_min, k_max + 1)]


def _cmap_for(dim: Dim, default: str = "viridis"):
    """`dim`'s preferred colormap, falling back to `default` if unset."""
    return plt.get_cmap(dim.cmap or default)


def _norm_for(dim: Dim, values):
    """(matplotlib Normalize, tick list-or-None) for a dimension's colorbar/
    color-mapped axis, honoring its `log` base (None = linear)."""
    if dim.log is not None:
        return LogNorm(vmin=np.min(values), vmax=np.max(values)), _log_ticks(
            values, dim.log
        )
    return Normalize(vmin=np.min(values), vmax=np.max(values)), None


def _apply_log_scale(ax, dim: Dim, axis: str) -> None:
    """Configure one matplotlib axis ('x' or 'y') for `dim`: log scale at its
    preferred base (a no-op if dim.log is None), decimal tick labels if
    dim.decimal_labels."""
    if dim.log is None:
        return
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=dim.log)
    if dim.decimal_labels:
        axis_obj = ax.xaxis if axis == "x" else ax.yaxis
        axis_obj.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _add_colorbar(fig, mappable, ax, dim: Dim, values, **kwargs):
    ticks = _log_ticks(values, dim.log) if dim.log is not None else None
    cbar = fig.colorbar(mappable, ax=ax, ticks=ticks, label=dim.label, **kwargs)
    if dim.log is not None and dim.decimal_labels:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    return cbar


def plot_vs_dim(
    agg: pd.DataFrame,
    x: Dim,
    color: Dim,
    value: Dim,
    out_path: pathlib.Path,
    fixed: dict | None = None,
    connect: bool = True,
) -> None:
    """`value` vs `x`, one line (connect=True) or scatter (connect=False)
    per `color` value, optionally restricted to rows where every column in
    `fixed` equals its given value. Use connect=False whenever several
    combinations of other dims can share the same `x` value (e.g. a derived
    product dim) -- connecting those with a line would draw a misleading
    jagged zigzag instead of a cloud. `color`'s own `cmap` (falling back to
    viridis) picks the marker/line colors."""
    sub = agg
    for col, val in (fixed or {}).items():
        sub = sub[sub[col] == val]
    if sub.empty:
        return
    norm, _ = _norm_for(color, sub[color.col])
    cmap = _cmap_for(color)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(sub[color.col].unique()):
        s = sub[sub[color.col] == c]
        if connect:
            s = s.sort_values(x.col)
            ax.plot(
                s[x.col],
                s[value.col],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=cmap(norm(c)),
            )
        else:
            ax.scatter(s[x.col], s[value.col], s=10, alpha=0.6, color=cmap(norm(c)))
    _apply_log_scale(ax, x, "x")
    _apply_log_scale(ax, value, "y")
    ax.set_xlabel(x.label)
    ax.set_ylabel(value.label)
    fixed_str = ", ".join(f"{k}={v}" for k, v in (fixed or {}).items())
    ax.set_title(
        f"{value.label} vs. {x.label}" + (f" ({fixed_str})" if fixed_str else "")
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    _add_colorbar(fig, sm, ax, color, sub[color.col])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(
    agg: pd.DataFrame,
    x: Dim,
    y: Dim,
    value: Dim,
    group: Dim,
    group_value,
    out_path: pathlib.Path,
) -> None:
    """Pivot of `x` x `y` -> `value`, at one fixed `group` value. Fully
    Dim-parametric -- no ScatterProblem here, so this (like the rest of the
    plot_* functions) can move into a shared plotting module."""
    sub = agg[agg[group.col] == group_value]
    if sub.empty:
        return
    pivot = sub.pivot(index=y.col, columns=x.col, values=value.col)
    pivot = pivot.sort_index().sort_index(axis=1)

    finite = pivot.values[~np.isnan(pivot.values)]
    norm, _ = _norm_for(value, finite)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.values,
        norm=norm,
        cmap=_cmap_for(value),
        shading="nearest",
    )
    _apply_log_scale(ax, x, "x")
    _apply_log_scale(ax, y, "y")
    ax.set_xlabel(x.label)
    ax.set_ylabel(y.label)
    ax.set_title(f"{value.label}, {group.col}={group_value}")
    _add_colorbar(fig, im, ax, value, finite)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_3d_scatter_html(
    agg: pd.DataFrame,
    x: Dim,
    y: Dim,
    z: Dim,
    z_values: np.ndarray,
    z_label: str,
    title: str,
    out_path: pathlib.Path,
    color: Dim | None = None,
) -> None:
    """Interactive 3D scatter (x, y, z=`z` described by `value`, optionally
    colored by `color`), written as a standalone HTML file via Plotly. Fully
    Dim-parametric -- no ScatterProblem here.

    Matplotlib's mplot3d does NOT reliably support log-scaled 3D axes --
    set_xscale/set_zscale exist but don't correctly rescale the 3D
    projection. Plotly's WebGL 3D scene supports log axes directly and
    correctly, plus rotate/zoom/hover instead of a single fixed static angle.

    `z` is passed as a raw array rather than read off `agg` because it may be
    a regression template's prediction, not a real column; `value` supplies
    its axis scale/label preferences (log base, decimal labels) while
    `z_label` supplies the specific wording for *this* call (e.g. "measured
    latency" vs. "predicted latency (some template)").
    """
    import plotly.graph_objects as go

    marker = dict(size=4, opacity=0.7)
    if color is not None:
        color_vals = agg[color.col]
        if color.log is not None:
            ticks = _log_ticks(color_vals, color.log)
            log_vals = np.log(color_vals) / np.log(color.log)
            marker.update(
                color=log_vals,
                colorscale=color.cmap or "Viridis",
                colorbar=dict(
                    title=color.label,
                    tickvals=(np.log(ticks) / np.log(color.log))
                    if color.decimal_labels
                    else None,
                    ticktext=[f"{t:g}" for t in ticks]
                    if color.decimal_labels
                    else None,
                ),
            )
        else:
            marker.update(
                color=color_vals,
                colorscale=color.cmap or "Viridis",
                colorbar=dict(title=color.label),
            )
    else:
        marker.update(
            color=z_values, colorscale=z.cmap or "Viridis", colorbar=dict(title=z_label)
        )

    hover_dims = [d for d in (x, y, color) if d is not None]
    customdata = np.column_stack([agg[d.col] for d in hover_dims] + [z_values])
    hover_lines = [f"{d.col}=%{{customdata[{i}]}}" for i, d in enumerate(hover_dims)]
    hover_lines.append(f"{z_label}=%{{customdata[{len(hover_dims)}]:.4g}}")
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=agg[x.col],
                y=agg[y.col],
                z=z_values,
                mode="markers",
                marker=marker,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        ]
    )
    # Plotly's 3D scene axis "type" only distinguishes log vs. linear, not
    # log base -- there's no "log2"/"log10" variant, just "log".
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title=x.label, type="log" if x.log is not None else "linear"),
            yaxis=dict(title=y.label, type="log" if y.log is not None else "linear"),
            zaxis=dict(
                title=f"{z_label} ({z.label})",
                type="log" if z.log is not None else "linear",
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def plot_3d_measured(
    agg: pd.DataFrame, x: Dim, y: Dim, z: Dim, color: Dim, out_path: pathlib.Path
) -> None:
    """Interactive 3D scatter of *measured* `value`."""
    title = f"Measured {z.label}: "
    _plot_3d_scatter_html(
        agg,
        x=x,
        y=y,
        z=z,
        z_values=agg[z.col].to_numpy(),
        z_label=f"measured {z.label}",
        title=title,
        out_path=out_path,
        color=color,
    )


def plot_3d_predicted(
    agg: pd.DataFrame,
    x: Dim,
    y: Dim,
    z: Dim,
    z_values: np.ndarray,
    color: Dim,
    template_name: str,
    out_path: pathlib.Path,
) -> None:
    """Same 3D view as plot_3d_measured, but z = a regression template's
    *predicted* value instead of the measured one. Comparing this against
    plot_3d_measured's point cloud shows what shape each model actually
    assumes -- e.g. a template with no term for some dimension predicts a
    surface that's completely flat along it, so if the measured cloud
    visibly fans out along that dimension instead, that mismatch is exactly
    why the template fits poorly."""
    title = f"Predicted {y.label} ({template_name})"

    _plot_3d_scatter_html(
        agg,
        x=x,
        y=y,
        z=z,
        z_values=z_values,
        z_label=f"predicted {z.label}",
        title=title,
        out_path=out_path,
        color=color,
    )


def plot_3d_fit_wireframe(
    agg: pd.DataFrame,
    x: Dim,
    y: Dim,
    z: Dim,
    pred: np.ndarray,
    group: Dim,
    template_name: str,
    out_path: pathlib.Path,
    splits: dict[str, list[float]] | None = None,
) -> None:
    """Interactive 3D view combining plot_3d_measured's point cloud with a
    wireframe of `pred` (some template's prediction, e.g. the per-regime
    LASSO "hybrid" fit) instead of another scatter cloud -- for each `group`
    value, pivot that group's rows into an (x, y) grid of predicted z, then
    draw one line through every fixed-x row (varying y) and one line
    through every fixed-y column (varying x), so a smooth model surface
    folds into a visible grid instead of blending into the measured cloud.
    Measured points are colored per `group` value (so each group's own
    cloud is identifiable); every wireframe is drawn in a fixed bright red,
    unrelated to `group`'s color, so the grid reads as "the model" against
    any/every group's points rather than needing its own color key. There's
    no legend -- with one entry per group per measured/wireframe trace it'd
    be a wall of text -- toggling measured points/wireframe wholesale is a
    button (see updatemenus below) instead of a legend click.

    `splits` (typically a stitched fit's own "splits", e.g.
    fit_regime_hybrid's) carve each group's rows into
    combined_regime_masks' combined regimes first, and the grid is
    pivoted/drawn separately per combined regime -- since a stitched
    per-regime pred is fit independently per regime, it's generally
    discontinuous at a regime boundary, and a single wireframe spanning both
    sides would draw a misleading line across that jump. None (the default)
    skips this and treats each group's rows as a single regime, for a
    `pred` that isn't regime-based.

    Unlike plot_3d_predicted, this needs `pred` values aligned 1:1 with
    `agg`'s rows -- any stitched-together per-regime pred array (e.g.
    fit_regime_hybrid's) works, it doesn't have to come from a single
    whole-dataset template."""
    import plotly.graph_objects as go

    agg = agg.assign(__pred=np.asarray(pred, dtype=float))
    group_values = sorted(agg[group.col].unique())
    norm, _ = _norm_for(group, agg[group.col])
    cmap = _cmap_for(group)

    if splits:
        regime_masks_list, _ = combined_regime_masks(agg, splits)
    else:
        regime_masks_list = [np.ones(len(agg), dtype=bool)]

    traces = []
    scatter_indices = []
    wireframe_indices = []
    for g in group_values:
        group_mask = (agg[group.col] == g).to_numpy()
        r, gr, b, _ = cmap(norm(g))
        color = f"rgb({r * 255:.0f},{gr * 255:.0f},{b * 255:.0f})"
        label = f"{group.col}={g:g}"

        sub_all = agg[group_mask]
        scatter_indices.append(len(traces))
        traces.append(
            go.Scatter3d(
                x=sub_all[x.col],
                y=sub_all[y.col],
                z=sub_all[z.col],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.5),
                name=f"{label} measured",
                showlegend=False,
            )
        )

        for rmask in regime_masks_list:
            sub = agg[group_mask & rmask]
            if sub.empty:
                continue

            pivot = sub.pivot_table(index=x.col, columns=y.col, values="__pred")
            pivot = pivot.sort_index().sort_index(axis=1)
            xs, ys = pivot.index.to_numpy(), pivot.columns.to_numpy()

            for xv in xs:
                wireframe_indices.append(len(traces))
                traces.append(
                    go.Scatter3d(
                        x=np.full(len(ys), xv),
                        y=ys,
                        z=pivot.loc[xv].to_numpy(),
                        mode="lines",
                        line=dict(color="red", width=3),
                        legendgroup=label,
                        showlegend=False,
                    )
                )
            for yv in ys:
                wireframe_indices.append(len(traces))
                traces.append(
                    go.Scatter3d(
                        x=xs,
                        y=np.full(len(xs), yv),
                        z=pivot[yv].to_numpy(),
                        mode="lines",
                        line=dict(color="red", width=3),
                        legendgroup=label,
                        showlegend=False,
                    )
                )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Measured {z.label} (points) vs. {template_name} prediction (wireframe)",
        scene=dict(
            xaxis=dict(title=x.label, type="log" if x.log is not None else "linear"),
            yaxis=dict(title=y.label, type="log" if y.log is not None else "linear"),
            zaxis=dict(title=z.label, type="log" if z.log is not None else "linear"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        # No legend -- every trace is showlegend=False (one entry per group
        # per measured/wireframe would be a wall of text for little value
        # now that the buttons below toggle each kind wholesale instead).
        showlegend=False,
        # One-click show/hide -- `args`/`args2` on a single button is
        # Plotly's native toggle idiom: click applies `args`, click again
        # applies `args2`, restyle's second positional element restricts the
        # {"visible": ...} patch to just these trace indices instead of
        # every trace in the figure.
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="Toggle measured points",
                        method="restyle",
                        args=[{"visible": False}, scatter_indices],
                        args2=[{"visible": True}, scatter_indices],
                    ),
                    dict(
                        label="Toggle wireframe",
                        method="restyle",
                        args=[{"visible": False}, wireframe_indices],
                        args2=[{"visible": True}, wireframe_indices],
                    ),
                ],
            )
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def plot_faceted_scatter(
    agg: pd.DataFrame,
    x: Dim,
    y: Dim,
    title: str,
    out_path: pathlib.Path,
    facet: Dim,
    color: Dim,
) -> None:
    """One small subplot per `facet` value (not colored by it, faceted on
    it), `value` vs. `x`, points colored by `color`. Every subplot shares
    the same x/y limits, so this shows whether `x` alone collapses `value`
    onto (roughly) the same curve regardless of `facet`, or whether the
    curves still shift from panel to panel."""
    facet_values = sorted(agg[facet.col].unique())
    n = len(facet_values)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols

    color_norm, _ = _norm_for(color, agg[color.col])
    cmap = _cmap_for(color, default="plasma")

    pad = 1.15
    x_lo, x_hi = agg[x.col].min() / pad, agg[x.col].max() * pad
    y_lo, y_hi = agg[y.col].min() / pad, agg[y.col].max() * pad

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    sc = None
    for i, v in enumerate(facet_values):
        ax = axes[divmod(i, ncols)[0]][divmod(i, ncols)[1]]
        sub = agg[agg[facet.col] == v]
        sc = ax.scatter(
            sub[x.col],
            sub[y.col],
            s=6,
            alpha=0.6,
            c=sub[color.col],
            cmap=cmap,
            norm=color_norm,
        )
        _apply_log_scale(ax, x, "x")
        _apply_log_scale(ax, y, "y")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{facet.col}={v}", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[divmod(j, ncols)[0]][divmod(j, ncols)[1]].axis("off")

    fig.supxlabel(x.label)
    fig.supylabel(y.label)
    fig.suptitle(title)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label=color.label)
        if color.log is not None and color.decimal_labels:
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_faceted_fit(
    agg: pd.DataFrame,
    x: Dim,
    value: Dim,
    pred: np.ndarray,
    facet: Dim,
    title: str,
    out_path: pathlib.Path,
) -> None:
    """One small subplot per `facet` value: measured `value` vs. `x` overlaid
    with the model's `pred` vs. `x` for the same rows, so over/undershoot
    within a facet is visible directly (not just in the aggregate rel_rmse).
    Each panel's title carries that facet's own relative RMSE, so the
    regimes where the model does worst stand out at a glance."""
    facet_values = sorted(agg[facet.col].unique())
    n = len(facet_values)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols

    y = agg[value.col].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)

    pad = 1.15
    x_lo, x_hi = agg[x.col].min() / pad, agg[x.col].max() * pad
    positive = np.concatenate([y[y > 0], pred[pred > 0]])
    y_lo, y_hi = positive.min() / pad, positive.max() * pad

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for i, v in enumerate(facet_values):
        ax = axes[divmod(i, ncols)[0]][divmod(i, ncols)[1]]
        mask = (agg[facet.col] == v).to_numpy()
        xv, yv, pv = agg.loc[mask, x.col], y[mask], pred[mask]
        rel = relative_rmse(yv - pv, yv)

        ax.scatter(xv, yv, s=10, alpha=0.6, color="tab:blue", label="measured")
        ax.scatter(xv, pv, s=10, alpha=0.6, color="tab:orange", marker="x", label="predicted")
        _apply_log_scale(ax, x, "x")
        _apply_log_scale(ax, value, "y")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{facet.col}={v}\nrelRMSE={rel * 100:.1f}%", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, nrows * ncols):
        axes[divmod(j, ncols)[0]][divmod(j, ncols)[1]].axis("off")

    fig.supxlabel(x.label)
    fig.supylabel(value.label)
    fig.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_regression_fits(
    data: pd.DataFrame,
    y: np.ndarray,
    fits: dict,
    table: pd.DataFrame,
    best_key: str,
    color: Dim,
    value: Dim,
    out_path: pathlib.Path,
) -> None:
    """One square measured-vs-predicted subplot per regression template, 2
    per row. The best template's title is red. Every subplot in a row shares
    the same color scale (`color`), so only one colorbar is drawn per row
    (at the row's right edge) instead of one per subplot."""
    keys = list(table["key"])
    ncols = 2
    nrows = (len(keys) + ncols - 1) // ncols

    norm, ticks = _norm_for(color, data[color.col])
    cmap = _cmap_for(color)
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
        ax.scatter(
            fit["pred"], y, s=8, alpha=0.5, c=data[color.col], cmap=cmap, norm=norm
        )
        ax.plot(
            [lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x"
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _apply_log_scale(ax, value, "x")
        _apply_log_scale(ax, value, "y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"predicted {value.label}")
        ax.set_ylabel(f"measured {value.label}")
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
        if color.log is not None and color.decimal_labels:
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
    color: Dim,
    value: Dim,
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

    norm, ticks = _norm_for(color, data[color.col])

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        pred, y, s=8, alpha=0.5, c=data[color.col], cmap=_cmap_for(color), norm=norm
    )
    ax.plot(
        [lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _apply_log_scale(ax, value, "x")
    _apply_log_scale(ax, value, "y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"predicted {value.label}")
    ax.set_ylabel(f"measured {value.label}")
    nl = '\n'
    ax.set_title(
        f"Best fit:\n {fit['name'].replace('/', '/' + nl)}\n"
        f"relRMSE={fit['rel_rmse'] * 100:.1f}%, R²={fit['r2']:.3f}, RMSE={fit['rmse']:.3g}ms",
        color="red",
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    cbar = fig.colorbar(sc, ax=ax, ticks=ticks, label=color.label)
    if color.log is not None and color.decimal_labels:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class PlotPool:
    """Dispatches plot_*-style calls (e.g. `p.plot_vs_dim(...)`) to
    background threads instead of running them inline, so a caller's dozen
    independent plot calls overlap instead of running strictly one after
    another. `p.<name>(...)` looks up `<name>` as a module-level function
    (any plot_* function below) and submits a call to it with the same
    args/kwargs -- so call sites need only add a `p.` prefix, no other
    change. Call `.join()` once every plot has been dispatched, to wait for
    them all and re-raise the first exception any of them hit (a
    worker-thread exception would otherwise just vanish).

    matplotlib's pyplot interface (plt.subplots/plt.figure/plt.close, used
    by every PNG plot here) is not thread-safe -- it tracks the "current
    figure" in a shared global registry -- so every plot not listed in
    `_PARALLEL_SAFE` is serialized behind a lock; only the Plotly-based 3D
    HTML plots (which never touch that registry) actually run concurrently
    with each other and with whichever matplotlib call currently holds the
    lock."""

    _PARALLEL_SAFE = {"plot_3d_measured", "plot_3d_predicted", "plot_3d_fit_wireframe"}

    def __init__(self, max_workers: int = 16):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[concurrent.futures.Future] = []
        self._mpl_lock = threading.Lock()

    def __getattr__(self, name: str):
        fn = globals().get(name)
        if not callable(fn):
            raise AttributeError(f"PlotPool: no such plot function {name!r}")
        parallel_safe = name in self._PARALLEL_SAFE

        def dispatch(*args, **kwargs) -> None:
            def run() -> None:
                if parallel_safe:
                    fn(*args, **kwargs)
                else:
                    with self._mpl_lock:
                        fn(*args, **kwargs)

            self._futures.append(self._executor.submit(run))

        return dispatch

    def join(self) -> None:
        """Blocks until every dispatched plot has finished (a tqdm bar
        advances as each one completes), then re-raises the first exception
        any of them hit -- only after every plot has had a chance to run,
        so one failure doesn't hide whether others also failed."""
        for _ in tqdm(
            concurrent.futures.as_completed(self._futures),
            total=len(self._futures),
            desc="Plots",
        ):
            pass
        for future in self._futures:
            future.result()


# ═══════════════════════════════════════════════════════════════════════════
# scatter_cost-specific configuration
# ═══════════════════════════════════════════════════════════════════════════

PROBLEM = ScatterProblem(
    group=Dim("num_dpus", "number of DPUs", log=2),
    shape=[
        Dim("blocks_per_dpu", "blocks per DPU", log=None),
        Dim("block_size", "block size (bytes)", cmap="plasma", log=2),
    ],
)

# The measured/predicted value every plot's "value" axis describes -- time-like,
# so base-10 (unlike the byte/count-like sweep dimensions above, which are base-2).
LATENCY = Dim(
    col="ms", label="latency (ms)", log=10, decimal_labels=False, cmap="viridis"
)


def _lasso_term_report(dims: list[Dim]) -> Callable[[dict], None]:
    """Builds a Template.report callback for a LASSO fit whose features were
    built by _pairwise_cols(dims, ...) -- prints which of those terms
    survived vs. were pruned to exactly zero, labeled by name (via
    _pairwise_col_names, matching _pairwise_cols' column order 1:1) instead
    of bare coefficient index."""
    names = _pairwise_col_names(dims)

    def report(fit: dict) -> None:
        print(
            f"\n=== LASSO term selection (alpha={fit['alpha']:.4g}, "
            f"{fit['n_nonzero']}/{len(names)} terms kept) ==="
        )
        for name, c in zip(names, fit["coef"]):
            print(f"  {name:24s} {c:14.6g}" + ("" if c != 0 else "  (pruned)"))

    return report


# Bespoke templates found by trial and error that don't fall out of
# build_templates' generic families -- kept as `extra` rather than forcing
# the generator to special-case them.
EXTRA_TEMPLATES: dict[str, Template] = {
    "simplest": Template(
        key="simplest",
        name="dpu, size",
        features=lambda d: _cols(
            d.num_dpus,
            d.block_size,
            # d.num_dpus * d.blocks_per_dpu * d.block_size,
        ),
    ),
    # "compound": Template(
    #     "compound",
    #     "compound",
    #     lambda d: _cols(
    #         np.log2(d.num_dpus),
    #         d.blocks_per_dpu,
    #         d.num_dpus * d.blocks_per_dpu * d.block_size,
    #     ),
    # ),
    # Standardized LASSO over the same terms as "pairwise" -- the automatic
    # term-pruning analogue of it, fit via fit_lasso instead of fit_ols.
    # Which terms LASSO zeros out is printed via `report` (see
    # _lasso_term_report below) instead of main() hardcoding this template's
    # own coefficient layout.
    "lasso": Template(
        "lasso",
        "LASSO (pairwise)",
        lambda d: _cols(*_pairwise_cols(PROBLEM.all_dims, d)),
        fit=fit_lasso,
        report=_lasso_term_report(PROBLEM.all_dims),
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
        "--split",
        action="append",
        nargs="+",
        default=None,
        metavar=("DIM", "BOUNDARY"),
        help="DIM BOUNDARY [BOUNDARY ...]: split column DIM into regimes at those "
        "boundaries (e.g. --split num_dpus 32 512 gives 3 regimes: <=32, 32-512, "
        ">512) and fit+report every template separately per regime (in addition "
        "to the whole-dataset fit), to compare which template wins in each. "
        "Repeatable, once per dimension, to split on several dimensions at once "
        "(e.g. --split block_size 1024 2048 --split num_dpus 64) -- a row's "
        "combined regime is then the cross product of every dimension's own "
        "regime. Also builds a 'hybrid' template out of whichever template won "
        "each combined regime, competing alongside every other template "
        "(default: no split, whole-dataset fit only, no hybrid)",
    )
    args = parser.parse_args()
    splits: dict[str, list[float]] = {}
    for dim, *boundaries in args.split or []:
        splits[dim] = sorted(float(b) for b in boundaries)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"{args.csv}: no rows (empty or header-only)")
    df["ms"] = df["ns"] / 1e6

    agg = df.groupby([d.col for d in PROBLEM.all_dims])["ms"].median().reset_index()
    PROBLEM.add_derived_columns(agg)
    templates = build_templates(PROBLEM, extra=EXTRA_TEMPLATES)

    blocks_dim, size_dim = PROBLEM.shape
    blocks_per_dpu = args.blocks_per_dpu or int(agg[blocks_dim.col].max())
    num_dpus = args.num_dpus or int(agg[PROBLEM.group.col].max())
    out_dir = pathlib.Path(args.out_dir)

    fixed_blocks = {blocks_dim.col: blocks_per_dpu}

    p_x, p_y = (
        (blocks_dim, size_dim)
        if agg[blocks_dim.col].nunique() > 1
        else (size_dim, PROBLEM.group)
    )

    p = PlotPool()

    p.plot_vs_dim(
        agg,
        size_dim,
        PROBLEM.group,
        LATENCY,
        out_dir / "latency_vs_block_size.png",
        fixed=fixed_blocks,
    )
    p.plot_vs_dim(
        agg,
        PROBLEM.group,
        size_dim,
        LATENCY,
        out_dir / "latency_vs_num_dpus.png",
        fixed=fixed_blocks,
    )
    p.plot_vs_dim(
        agg,
        PROBLEM.derived["shape_product"],
        PROBLEM.group,
        LATENCY,
        out_dir / "latency_vs_bytes_per_dpu.png",
        connect=False,
    )
    p.plot_3d_measured(
        agg,
        x=p_x,
        y=p_y,
        z=LATENCY,
        color=PROBLEM.group,
        out_path=out_dir / "latency_3d.html",
    )
    if len(PROBLEM.shape) >= 2:
        p.plot_heatmap(
            agg,
            x=p_x,
            y=p_y,
            value=LATENCY,
            group=PROBLEM.group,
            group_value=num_dpus,
            out_path=out_dir / "heatmap_blocks_vs_size.png",
        )
    # p.plot_faceted_scatter(
    #     agg,
    #     x=PROBLEM.derived["total"],
    #     y=LATENCY,
    #     title="Total bytes, one panel per DPU count",
    #     out_path=out_dir / "latency_vs_total_bytes_faceted.png",
    #     facet=PROBLEM.group,
    #     color=size_dim,
    # )
    # p.plot_faceted_scatter(
    #     agg,
    #     x=PROBLEM.derived["shape_product"],
    #     y=LATENCY,
    #     title="Bytes per DPU, one panel per DPU count",
    #     out_path=out_dir / "latency_vs_bytes_per_dpu_faceted.png",
    #     facet=PROBLEM.group,
    #     color=size_dim,
    # )

    y = agg["ms"].to_numpy(dtype=float)
    table, fits = fit_and_report(templates, agg, PROBLEM, "All DPU counts")

    for key, tmpl in templates.items():
        if tmpl.report is not None:
            tmpl.report(fits[key])

    if splits:
        masks, labels = combined_regime_masks(agg, splits)

        mask_templates = [templates["pairwise"]]
        mask_templates = {t.key : t for t in mask_templates}
        regime_best_keys = []
        for mask, label in zip(masks, labels):
            sub = agg[mask]
            if sub.empty:
                regime_best_keys.append(None)
                print(f"Not building hybrid model because mask empty: {label}")
                continue
            regime_table, _ = fit_and_report(mask_templates, sub, PROBLEM, label)
            regime_best_keys.append(regime_table.iloc[0]["key"])

        # Hybrid: whichever template won each regime, fit only on that
        # regime's own rows, stitched together -- added to `fits`/`table`
        # alongside every whole-dataset template, so it competes for
        # best_key and shows up in regression_fit.png (and
        # regression_fit_best.png if it wins). Only meaningful with every
        # regime populated.
        if all(k is not None for k in regime_best_keys):
            hybrid_key = "hybrid"
            hybrid_name = "hybrid"
            #  + " / ".join(
            #     f"{templates[key].name} ({label})"
            #     for key, label in zip(regime_best_keys, labels)
            # )
            hybrid_fit = fit_regime_hybrid(
                templates, agg, y, splits, regime_best_keys
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
    p.plot_all_regression_fits(
        agg,
        y,
        fits,
        table,
        best_key,
        PROBLEM.group,
        LATENCY,
        out_dir / "regression_fit.png",
    )
    p.plot_best_regression_fit(
        agg,
        y,
        fits[best_key],
        PROBLEM.group,
        LATENCY,
        out_dir / "regression_fit_best.png",
    )
    p.plot_3d_predicted(
        agg,
        x=p_x,
        y=p_y,
        z=LATENCY,
        color=PROBLEM.group,
        z_values=fits[best_key]["pred"],
        template_name=fits[best_key]["name"],
        out_path=out_dir / "latency_3d_predicted_best.html",
    )
    if "hybrid" in fits:
        p.plot_3d_fit_wireframe(
            agg,
            x=p_x,
            y=p_y,
            z=LATENCY,
            pred=fits["hybrid"]["pred"],
            group=PROBLEM.group,
            template_name=fits["hybrid"]["name"],
            out_path=out_dir / "latency_3d_hybrid_wireframe.html",
            splits=fits["hybrid"]["splits"],
        )
    best_fit = fits[best_key]
    # p.plot_faceted_fit(
    #     agg,
    #     x=PROBLEM.derived["total"],
    #     value=LATENCY,
    #     pred=best_fit["pred"],
    #     facet=PROBLEM.group,
    #     title=f"Measured vs. {best_fit['name']} prediction, per DPU count",
    #     out_path=out_dir / "latency_vs_total_bytes_fit.png",
    # )
    print(f"\nBest fit: {best_fit['name']} (key={best_key})")
    print(
        f"  relRMSE = {best_fit['rel_rmse'] * 100:.2f}%   "
        f"RMSE = {best_fit['rmse']:.4g} ms   R² = {best_fit['r2']:.4f}"
    )
    if "regime_fits" in best_fit:
        for label, key, fit in zip(
            best_fit["regime_labels"], best_fit["regime_templates"], best_fit["regime_fits"]
        ):
            print(
                f"  if ({label}) {{\n"
                f"    intercept = {fit['intercept']:.7g};\n"
                f"    coef =  {{{', '.join(f'{c:.7g}' for c in fit['coef'])}}}; }}"
            )
        print("  double result = intercept + ", end=None)
        terms = [f"{name} * coef[{i}]" for i, name in enumerate(_pairwise_col_names(PROBLEM.all_dims))]
        print(' + '.join(terms), end=";\n")


    else:
        print(f"  intercept = {best_fit['intercept']:.4g};")
        print(f"  coef = {{{', '.join(f'{c:.4g}' for c in best_fit['coef'])}}};")
    p.join()
    print(f"\nPlots written to {out_dir}/")


if __name__ == "__main__":
    main()

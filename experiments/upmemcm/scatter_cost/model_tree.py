#!/usr/bin/env python3
"""Infer regime boundaries for the scatter cost model with a model tree.

analyze.py's --split regimes (see dodo.py's task_plot) are hand-found
boundaries: each one was guessed from the latency plots, baked into the
task, and validated after the fact by whether the per-regime hybrid fit
improved. This script learns the cuts instead: it fits a *model tree* (a
decision tree whose leaves are linear regressions, M5-style), where

  - splits are thresholds on the raw sweep dimensions (num_dpus,
    block_size, and blocks_per_dpu when it actually varies -- D/B/N for
    short) and, by default, also on their products (--split-terms raw
    restricts to the raw dimensions): a cut on B*N cuts on total bytes per
    DPU and one on D*B*N on total transferred bytes -- physically natural
    regime boundaries that axis-aligned cuts can only staircase around --
    and
  - every leaf fits the full interaction model on its own region: D, B,
    D*B for the 2-dimension benchmarks (block/broadcast/gather, where
    blocks_per_dpu is constant 1), plus N, D*N, B*N, D*B*N for scatter:sg
    -- i.e. analyze.py's _interaction_cols family.

Both the split search and the leaf fits use sample_weight = 1/y^2, the same
relative-error weighting as analyze.py's fit_ols (see relative_rmse there):
a candidate cut is scored by how much it reduces *relative* error, so the
tree spends its cuts where small-transfer predictions are bad, not where
the largest latencies happen to live.

Built on lineartree.LinearTreeRegressor (sklearn-compatible model trees;
sklearn's own DecisionTreeRegressor only fits constants in leaves), with
split_features restricted to the raw dimensions so thresholds never land on
an interaction column.

Reported at the end:
  - the tree itself, thresholds snapped back to actual sweep values so each
    cut reads exactly like a dodo.py --split boundary ("num_dpus <= 32"),
  - per-leaf coefficients (original units) and per-leaf relative RMSE,
  - the flat --split equivalent of all inferred cuts (NB: a tree is
    hierarchical, so the --split cross product is a superset of the tree's
    regions -- it's printed for direct comparison with dodo.py's manual
    boundaries, not as an identical model; B*N / D*B*N cuts map onto
    analyze.py's derived shape_product / total columns, other product cuts
    have no --split column and are called out as such),
  - a copy-pasteable C++ nested-if of the whole tree,
  - plots: the 2D partition of the (num_dpus, block_size) plane, points
    colored by leaf, cut lines drawn hierarchically -- an equal-product cut
    (num_dpus*block_size <= c) is a straight diagonal on the log-log axes
    -- (for scatter:sg, one such panel per blocks_per_dpu regime inferred
    from the tree's raw blocks_per_dpu cuts; product cuts involving
    blocks_per_dpu are drawn where the regime's representative value puts
    them, while each *point's color* is always its true leaf), and a
    measured-vs-predicted scatter for the stitched tree prediction.

Like analyze.py, everything is in-sample (no train/test split): the sweep
covers the whole domain the cost model will ever be asked about, and the
leaf models are tiny (<= 8 terms) relative to their row counts.

Usage:
  python3 model_tree.py plots/gather/results_agg.csv --out-dir plots/gather
  python3 model_tree.py plots/sg/results_agg.csv --out-dir plots/sg --max-depth 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from cinm_experiments import plots as shared_plots  # noqa: E402

from analyze import (  # noqa: E402
    LATENCY,
    PROBLEM,
    Dim,
    _add_colorbar,
    _apply_log_scale,
    _cmap_for,
    _interaction_col_names,
    _interaction_cols,
    _norm_for,
    fit_ols,
    relative_rmse,
)

# lineartree 0.3.x targets sklearn < 1.6: it calls the removed
# BaseEstimator._validate_data method (and its old force_all_finite kwarg).
# sklearn 1.6 moved that to the free function
# sklearn.utils.validation.validate_data and renamed the kwarg to
# ensure_all_finite -- shim both back onto lineartree's base class rather
# than pinning the whole venv's sklearn back to 1.5.
import lineartree._classes as _lineartree_classes  # noqa: E402
from lineartree import LinearTreeRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.utils.validation import validate_data  # noqa: E402


def _validate_data_shim(self, X="no_validation", y="no_validation", **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return validate_data(self, X=X, y=y, **kwargs)


_lineartree_classes._LinearTree._validate_data = _validate_data_shim


# ── Tree fitting ────────────────────────────────────────────────────────────


def build_design(agg: pd.DataFrame, dims: list[Dim]) -> tuple[np.ndarray, list[str]]:
    """Design matrix [each dim] + [pairwise products] + [product of all] and
    matching term names -- exactly analyze.py's _interaction_cols layout, as
    one float64 array. The first len(dims) columns are the raw dimensions
    themselves, which is what lets split_features address them by index."""
    X = np.column_stack(
        [np.asarray(c, dtype=float) for c in _interaction_cols(dims, agg)]
    )
    return X, _interaction_col_names(dims)


def fit_model_tree(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: list[int],
    max_depth: int,
    min_leaf: int,
    max_bins: int,
    min_impurity_decrease: float,
) -> tuple[LinearTreeRegressor, np.ndarray]:
    """Fit the model tree on a column-scaled copy of X and return (model,
    scale). lineartree casts X to float32 internally, and raw interaction
    products (num_dpus * blocks_per_dpu * block_size) reach ~1e11 -- big
    enough that an unscaled float32 least-squares fit inside each leaf
    loses real precision. Dividing every column by its max puts all of them
    in (0, 1]; thresholds and coefficients are mapped back to original
    units by the callers via `scale` (threshold_orig = threshold * scale[j],
    coef_orig = coef / scale[j]).

    min_impurity_decrease is in the split criterion's own units: weighted
    MSE with weight 1/y^2, i.e. mean squared *relative* error -- e.g. 1e-4
    means "don't split unless mean squared relative error drops by at least
    (1%)^2"."""
    scale = X.max(axis=0)
    scale[scale == 0] = 1.0
    model = LinearTreeRegressor(
        LinearRegression(),
        criterion="mse",
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        min_samples_split=max(6, 2 * min_leaf),
        max_bins=max_bins,
        min_impurity_decrease=min_impurity_decrease,
        split_features=split_indices,
        n_jobs=-1,
    )
    model.fit(X / scale, y, sample_weight=1.0 / y**2)
    return model, scale


def unscaled_threshold(node: dict, scale_by_col: dict[str, float]) -> float:
    """A split node's threshold in original data units (summary() reports it
    on the scaled columns fit_model_tree fed the tree)."""
    return float(node["th"]) * scale_by_col[node["col"]]


def snap_threshold(th: float, uniques: np.ndarray) -> float:
    """Largest actual value of the split column <= th. lineartree's
    thresholds are bin midpoints (e.g. 32.24); the sweep is a discrete
    grid, so the cut "num_dpus <= 32.24" is exactly "num_dpus <= 32" --
    snapping makes every reported cut directly comparable to a dodo.py
    --split boundary (and exact as a C++ integer comparison). Product
    columns (num_dpus*block_size, ...) snap against the products actually
    occurring in the sweep, for the same reason."""
    below = uniques[uniques <= th]
    return float(below.max()) if len(below) else float(uniques.min())


# ── Reporting ───────────────────────────────────────────────────────────────


def leaf_nodes(summary: dict) -> list[int]:
    return [nid for nid, node in summary.items() if "col" not in node]


def compute_leaf_stats(
    summary: dict, leaf_ids: np.ndarray, y: np.ndarray, pred: np.ndarray
) -> dict[int, dict]:
    stats = {}
    for nid in leaf_nodes(summary):
        mask = leaf_ids == nid
        stats[nid] = {
            "n": int(mask.sum()),
            "rel_rmse": relative_rmse(y[mask] - pred[mask], y[mask])
            if mask.any()
            else float("nan"),
        }
    return stats


def walk_tree(
    summary: dict,
    scale_by_col: dict[str, float],
    uniques: dict[str, np.ndarray],
) -> tuple[list[str], dict[int, str]]:
    """Renders the tree as indented if/else lines (thresholds snapped, raw
    threshold kept in a trailing comment for reference) and, as a byproduct,
    each leaf's full path condition ("num_dpus <= 32 && block_size > 1016"),
    used to label the leaf-model table."""
    lines: list[str] = []
    leaf_paths: dict[int, str] = {}

    def rec(nid: int, indent: str, path: list[str]) -> None:
        node = summary[nid]
        if "col" not in node:
            leaf_paths[nid] = " && ".join(path) if path else "(root)"
            lines.append(f"{indent}leaf #{nid}  (n={node['samples']})")
            return
        col = node["col"]
        snapped = snap_threshold(unscaled_threshold(node, scale_by_col), uniques[col])
        left, right = node["children"]
        lines.append(f"{indent}if {col} <= {snapped:g}  (n={node['samples']}):")
        rec(left, indent + "    ", path + [f"{col} <= {snapped:g}"])
        lines.append(f"{indent}else:")
        rec(right, indent + "    ", path + [f"{col} > {snapped:g}"])

    rec(0, "", [])
    return lines, leaf_paths


def collect_cuts(
    summary: dict, scale_by_col: dict[str, float], uniques: dict[str, np.ndarray]
) -> dict[str, list[float]]:
    """All snapped cut values per dimension, sorted -- the tree's answer to
    "which boundaries should --split use"."""
    cuts: dict[str, set[float]] = {}
    for node in summary.values():
        if "col" in node:
            col = node["col"]
            cuts.setdefault(col, set()).add(
                snap_threshold(unscaled_threshold(node, scale_by_col), uniques[col])
            )
    return {col: sorted(vals) for col, vals in cuts.items()}


def leaf_coefs(node: dict, scale: np.ndarray) -> tuple[float, np.ndarray]:
    """(intercept, coefs) of a leaf's LinearRegression in original units
    (the model was fit on X/scale, so coef_orig = coef_scaled / scale)."""
    model = node["models"]
    return float(model.intercept_), np.asarray(model.coef_).ravel() / scale


def emit_cpp(
    summary: dict,
    names: list[str],
    dims: list[Dim],
    scale: np.ndarray,
    scale_by_col: dict[str, float],
    uniques: dict[str, np.ndarray],
    fn_name: str,
) -> str:
    """The whole tree as one C++ function: nested threshold ifs, each leaf
    returning its interaction polynomial. The first factor of every product
    (in leaf terms and split conditions alike) is cast to double so int
    overflow (num_dpus * blocks_per_dpu * block_size exceeds int32) can't
    happen; single-dimension conditions stay exact integer comparisons."""

    def term(name: str) -> str:
        factors = name.split("*")
        if len(factors) == 1:
            return factors[0]
        return " * ".join([f"(double){factors[0]}"] + factors[1:])

    lines = [f"double {fn_name}({', '.join(f'int {d.col}' for d in dims)}) {{"]

    def rec(nid: int, indent: str) -> None:
        node = summary[nid]
        if "col" not in node:
            intercept, coefs = leaf_coefs(node, scale)
            expr = " + ".join(
                [f"{intercept:.7g}"]
                + [f"{c:.7g} * {term(n)}" for n, c in zip(names, coefs)]
            )
            lines.append(f"{indent}return {expr};")
            return
        col = node["col"]
        snapped = snap_threshold(unscaled_threshold(node, scale_by_col), uniques[col])
        left, right = node["children"]
        lines.append(f"{indent}if ({term(col)} <= {snapped:g}) {{")
        rec(left, indent + "  ")
        lines.append(f"{indent}}} else {{")
        rec(right, indent + "  ")
        lines.append(f"{indent}}}")

    rec(0, "  ")
    lines.append("}")
    return "\n".join(lines)


# ── Plots ───────────────────────────────────────────────────────────────────

_LEAF_CMAP = plt.get_cmap("tab20")


def _leaf_color(index: int):
    return _LEAF_CMAP(index % 20)


def draw_partition(
    ax,
    sub: pd.DataFrame,
    summary: dict,
    x: Dim,
    y: Dim,
    fixed: dict[str, float],
    leaf_ids: np.ndarray,
    leaf_order: dict[int, int],
    leaf_stats: dict[int, dict],
    scale_by_col: dict[str, float],
    uniques: dict[str, np.ndarray],
) -> None:
    """One panel of the partition plot: `sub`'s points in the (x, y) plane
    colored by their leaf, plus the tree's cut lines. `fixed` resolves any
    split factor that isn't one of the two plot axes (the blocks_per_dpu
    facet for scatter:sg) to a representative value.

    A cut whose factors reduce to a single plot axis (a raw-dimension cut,
    or a product cut whose other factors are all in `fixed` -- e.g. a
    blocks_per_dpu*block_size cut inside a fixed-blocks_per_dpu panel,
    drawn at block_size = threshold/blocks_per_dpu) is a vertical/
    horizontal line, and it only spans the box of the subtree it belongs
    to, so the hierarchical structure (a block_size cut that only exists
    below some num_dpus cut) is visible directly, unlike a flat --split
    grid. A cut involving both plot axes (num_dpus*block_size <= c) is an
    equal-product curve -- a straight diagonal on these log-log axes --
    drawn clipped to its subtree's box; such a cut doesn't shrink the
    recursion box (its child regions aren't rectangles), so a deeper cut's
    line can overshoot the diagonal that actually bounds its region.

    Leaf labels sit at the log-space median of each leaf's own points in
    this panel (not at a box center -- with diagonal cuts a leaf's region
    isn't a box, and a leaf can be entirely absent from a facet panel)."""
    ax.scatter(
        sub[x.col],
        sub[y.col],
        s=4,
        alpha=0.5,
        c=[_leaf_color(leaf_order[l]) for l in leaf_ids],
        linewidths=0,
    )

    pad = 1.15
    box = {
        x.col: (sub[x.col].min() / pad, sub[x.col].max() * pad),
        y.col: (sub[y.col].min() / pad, sub[y.col].max() * pad),
    }

    def rec(nid: int, box: dict) -> None:
        node = summary[nid]
        if "col" not in node:
            return
        col = node["col"]
        th = unscaled_threshold(node, scale_by_col)
        left, right = node["children"]

        # Fold the fixed factors' values into the threshold: a cut
        # "f1*...*fk <= th" with every fi in `fixed` except those on the
        # plot axes is, within this panel, a cut on the axis factors alone
        # at th / prod(fixed values).
        axis_factors = []
        const = 1.0
        for f in col.split("*"):
            if f in fixed:
                const *= fixed[f]
            else:
                axis_factors.append(f)
        eff = th / const
        snapped = snap_threshold(th, uniques[col])
        label = f"≤{snapped:g}" if "*" not in col else f"{col} ≤ {snapped:g}"

        if not axis_factors:
            rec(left if const <= th else right, box)
        elif axis_factors == [x.col]:
            ax.plot([eff, eff], box[y.col], color="black", linewidth=1)
            ax.text(
                eff,
                box[y.col][1],
                f" {label}",
                rotation=90,
                ha="left",
                va="top",
                fontsize=6,
                color="darkred",
            )
            rec(left, {**box, x.col: (box[x.col][0], eff)})
            rec(right, {**box, x.col: (eff, box[x.col][1])})
        elif axis_factors == [y.col]:
            ax.plot(box[x.col], [eff, eff], color="black", linewidth=1)
            ax.text(
                box[x.col][0],
                eff,
                label,
                ha="left",
                va="bottom",
                fontsize=6,
                color="darkred",
            )
            rec(left, {**box, y.col: (box[y.col][0], eff)})
            rec(right, {**box, y.col: (eff, box[y.col][1])})
        else:
            # Both axes involved: the boundary x*y = eff, sampled densely
            # (it's only straight in log-log space, and matplotlib
            # interpolates segments in data space).
            xs = np.geomspace(*box[x.col], 64)
            ys = eff / xs
            inside = (ys >= box[y.col][0]) & (ys <= box[y.col][1])
            if inside.any():
                ax.plot(xs[inside], ys[inside], color="black", linewidth=1)
                # Label at the diagonal's bottom-right end (anchored so the
                # text extends down-left, into the below-the-cut region) --
                # the midpoint would sit among the leaf labels, which
                # cluster around their regions' centers.
                end = np.flatnonzero(inside)[-1]
                ax.text(
                    xs[end],
                    ys[end],
                    label,
                    ha="right",
                    va="top",
                    fontsize=6,
                    color="darkred",
                )
            rec(left, box)
            rec(right, box)

    rec(0, box)

    def log_median(values: pd.Series) -> float:
        return float(np.exp(np.median(np.log(values))))

    for nid in np.unique(leaf_ids):
        pts = sub[leaf_ids == nid]
        ax.text(
            log_median(pts[x.col]),
            log_median(pts[y.col]),
            f"#{nid}\n{leaf_stats[nid]['rel_rmse'] * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    _apply_log_scale(ax, x, "x")
    _apply_log_scale(ax, y, "y")
    ax.set_xlabel(x.label)
    ax.set_ylabel(y.label)


def plot_partition(
    agg: pd.DataFrame,
    summary: dict,
    x: Dim,
    y: Dim,
    facet: Dim | None,
    leaf_ids: np.ndarray,
    leaf_stats: dict[int, dict],
    scale_by_col: dict[str, float],
    uniques: dict[str, np.ndarray],
    rel_rmse: float,
    out_path: pathlib.Path,
) -> None:
    """The partition figure. Without `facet` (block/broadcast/gather): one
    panel. With `facet` (scatter:sg's blocks_per_dpu): one panel per facet
    regime, where the regimes are the intervals between the tree's cuts on
    the *raw* facet dimension. When the tree only splits on raw dimensions,
    every facet value inside one interval resolves every tree comparison
    identically, so each panel's partition is exact for all the points it
    shows; product cuts involving the facet dimension (blocks_per_dpu*
    block_size etc.) break that -- their lines are drawn where the regime's
    representative (median) value puts them, and points elsewhere in the
    regime may sit on the "wrong" side of the drawn line. Point colors are
    always each point's true leaf, whatever the lines say."""
    leaf_order = {nid: i for i, nid in enumerate(sorted(leaf_nodes(summary)))}

    if facet is None:
        panels = [(np.ones(len(agg), dtype=bool), {}, None)]
    else:
        facet_cuts = [
            unscaled_threshold(node, scale_by_col)
            for node in summary.values()
            if node.get("col") == facet.col
        ]
        edges = [-np.inf] + sorted(set(facet_cuts)) + [np.inf]
        vals = agg[facet.col]
        panels = []
        for lo, hi in zip(edges, edges[1:]):
            mask = ((vals > lo) & (vals <= hi)).to_numpy()
            if not mask.any():
                continue
            lo_s = snap_threshold(lo, uniques[facet.col]) if np.isfinite(lo) else None
            hi_s = snap_threshold(hi, uniques[facet.col]) if np.isfinite(hi) else None
            title = (
                f"{facet.col} <= {hi_s:g}"
                if lo_s is None and hi_s is not None
                else f"{facet.col} > {lo_s:g}"
                if hi_s is None
                else f"{lo_s:g} < {facet.col} <= {hi_s:g}"
            )
            # Any value inside (lo, hi] resolves the tree identically; the
            # median actual value is as good a representative as any.
            rep = float(np.median(vals[mask]))
            panels.append((mask, {facet.col: rep}, title))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), squeeze=False)
    for ax, (mask, fixed, title) in zip(axes[0], panels):
        draw_partition(
            ax,
            agg[mask],
            summary,
            x,
            y,
            fixed,
            leaf_ids[mask],
            leaf_order,
            leaf_stats,
            scale_by_col,
            uniques,
        )
        if title:
            ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"Model-tree partition: {len(leaf_order)} leaves (labeled #id, own "
        f"relRMSE), total relRMSE={rel_rmse * 100:.2f}%",
        fontsize=10,
    )
    # rect reserves headroom for the suptitle, which plain tight_layout
    # ignores (it would let the axes clip into it).
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fit(
    agg: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    title: str,
    out_path: pathlib.Path,
) -> None:
    """Measured-vs-predicted for the stitched tree prediction, same style as
    analyze.py's regression_fit_best.png (colored by num_dpus)."""
    positive = np.concatenate([pred[pred > 0], y[y > 0]])
    pad = 1.15
    norm, _ = _norm_for(PROBLEM.group, agg[PROBLEM.group.col])
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = shared_plots.plot_measured_vs_predicted(
        pred,
        y,
        ax=ax,
        color=agg[PROBLEM.group.col],
        cmap=_cmap_for(PROBLEM.group),
        norm=norm,
        xlabel=f"predicted {LATENCY.label}",
        ylabel=f"measured {LATENCY.label}",
        lim=(positive.min() / pad, positive.max() * pad),
    )
    _add_colorbar(fig, sc, ax, PROBLEM.group, agg[PROBLEM.group.col])
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", help="Path to results_agg.csv (see dodo.py's agg task)")
    parser.add_argument(
        "--out-dir", default="plots", help="Output directory for plots (default: plots)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth: at most 2^depth leaves (default: 4)",
    )
    parser.add_argument(
        "--min-leaf",
        type=int,
        default=64,
        help="Minimum rows per leaf (default: 64) -- keep well above the leaf "
        "model's term count so no leaf is an underdetermined fit",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=119,
        help="Candidate thresholds tried per split dimension (default: 119, "
        "lineartree's maximum -- enough to place a cut between any two "
        "adjacent num_dpus values in this sweep; lower it to speed up "
        "scatter:sg's much larger fit)",
    )
    parser.add_argument(
        "--min-impurity-decrease",
        type=float,
        default=0.0,
        help="Minimum drop in mean squared relative error a split must "
        "achieve (e.g. 1e-4 = (1%%)^2; default: 0, i.e. grow to --max-depth "
        "subject only to --min-leaf)",
    )
    parser.add_argument(
        "--split-terms",
        choices=["all", "raw"],
        default="all",
        help="Which design-matrix columns the tree may split on: 'all' "
        "(default) allows cuts on the interaction products too (e.g. "
        "blocks_per_dpu*block_size = bytes per DPU, a physically natural "
        "transfer-regime boundary); 'raw' restricts cuts to the sweep "
        "dimensions themselves, keeping every region a plain box",
    )
    args = parser.parse_args()

    agg = pd.read_csv(args.csv)
    if agg.empty:
        raise SystemExit(f"{args.csv}: no rows (empty or header-only)")
    out_dir = pathlib.Path(args.out_dir)

    # Constant columns (blocks_per_dpu = 1 everywhere except scatter:sg)
    # carry no information for either splitting or regression.
    dims = [d for d in PROBLEM.all_dims if agg[d.col].nunique() > 1]
    X, names = build_design(agg, dims)
    y = agg["ms"].to_numpy(dtype=float)
    # Snap targets per split column -- for product columns these are the
    # products actually occurring in the sweep, not a raw dim's values.
    uniques = {name: np.unique(X[:, j]) for j, name in enumerate(names)}

    baseline = fit_ols(X, y)
    print(
        f"n={len(agg)} configs, dims: {', '.join(d.col for d in dims)}; "
        f"leaf terms: {', '.join(names)}"
    )
    print(
        f"baseline (single whole-dataset interaction fit): "
        f"relRMSE = {baseline['rel_rmse'] * 100:.2f}%   "
        f"RMSE = {baseline['rmse']:.4g} ms   R² = {baseline['r2']:.4f}"
    )

    model, scale = fit_model_tree(
        X,
        y,
        split_indices=list(
            range(X.shape[1] if args.split_terms == "all" else len(dims))
        ),
        max_depth=args.max_depth,
        min_leaf=args.min_leaf,
        max_bins=args.max_bins,
        min_impurity_decrease=args.min_impurity_decrease,
    )
    scale_by_col = {name: scale[j] for j, name in enumerate(names)}
    summary = model.summary(feature_names=names)
    Xs = X / scale
    pred = model.predict(Xs)
    leaf_ids = model.apply(Xs)
    assert set(leaf_ids) <= set(leaf_nodes(summary)), (
        "apply() leaf ids don't match summary() node ids"
    )

    resid = y - pred
    rel = relative_rmse(resid, y)
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    stats = compute_leaf_stats(summary, leaf_ids, y, pred)

    print(
        f"\n=== model tree (depth<={args.max_depth}, {len(stats)} leaves) ===\n"
        f"relRMSE = {rel * 100:.2f}%   RMSE = {rmse:.4g} ms   R² = {r2:.4f}"
    )
    lines, leaf_paths = walk_tree(summary, scale_by_col, uniques)
    print("\n".join(lines))

    print("\n=== leaf models (coefficients in original units) ===")
    for nid in sorted(stats):
        s = stats[nid]
        intercept, coefs = leaf_coefs(summary[nid], scale)
        print(f"leaf #{nid}: n={s['n']}, relRMSE={s['rel_rmse'] * 100:.2f}%")
        print(f"    where {leaf_paths[nid]}")
        print(f"    {'intercept':24s} {intercept:14.6g}")
        for name, c in zip(names, coefs):
            print(f"    {name:24s} {c:14.6g}")

    cuts = collect_cuts(summary, scale_by_col, uniques)
    print("\n=== inferred cuts ===")
    if not cuts:
        print("(none -- the tree never split; the baseline fit is the model)")
    for col, vals in cuts.items():
        print(f"{col}: {' '.join(f'{v:g}' for v in vals)}")
    if cuts:
        # Product cuts that match one of analyze.py's derived columns (see
        # ScatterProblem.add_derived_columns) can be expressed as --split
        # flags too; the other products can't and are only usable via this
        # script's own tree.
        aliases = {"*".join(d.col for d in dims): "total"}
        shape_cols = [d.col for d in dims if d.col != PROBLEM.group.col]
        if len(shape_cols) > 1:
            aliases["*".join(shape_cols)] = "shape_product"
        raw_cols = {d.col for d in dims}
        flags, unsplittable = [], []
        for col, vals in cuts.items():
            target = col if col in raw_cols else aliases.get(col)
            if target:
                flags.append(f"--split {target} {' '.join(f'{v:g}' for v in vals)}")
            else:
                unsplittable.append(col)
        if flags:
            print(
                "as analyze.py flags (cross product, a superset of the tree's "
                "regions):\n  " + " ".join(flags)
            )
        if unsplittable:
            print(
                "(cuts on "
                + ", ".join(unsplittable)
                + " have no analyze.py --split column and are omitted above)"
            )

    fn = pathlib.Path(args.csv).resolve().parent.name
    fn_name = f"scatter{fn.capitalize()}TreeCostMs"
    print("\n=== C++ ===")
    print(emit_cpp(summary, names, dims, scale, scale_by_col, uniques, fn_name))

    # x/y/facet follow PROBLEM.all_dims' order: num_dpus (group) and
    # block_size are always active; blocks_per_dpu (the middle dim) only
    # varies for scatter:sg and becomes the facet.
    x_dim, y_dim = dims[0], dims[-1]
    facet = dims[1] if len(dims) == 3 else None
    plot_partition(
        agg,
        summary,
        x_dim,
        y_dim,
        facet,
        leaf_ids,
        stats,
        scale_by_col,
        uniques,
        rel,
        out_dir / "model_tree_partition.png",
    )
    plot_fit(
        agg,
        y,
        pred,
        f"Model tree (depth<={args.max_depth}, {len(stats)} leaves)\n"
        f"relRMSE={rel * 100:.2f}%, R²={r2:.3f}, RMSE={rmse:.3g}ms",
        out_dir / "model_tree_fit.png",
    )
    print(
        f"\nPlots written to {out_dir}/model_tree_partition.png, "
        f"{out_dir}/model_tree_fit.png"
    )


if __name__ == "__main__":
    main()

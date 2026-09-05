#!/usr/bin/env python3
"""
Fit an additive overhead term f(dpus; params) that, added to a kernel-only
cost model's predicted cost, best matches the measured launch (kernel
execution) cost.

Two fitting strategies:
  1. Weighted non-negative least squares (NNLS) per template — optimises
     weighted RMSE.
  2. Bayesian optimisation via SMAC3 over (template, coefficients), cross-
     validated across all problems, maximising Spearman ρ@top-K (the metric
     most relevant for config selection: does the simulator rank the best
     configs correctly?).

Weighting for NNLS: every config is weighted by 1 / (mean measured_launch_cost
in its dpus group)^2, then divided by the group's sample count.

Usage:
  python3 fit_overhead_term.py --kernel-oracle ../../data/prim_red_oracle_notransfer
  python3 fit_overhead_term.py --kernel-oracle ../../data/prim_red_oracle_notransfer \\
      --smac-trials 300 --top-quantile 0.20
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

from plot_cost import find_function_pools, compute_measured_cost

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from cinm_experiments import plots as shared_plots  # noqa: E402

# ── Templates: name -> feature function (dpus, mramCols -> [n, k] feature matrix) ─

TEMPLATES = {
    "a (baseline)": lambda d, c: np.zeros((len(d), 0)),
    "a + b⋅dpus": lambda d, c: np.column_stack([d]),
    "a + b⋅⌊dpus/64⌋": lambda d, c: np.column_stack([d // 64]),
    "a + b⋅log₂(dpus)": lambda d, c: np.column_stack([np.log2(d)]),
    "a + b⋅dpus + c⋅⌊dpus/64⌋": lambda d, c: np.column_stack([d, d // 64]),
    "a + b⋅dpus + c⋅log₂(dpus)": lambda d, c: np.column_stack([d, np.log2(d)]),
}

# Pre-compute number of features per template (0, 1 or 2)
TEMPLATE_N_FEATURES: dict[str, int] = {
    name: fn(np.array([1.0]), np.array([1.0])).shape[1]
    for name, fn in TEMPLATES.items()
}


# ── Ranking metric ─────────────────────────────────────────────────────────────


def false_positive_count(predicted_ms: np.ndarray, measured_ms: np.ndarray) -> int:
    """Number of configs the simulator ranks better than the true optimum.

    Equivalently, the 0-based rank of argmin(measured_ms) in the predicted ranking.
    Zero means the simulator's top pick is the true best config.
    """
    true_best = np.argmin(measured_ms)
    return int(np.sum(predicted_ms < predicted_ms[true_best]))


def spearman_topk_rho(
    corrected_ms: np.ndarray, measured_ms: np.ndarray, top_quantile: float
) -> float:
    """Spearman ρ between corrected-predicted and measured cost, restricted to
    the top `top_quantile` fraction of configs by corrected predicted cost
    (ascending — lower cost is better).

    Returns 0.0 on degenerate inputs (< 2 points, all-identical predictions).
    """
    k = max(2, int(np.ceil(top_quantile * len(corrected_ms))))
    top_idx = np.argsort(corrected_ms)[:k]
    rho, _ = spearmanr(corrected_ms[top_idx], measured_ms[top_idx])
    return float(rho) if np.isfinite(rho) else 0.0


# ── Data helpers ───────────────────────────────────────────────────────────────


def _key_cols(pool: pd.DataFrame) -> list[str]:
    """Columns that form the config vector: everything before 'visited'."""
    cols = list(pool.columns)
    cut = cols.index("visited") if "visited" in cols else len(cols)
    return [c for c in cols[:cut] if c != "cost"]


def compute_measured_launch_cost(
    agg_dir: pathlib.Path, fn_name: str, key_cols: list
) -> pd.DataFrame:
    """Return columns [*key_cols, measured_launch_cost, n_launches] (ns), one row per config.

    For each (label, iteration), launch cost = sum of all dpu_launch call
    durations in that iteration (the kernel may launch multiple times per
    iteration, e.g. one launch per reduction-tree stage). measured_launch_cost
    is the mean of that per-iteration sum, over iterations, per config —
    for comparison against a cost model that predicts only the on-DPU kernel
    cost (no transfer/alloc/free overhead).

    n_launches is the mean number of dpu_launch rows per iteration (i.e. how
    many calls were summed over in the per-iteration groupby).
    """
    launch = pd.read_csv(agg_dir / "launch.csv")
    launch = launch[launch["fn_name"] == fn_name]
    grp = launch.groupby(["label", "iteration"])["elapsed_ns"]
    per_iter = pd.DataFrame(
        {
            "measured_launch_cost": grp.mean(),
            "n_launches": grp.count(),
        }
    )
    measured = per_iter.groupby("label").mean().reset_index()
    key_df = launch.drop_duplicates("label")[["label"] + key_cols]
    return measured.merge(key_df, on="label")[
        key_cols + ["measured_launch_cost", "n_launches"]
    ]


def build_full_cost_data(
    agg_dir: pathlib.Path, full_oracle_pool_csv: pathlib.Path, fn_name: str
) -> pd.DataFrame:
    """Join a full-simulation oracle pool with measured full-loop cost and n_launches.

    Returns a DataFrame with at least: dpus, mramCol, cost (ms, full simulation),
    measured_cost (ns), n_launches. Rows with NaN/inf in cost or measured_cost
    are dropped; no dpus filter is applied.
    """
    pool = pd.read_csv(full_oracle_pool_csv)
    key_cols = _key_cols(pool)
    mc = compute_measured_cost(agg_dir, fn_name, key_cols)
    if mc.empty:
        return pd.DataFrame()
    lc = compute_measured_launch_cost(agg_dir, fn_name, key_cols)

    pool = pool.merge(mc, on=key_cols, how="left")
    pool = pool.merge(lc[key_cols + ["n_launches"]], on=key_cols, how="left")
    pool = pool.dropna(subset=["measured_cost", "n_launches"])
    pool = pool[np.isfinite(pool["cost"]) & np.isfinite(pool["measured_cost"])]
    return pool


def build_clean_pool(
    agg_dir: pathlib.Path, pool_csv: pathlib.Path, fn_name: str
) -> pd.DataFrame:
    """measured_launch_cost/cost/dpus for fn_name, dropping unmeasured/timeout configs."""
    pool = pd.read_csv(pool_csv)
    key_cols = _key_cols(pool)
    measured = compute_measured_launch_cost(agg_dir, fn_name, key_cols)
    if measured.empty:
        return measured
    pool = pool.merge(measured, on=key_cols, how="left")
    data = pool.dropna(subset=["measured_launch_cost"])
    data = data[np.isfinite(data["cost"]) & np.isfinite(data["measured_launch_cost"])]
    data["cost"] = data["cost"] / data["n_launches"]
    return data


def group_weights(dpus: np.ndarray, measured_launch_cost_ms: np.ndarray) -> np.ndarray:
    """1 / (group mean measured_launch_cost)^2, spread evenly within each dpus group."""
    df = pd.DataFrame(
        {"dpus": dpus, "measured_launch_cost_ms": measured_launch_cost_ms}
    )
    group_mean = df.groupby("dpus")["measured_launch_cost_ms"].transform("mean")
    group_size = df.groupby("dpus")["measured_launch_cost_ms"].transform("size")
    return (1.0 / (group_mean**2 * group_size)).to_numpy()


def _build_X(dpus: np.ndarray, mramCols: np.ndarray, feature_fn) -> np.ndarray:
    """[ones | features] design matrix; safe for 0-feature (baseline) templates."""
    X = feature_fn(dpus, mramCols)
    if X.shape[1] == 0:
        return np.ones((len(dpus), 1))
    return np.column_stack([np.ones(len(dpus)), X])


def apply_correction(
    dpus: np.ndarray,
    mramCols: np.ndarray,
    intercept: float,
    coef: list[float],
    feature_fn,
) -> np.ndarray:
    """Evaluate f(dpus) = intercept + coef @ features for each config row."""
    return _build_X(dpus, mramCols, feature_fn) @ np.array([intercept] + list(coef))


# ── NNLS fitting ───────────────────────────────────────────────────────────────


def weighted_rmse(residual: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.average((residual - pred) ** 2, weights=weight)))


def fit_template(
    dpus: np.ndarray,
    mramCols: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    feature_fn,
) -> dict:
    """Weighted NNLS: coefficients >= 0, keeping the correction non-negative."""
    X_full = _build_X(dpus, mramCols, feature_fn)
    sqrt_w = np.sqrt(weight)
    coef_full, _ = nnls(X_full * sqrt_w[:, None], residual * sqrt_w)
    return {
        "intercept": float(coef_full[0]),
        "coef": list(coef_full[1:]),
        "pred": X_full @ coef_full,
    }


def fit_all_templates(
    dpus: np.ndarray, mramCols: np.ndarray, residual: np.ndarray, weight: np.ndarray
) -> pd.DataFrame:
    rows = []
    for name, feature_fn in TEMPLATES.items():
        fit = fit_template(dpus, mramCols, residual, weight, feature_fn)
        rmse = weighted_rmse(residual, fit["pred"], weight)
        rows.append(
            {
                "template": name,
                "weighted_rmse_ms": rmse,
                "intercept": fit["intercept"],
                "coef": fit["coef"],
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_rmse_ms").reset_index(drop=True)


# ── SMAC3 fitting ──────────────────────────────────────────────────────────────


def fit_smac(
    datasets: list[tuple],  # [(dpus, mramCols, predicted_ms, measured_ms, weight), ...]
    top_quantile: float = 0.20,
    rmse_weight: float = 0.5,
    cost_weighted_rmse: bool = True,
    n_trials: int = 200,
    seed: int = 42,
    output_dir: pathlib.Path = pathlib.Path("smac3_output"),
) -> dict:
    """Bayesian search over (template, coefficients) with a blended objective:

        loss = rmse_weight * norm_rmse + (1 - rmse_weight) * (1 - mean_rho)

    where norm_rmse = fold_rmse / baseline_rmse (1.0 = no improvement) and
    mean_rho = cross-validated Spearman ρ@top_quantile.

    If cost_weighted_rmse=True (default), the RMSE term uses per-config weights
    proportional to 1/measured_ms², so low-cost configs dominate and high-cost
    outliers are down-weighted quadratically. Otherwise the group-level weights
    (passed in the dataset tuples) are used.

    The baseline RMSE is always computed with the same weighting scheme so the
    normalisation is consistent.

    Returns {"template": str, "intercept": float, "coef": list[float],
             "cv_spearman_rho": float, "cv_normalized_rmse": float}.
    """
    from smac import Scenario, HyperparameterOptimizationFacade
    from ConfigSpace import (
        ConfigurationSpace,
        CategoricalHyperparameter,
        UniformFloatHyperparameter,
    )
    from ConfigSpace.conditions import InCondition

    # Estimate coefficient upper bounds from pooled residuals (3× 99th percentile).
    all_residuals = np.concatenate([m - p for _, _, p, m, _ in datasets])
    a_hi = float(np.percentile(np.abs(all_residuals), 99)) * 3.0
    b_hi = a_hi  # surrogate model learns per-template scales

    # Per-fold RMSE weights: 1/cost² (low-cost focused) or group weights.
    # Baseline uses the same scheme so normalisation stays meaningful.
    if cost_weighted_rmse:
        rmse_weights = [1.0 / (m**2) for _, _, _, m, _ in datasets]
        rmse_weights = [w / w.sum() for w in rmse_weights]
    else:
        rmse_weights = [w for _, _, _, _, w in datasets]

    baseline_rmses = [
        weighted_rmse(m - p, np.zeros(len(p)), rw)
        for (_, _, p, m, _), rw in zip(datasets, rmse_weights)
    ]

    template_names = list(TEMPLATES.keys())
    templates_with_b = [n for n, k in TEMPLATE_N_FEATURES.items() if k >= 1]
    templates_with_c = [n for n, k in TEMPLATE_N_FEATURES.items() if k >= 2]

    cs = ConfigurationSpace(seed=seed)
    t_hp = CategoricalHyperparameter("template", template_names)
    a_hp = UniformFloatHyperparameter("a", 0.0, a_hi)
    b_hp = UniformFloatHyperparameter("b", 0.0, b_hi)
    c_hp = UniformFloatHyperparameter("c", 0.0, b_hi)
    cs.add([t_hp, a_hp, b_hp, c_hp])
    cs.add_condition(InCondition(b_hp, t_hp, templates_with_b))
    cs.add_condition(InCondition(c_hp, t_hp, templates_with_c))

    def _coef_from_config(config):
        name = config["template"]
        n_feat = TEMPLATE_N_FEATURES[name]
        intercept = float(config["a"])
        b_val = config.get("b")
        c_val = config.get("c")
        b = float(b_val) if (n_feat >= 1 and b_val is not None) else 0.0
        c = float(c_val) if (n_feat >= 2 and c_val is not None) else 0.0
        coef = [] if n_feat == 0 else [b] if n_feat == 1 else [b, c]
        return intercept, coef

    def objective(config, seed: int = 0) -> float:
        name = config["template"]
        feature_fn = TEMPLATES[name]
        intercept, coef = _coef_from_config(config)

        rhos, norm_rmses = [], []
        for (
            dpus,
            mramCols,
            predicted_ms,
            measured_ms,
            _,
        ), rmse_w, baseline_rmse in zip(datasets, rmse_weights, baseline_rmses):
            corr = apply_correction(dpus, mramCols, intercept, coef, feature_fn)
            residual = measured_ms - predicted_ms
            rhos.append(
                spearman_topk_rho(predicted_ms + corr, measured_ms, top_quantile)
            )
            norm_rmses.append(weighted_rmse(residual, corr, rmse_w) / baseline_rmse)

        rho_loss = 1.0 - float(np.mean(rhos))
        rmse_loss = float(np.mean(norm_rmses))
        return rmse_weight * rmse_loss + (1.0 - rmse_weight) * rho_loss

    scenario = Scenario(
        cs,
        n_trials=n_trials,
        seed=seed,
        deterministic=True,
        output_directory=output_dir,
    )
    smac_inst = HyperparameterOptimizationFacade(scenario, objective, overwrite=True)
    incumbent = smac_inst.optimize()

    name = incumbent["template"]
    intercept, coef = _coef_from_config(incumbent)

    # Evaluate each component separately on the incumbent for reporting.
    feature_fn = TEMPLATES[name]
    rhos, norm_rmses = [], []
    for (dpus, mramCols, predicted_ms, measured_ms, _), rmse_w, baseline_rmse in zip(
        datasets, rmse_weights, baseline_rmses
    ):
        corr = apply_correction(dpus, mramCols, intercept, coef, feature_fn)
        residual = measured_ms - predicted_ms
        rhos.append(spearman_topk_rho(predicted_ms + corr, measured_ms, top_quantile))
        norm_rmses.append(weighted_rmse(residual, corr, rmse_w) / baseline_rmse)

    return {
        "template": name,
        "intercept": intercept,
        "coef": coef,
        "cv_spearman_rho": float(np.mean(rhos)),
        "cv_normalized_rmse": float(np.mean(norm_rmses)),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────


def make_fit_plot(
    dpus: np.ndarray,
    mramCols: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    best_name: str,
    out_path: pathlib.Path,
    title: str,
):
    fit = fit_template(dpus, mramCols, residual, weight, TEMPLATES[best_name])
    order = np.argsort(dpus)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(
        dpus,
        residual,
        s=10,
        alpha=0.4,
        color="darkorange",
        label="residual (measured - predicted)",
    )
    ax.plot(
        dpus[order],
        fit["pred"][order],
        color="crimson",
        linewidth=1.5,
        label=f"fit: {best_name}",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of DPUs")
    ax.set_ylabel("residual (ms)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_calibration_scatter_plot(
    dpus: np.ndarray,
    predicted_ms: np.ndarray,
    measured_ms: np.ndarray,
    correction_ms: np.ndarray,
    correction_label: str,
    top_quantile: float,
    out_path: pathlib.Path,
    title: str,
    correction_params: list[tuple[str, float]] | None = None,
    cmap: str = "viridis",
):
    """Two-panel scatter: raw predicted vs measured (left), corrected vs measured (right).
    Each panel is annotated with Spearman ρ, wRMSE, and (right panel) the fitted coefficients."""
    corrected_ms = predicted_ms + correction_ms
    rho_raw = spearman_topk_rho(predicted_ms, measured_ms, top_quantile)
    rho_corr = spearman_topk_rho(corrected_ms, measured_ms, top_quantile)
    w = 1.0 / (measured_ms**2)
    w = w / w.sum()
    rmse_raw = float(np.sqrt(np.average((predicted_ms - measured_ms) ** 2, weights=w)))
    rmse_corr = float(np.sqrt(np.average((corrected_ms - measured_ms) ** 2, weights=w)))
    k = max(2, int(np.ceil(top_quantile * len(predicted_ms))))

    k_min = int(np.floor(np.log2(dpus.min())))
    k_max = int(np.ceil(np.log2(dpus.max())))
    dpu_ticks = [2**i for i in range(k_min, k_max + 1)]

    # Shared axis limits across both panels for direct comparability
    all_x = np.concatenate([predicted_ms, corrected_ms])
    pad = 1.15  # multiplicative margin on log scale: equal visual gap on both sides
    lo = min(all_x.min(), measured_ms.min()) / pad
    hi = max(all_x.max(), measured_ms.max()) * pad
    dpu_norm = LogNorm(vmin=dpus.min(), vmax=dpus.max())

    # 3-column GridSpec: two equal plot columns + one narrow colorbar column.
    # This keeps both plotting areas the same width regardless of the colorbar.
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    sc = None
    for ax, x, subtitle, rho, rmse, params in [
        (ax0, predicted_ms, "without correction", rho_raw, rmse_raw, None),
        (
            ax1,
            corrected_ms,
            f"corrected ({correction_label})",
            rho_corr,
            rmse_corr,
            correction_params,
        ),
    ]:
        fp = false_positive_count(x, measured_ms)
        lines = [
            f"Spearman ρ@top-{top_quantile:.0%} (n={k}): {rho:.3f}",
            f"false positives (rank of true best): {fp}",
            f"wRMSE (1/cost²): {rmse:.3f} ms",
        ]
        if params:
            lines.extend(f"{name} = {val:.6g}" for name, val in params)
        sc = shared_plots.plot_measured_vs_predicted(
            x,
            measured_ms,
            ax=ax,
            color=dpus,
            norm=dpu_norm,
            cmap=cmap,
            xlabel="predicted kernel cost (ms)",
            ylabel="measured launch cost (ms)",
            title=subtitle,
            annotate_lines=lines,
            lim=(lo, hi),
        )

    cbar = fig.colorbar(sc, cax=cax, label="Number of DPUs", ticks=dpu_ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kernel-oracle",
        required=True,
        help="Path to kernel-only oracle directory (infer_{fn_name}/pool.csv layout)",
    )
    parser.add_argument(
        "--full-oracle",
        default=None,
        help="Path to full-simulation oracle directory (same layout, cost = full "
        "loop simulation cost in ms). When given, generates an additional "
        "scatter plot of full loop cost: predicted = full_oracle_cost + "
        "correction × n_launches, measured = total - alloc - free.",
    )
    parser.add_argument(
        "--in-dir",
        default="aggregated",
        help="Directory with aggregated CSVs (default: aggregated)",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Root output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.20,
        help="Fraction of top configs used for Spearman ρ ranking metric "
        "(default: 0.20 = top 20%%)",
    )
    parser.add_argument(
        "--smac-trials",
        type=int,
        default=200,
        help="SMAC3 evaluation budget (default: 200; 0 to skip SMAC)",
    )
    parser.add_argument(
        "--rmse-weight",
        type=float,
        default=0.5,
        help="Weight of the normalised RMSE term in the SMAC objective "
        "(0 = pure ranking / Spearman ρ only, "
        "1 = pure RMSE, default: 0.5)",
    )
    parser.add_argument(
        "--no-rmse-cost-weighted",
        dest="rmse_cost_weighted",
        action="store_false",
        default=True,
        help="Disable per-config 1/cost² weighting for the RMSE term "
        "(uses group-level weights instead). By default, low-cost "
        "configs dominate RMSE and high-cost outliers are suppressed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agg_dir = pathlib.Path(args.in_dir)
    plots_dir = pathlib.Path(args.plots_dir)
    top_q = args.top_quantile
    full_oracle = pathlib.Path(args.full_oracle) if args.full_oracle else None

    all_dpus, all_mramCols, all_residual_ms = [], [], []
    all_measured_ms, all_predicted_ms = [], []
    smac_datasets: list[tuple] = []

    # Accumulators for full-loop-cost scatter (only populated when --full-oracle given)
    all_full_dpus, all_full_mramCols = [], []
    all_full_predicted_ms, all_full_measured_ms, all_full_n_launches = [], [], []

    for fn_name, pool_csv in find_function_pools(pathlib.Path(args.kernel_oracle)):
        data = build_clean_pool(agg_dir, pool_csv, fn_name)
        if data.empty:
            print(f"  {fn_name}: no usable data, skipping", file=sys.stderr)
            continue

        dpus = data["dpus"].to_numpy(dtype=float)
        mramCols = data["mramCol"].to_numpy(dtype=float)
        predicted_ms = data["cost"].to_numpy()
        measured_ms = data["measured_launch_cost"].to_numpy() / 1e6
        residual_ms = measured_ms - predicted_ms
        weight = group_weights(dpus, measured_ms)

        results = fit_all_templates(dpus, mramCols, residual_ms, weight)
        print(f"\n=== {fn_name} (n={len(data)}) ===")
        print(results.to_string(index=False))

        best = results.iloc[0]["template"]
        fit = fit_template(dpus, mramCols, residual_ms, weight, TEMPLATES[best])
        rho = spearman_topk_rho(predicted_ms + fit["pred"], measured_ms, top_q)
        print(f"  NNLS best ({best}): Spearman ρ@top-{top_q:.0%} = {rho:.3f}")

        make_fit_plot(
            dpus,
            mramCols,
            residual_ms,
            weight,
            best,
            plots_dir / fn_name / "overhead_fit.png",
            f"{fn_name}: residual + best fit ({best})",
        )
        make_calibration_scatter_plot(
            dpus,
            predicted_ms,
            measured_ms,
            fit["pred"],
            f"NNLS: {best}",
            top_q,
            plots_dir / fn_name / "overhead_fit_calibration.png",
            f"{fn_name}: predicted vs measured kernel cost",
            correction_params=[("a", fit["intercept"])] + list(zip("bc", fit["coef"])),
            cmap="plasma",
        )

        if full_oracle is not None:
            full_pool_csv = full_oracle / f"infer_{fn_name}" / "pool.csv"
            if full_pool_csv.exists():
                fd = build_full_cost_data(agg_dir, full_pool_csv, fn_name)
                if not fd.empty:
                    fd_dpus = fd["dpus"].to_numpy(dtype=float)
                    fd_mram = fd["mramCol"].to_numpy(dtype=float)
                    fd_pred = fd["cost"].to_numpy()  # ms
                    fd_meas = fd["measured_cost"].to_numpy() / 1e6  # ns → ms
                    fd_nl = fd["n_launches"].to_numpy()
                    fd_corr = (
                        apply_correction(
                            fd_dpus,
                            fd_mram,
                            fit["intercept"],
                            fit["coef"],
                            TEMPLATES[best],
                        )
                        * fd_nl
                    )
                    make_calibration_scatter_plot(
                        fd_dpus,
                        fd_pred,
                        fd_meas,
                        fd_corr,
                        f"NNLS: {best}",
                        top_q,
                        plots_dir / fn_name / "full_cost_calibration.png",
                        f"{fn_name}: full loop cost (predicted vs measured)",
                        correction_params=[("a", fit["intercept"])]
                        + list(zip("bc", fit["coef"])),
                    )
                    all_full_dpus.append(fd_dpus)
                    all_full_mramCols.append(fd_mram)
                    all_full_predicted_ms.append(fd_pred)
                    all_full_measured_ms.append(fd_meas)
                    all_full_n_launches.append(fd_nl)

        all_dpus.append(dpus)
        all_mramCols.append(mramCols)
        all_residual_ms.append(residual_ms)
        all_measured_ms.append(measured_ms)
        all_predicted_ms.append(predicted_ms)
        smac_datasets.append((dpus, mramCols, predicted_ms, measured_ms, weight))

    if not all_dpus:
        print("No data found for any problem.", file=sys.stderr)
        sys.exit(1)

    dpus_pooled = np.concatenate(all_dpus)
    mramCols_pooled = np.concatenate(all_mramCols)
    residual_pooled = np.concatenate(all_residual_ms)
    measured_pooled = np.concatenate(all_measured_ms)
    predicted_pooled = np.concatenate(all_predicted_ms)
    weight_pooled = group_weights(dpus_pooled, measured_pooled)

    results_pooled = fit_all_templates(
        dpus_pooled, mramCols_pooled, residual_pooled, weight_pooled
    )
    print(f"\n=== ALL PROBLEMS POOLED (n={len(dpus_pooled)}) ===")
    print(results_pooled.to_string(index=False))

    best_pooled = results_pooled.iloc[0]["template"]
    fit_pooled = fit_template(
        dpus_pooled,
        mramCols_pooled,
        residual_pooled,
        weight_pooled,
        TEMPLATES[best_pooled],
    )
    rho_pooled = spearman_topk_rho(
        predicted_pooled + fit_pooled["pred"], measured_pooled, top_q
    )
    print(
        f"  NNLS best pooled ({best_pooled}): Spearman ρ@top-{top_q:.0%} = {rho_pooled:.3f}"
    )

    make_fit_plot(
        dpus_pooled,
        mramCols_pooled,
        residual_pooled,
        weight_pooled,
        best_pooled,
        plots_dir / "overhead_fit_pooled.png",
        f"All problems pooled: residual + best fit ({best_pooled})",
    )
    make_calibration_scatter_plot(
        dpus_pooled,
        predicted_pooled,
        measured_pooled,
        fit_pooled["pred"],
        f"NNLS: {best_pooled}",
        top_q,
        plots_dir / "overhead_fit_calibration_pooled.png",
        "All problems pooled: predicted vs measured kernel cost",
        correction_params=[("a", fit_pooled["intercept"])]
        + list(zip("bc", fit_pooled["coef"])),
        cmap="plasma",
    )

    if all_full_dpus:
        dpus_fp = np.concatenate(all_full_dpus)
        mramCols_fp = np.concatenate(all_full_mramCols)
        predicted_fp = np.concatenate(all_full_predicted_ms)
        measured_fp = np.concatenate(all_full_measured_ms)
        n_launches_fp = np.concatenate(all_full_n_launches)
        corr_fp = (
            apply_correction(
                dpus_fp,
                mramCols_fp,
                fit_pooled["intercept"],
                fit_pooled["coef"],
                TEMPLATES[best_pooled],
            )
            * n_launches_fp
        )
        make_calibration_scatter_plot(
            dpus_fp,
            predicted_fp,
            measured_fp,
            corr_fp,
            f"NNLS: {best_pooled}",
            top_q,
            plots_dir / "full_cost_calibration_pooled.png",
            "All problems pooled: full loop cost (predicted vs measured)",
            correction_params=[("a", fit_pooled["intercept"])]
            + list(zip("bc", fit_pooled["coef"])),
        )

    # ── SMAC: cross-validated ranking-objective fit ────────────────────────────
    if args.smac_trials <= 0:
        return

    print(
        f"\n=== SMAC (top-{top_q:.0%}, rmse-weight={args.rmse_weight}, "
        f"cost-weighted-rmse={args.rmse_cost_weighted}, "
        f"{args.smac_trials} trials, {len(smac_datasets)} folds) ==="
    )
    smac_result = fit_smac(
        smac_datasets,
        top_quantile=top_q,
        rmse_weight=args.rmse_weight,
        cost_weighted_rmse=args.rmse_cost_weighted,
        n_trials=args.smac_trials,
        seed=args.seed,
        output_dir=plots_dir / "smac3_output",
    )
    print(f"  best template      : {smac_result['template']}")
    print(f"  intercept          : {smac_result['intercept']:.6f}")
    print(f"  coef               : {smac_result['coef']}")
    print(
        f"  CV Spearman ρ@top-{top_q:.0%}: {smac_result['cv_spearman_rho']:.3f}  "
        f"(vs NNLS pooled: {rho_pooled:.3f})"
    )
    print(
        f"  CV normalised RMSE : {smac_result['cv_normalized_rmse']:.3f}  "
        f"(1.0 = no improvement over no correction)"
    )

    smac_correction = apply_correction(
        dpus_pooled,
        mramCols_pooled,
        smac_result["intercept"],
        smac_result["coef"],
        TEMPLATES[smac_result["template"]],
    )
    make_calibration_scatter_plot(
        dpus_pooled,
        predicted_pooled,
        measured_pooled,
        smac_correction,
        f"SMAC: {smac_result['template']}",
        top_q,
        plots_dir / "overhead_fit_calibration_smac_pooled.png",
        f"All problems pooled: predicted vs measured kernel cost "
        f"(SMAC, top-{top_q:.0%})",
        correction_params=[("a", smac_result["intercept"])]
        + list(zip("bc", smac_result["coef"])),
        cmap="plasma",
    )

    if all_full_dpus:
        smac_corr_fp = (
            apply_correction(
                dpus_fp,
                mramCols_fp,
                smac_result["intercept"],
                smac_result["coef"],
                TEMPLATES[smac_result["template"]],
            )
            * n_launches_fp
        )
        make_calibration_scatter_plot(
            dpus_fp,
            predicted_fp,
            measured_fp,
            smac_corr_fp,
            f"SMAC: {smac_result['template']}",
            top_q,
            plots_dir / "full_cost_calibration_smac_pooled.png",
            f"All problems pooled: full loop cost (predicted vs measured, SMAC, top-{top_q:.0%})",
            correction_params=[("a", smac_result["intercept"])]
            + list(zip("bc", smac_result["coef"])),
        )


if __name__ == "__main__":
    main()

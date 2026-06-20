#!/usr/bin/env python3
"""
Meta-optimizer: use SMAC3 to find the best MLP surrogate hyperparameters for
the UPMEM accelerator inference pass (--upmem-infer-accelerator).

For each SMAC trial the script runs cinm-opt with the candidate hyperparameters
in a temporary directory, reads the validation.csv produced by the C++ BO, and
evaluates three objectives:

  1. rmse        — final surrogate RMSE on the validation set  (lower is better)
  2. instability — total "upward" movement in the RMSE learning curve, weighted
                   so late-stage spikes count more              (lower is better)
  3. time        — total wall-clock time for the cinm-opt run   (lower is better)

Search parameters (meta-optimized):
  hidden, depth, n_init, neighbor_depth, epochs, neighbor_frontier_only

Usage:
    python bo_meta.py --input gemv [options]
    python bo_meta.py --input gemv.mlir --n-trials 100 --workers 4

Requires: smac, ConfigSpace, pandas, numpy
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn >= 1.3 removed DTYPE from the internal _tree module; SMAC still
# references it.  Restore the constant before SMAC is imported.
try:
    import sklearn.tree._tree as _skt
    if not hasattr(_skt, "DTYPE"):
        _skt.DTYPE = np.float32
except Exception:
    pass

try:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        UniformIntegerHyperparameter,
    )
    from smac import HyperparameterOptimizationFacade as HPOFacade
    from smac import Scenario
except ImportError as exc:
    print(f"SMAC / ConfigSpace not available: {exc}", file=sys.stderr)
    sys.exit(1)


# ── Scale helpers (mirrors plot_bo.py) ────────────────────────────────────────

def _apply_scale(costs: np.ndarray, scale: str) -> np.ndarray:
    eps = 1e-30
    if scale == "log2":  return np.log2(np.maximum(costs, eps))
    if scale == "ln":    return np.log(np.maximum(costs, eps))
    if scale == "sqrt":  return np.sqrt(np.maximum(costs, 0.0))
    if scale == "cbrt":  return np.cbrt(costs)
    if scale == "log10": return np.log10(np.maximum(costs, eps))
    return costs  # linear


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(val_csv: Path, scale: str) -> tuple[float, float]:
    """
    Parse validation.csv and return (final_rmse, instability).

    validation.csv columns: iter, mu, cost  (plus optional: tasklets, sigma)
      - iter: BO iteration at which this snapshot was recorded
      - mu:   surrogate prediction for this validation point
      - cost: true cost of this validation point

    final_rmse:
        RMSE between mu and scale(cost) at the last recorded BO iteration.
        This measures absolute surrogate accuracy at the end of learning.

    instability:
        How much the per-iteration RMSE fluctuates, with late-stage increases
        weighted more. Computed as:
          total_upward  = sum of max(0, rmse[i] - rmse[i-1])   # all rises
          late_std      = std of rmse in the last 25% of iters  # tail noise
          instability   = (total_upward + 2 * late_std) / rmse[0]
        Normalised by the initial RMSE so the value is scale-invariant.
    """
    val = pd.read_csv(val_csv)
    required = {"iter", "mu", "cost"}
    if not required.issubset(val.columns):
        missing = required - set(val.columns)
        raise ValueError(f"validation.csv missing columns: {missing}")

    scaled_true = _apply_scale(val["cost"].values, scale)
    val["sq_err"] = (val["mu"].values - scaled_true) ** 2

    rmse_by_iter = (
        val.groupby("iter")["sq_err"]
           .apply(lambda s: float(np.sqrt(s.mean())))
           .sort_index()
    )

    if rmse_by_iter.empty:
        return np.inf, np.inf

    rmse_vals = rmse_by_iter.values
    final_rmse = float(rmse_vals[-1])

    # Upward movement across the full curve
    diffs = np.diff(rmse_vals)
    total_upward = float(np.sum(np.maximum(0.0, diffs)))

    # Late-stage standard deviation (last 25% of iterations)
    n = len(rmse_vals)
    late = rmse_vals[max(0, n - max(1, n // 4)):]
    late_std = float(np.std(late)) if len(late) > 1 else 0.0

    # Normalise by initial RMSE so the metric is scale-invariant
    ref = rmse_vals[0] if rmse_vals[0] > 1e-12 else 1.0
    instability = (total_upward + 2.0 * late_std) / ref

    return final_rmse, instability


# ── Config space ───────────────────────────────────────────────────────────────

def build_config_space() -> ConfigurationSpace:
    """
    Hyperparameters of the BANANAS MLP ensemble used as the BO surrogate,
    plus BO search strategy parameters that affect surrogate quality.
    """
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        # MLP architecture
        CategoricalHyperparameter(
            "hidden", choices=[16, 32, 48, 64, 80, 96, 128, 256], default_value=64,
        ),
        UniformIntegerHyperparameter(
            "depth", lower=1, upper=8, default_value=2,
        ),
        # MLP training
        UniformIntegerHyperparameter(
            "epochs", lower=100, upper=3000, default_value=500,
        ),
        # BO initialization: how many LHS samples before the surrogate takes over
        # UniformIntegerHyperparameter(
        #     "n_init", lower=5, upper=25, default_value=20,
        # ),
        # Candidate neighbour generation
        # UniformIntegerHyperparameter(
        #     "neighbor_depth", lower=1, upper=5, default_value=1,
        # ),
        # CategoricalHyperparameter(
        #     "neighbor_frontier_only", choices=["true", "false"], default_value="false",
        # ),
    ])
    return cs


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(
    cfg,
    *,
    mlir_file: str,
    scale: str,
    max_evals: int,
    n_validation: int,
    validation_interval: int,
    simulator: str,
    cinm_opt: str,
    seed: int,
    split_input: bool,
    eval_timeout_ms: int,
    extra_opts: list[str],
    problem_name: str,
    trial_timeout: float,
) -> dict[str, float]:
    """
    Run cinm-opt with the given hyperparameter configuration in a temp directory
    and return the three objective values.
    Returns large penalty values on failure.
    """
    with tempfile.TemporaryDirectory(prefix="bo_meta_") as tmp:
        opts_parts = [
            f"objective-scale={scale}",
            f"dump-dir={tmp}",
            f"rng-seed={seed}",
            f"max-evals={max_evals}",
            f"n-init={cfg.get('n_init', 10)}",
            f"epochs={cfg['epochs']}",
            f"hidden-width={cfg['hidden']}",
            f"hidden-depth={cfg['depth']}",
            f"neighbor-depth={cfg.get('neighbor_depth', 2)}",
            f"eval-timeout-ms={cfg.get('eval_timeout_ms', 1000)}",
            f"neighbor-frontier-only={cfg.get('neighbor_frontier_only','false')}",
            "simulator=opcount",
            f"n-validation={n_validation}",
            f"validation-interval={validation_interval}",
            # f"simulator={simulator}", # TODO
        ] + extra_opts

        cmd = [
            cinm_opt, mlir_file,
            "--cinm-assign-platforms",
            "--cinm-isolate-compute-blocks",
            f"--upmem-infer-accelerator={' '.join(opts_parts)}",
            "--mlir-disable-threading",
        ]
        if split_input:
            cmd.append("--split-input-file")

        t0 = time.perf_counter()
        try:
            print(' '.join(cmd))
            proc = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                timeout=trial_timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            print(f"[bo_meta] trial timed out after {elapsed:.1f}s", file=sys.stderr)
            return {"rmse": 1e6, "instability": 1e6, "time": elapsed}
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            if not run_trial._warned:
                run_trial._warned = True
                stderr_excerpt = (proc.stderr or "").strip()[:800]
                print(
                    f"\n[bo_meta] WARNING: cinm-opt failed (exit {proc.returncode}). "
                    f"Subsequent failures will be silent.\n"
                    f"  cmd: {' '.join(cmd[:4])} ...\n"
                    f"  stderr: {stderr_excerpt or '(empty)'}",
                    file=sys.stderr,
                )
            return {"rmse": 1e6, "instability": 1e6, "time": elapsed}

        val_csv = Path(tmp) / problem_name / f"seed_{seed}" / "validation.csv"
        if not val_csv.exists():
            debug_dir = Path(f"data/bo_meta_debug")
            shutil.rmtree(debug_dir, ignore_errors=True)
            shutil.copytree(tmp, debug_dir)
            print(
                f"\n[bo_meta] WARNING: no validation.csv for problem '{problem_name}'. "
                f"Temp dir preserved at {debug_dir}",
                file=sys.stderr,
            )
            return {"rmse": 1e6, "instability": 1e6, "time": elapsed}

        try:
            final_rmse, instability = compute_metrics(val_csv, scale)
        except Exception as exc:
            print(f"[bo_meta] WARNING: metric computation failed: {exc}", file=sys.stderr)
            return {"rmse": 1e6, "instability": 1e6, "time": elapsed}

        return {
            "rmse":        final_rmse if np.isfinite(final_rmse) else 1e6,
            "instability": instability if np.isfinite(instability) else 1e6,
            "time":        elapsed,
        }


# Flag reset to False each run; set True after the first cinm-opt failure so
# we don't spam stderr with repeated identical error messages.
run_trial._warned = False


# ── SMAC objective factory ────────────────────────────────────────────────────

def make_objective(
    *,
    mlir_file: str,
    scale: str,
    max_evals: int,
    n_validation: int,
    validation_interval: int,
    simulator: str,
    cinm_opt: str,
    n_seeds: int,
    split_input: bool,
    extra_opts: list[str],
    eval_timeout_ms: int,
    problem_name: str,
    trial_timeout: float,
):
    """
    Returns the SMAC target function.  Each call runs `n_seeds` independent
    cinm-opt invocations (different RNG seeds) and averages their objectives
    to reduce noise.
    """
    def objective(smac_cfg, seed: int = 0) -> dict:
        results = [
            run_trial(
                smac_cfg,
                mlir_file=mlir_file,
                scale=scale,
                max_evals=max_evals,
                n_validation=n_validation,
                validation_interval=validation_interval,
                simulator=simulator,
                cinm_opt=cinm_opt,
                seed=(seed * 100 + s + 1) % (2**31),
                split_input=split_input,
                eval_timeout_ms=eval_timeout_ms,
                extra_opts=extra_opts,
                problem_name=problem_name,
                trial_timeout=trial_timeout,
            )
            for s in range(n_seeds)
        ]
        return {
            "rmse":        float(np.mean([r["rmse"]        for r in results])),
            "instability": float(np.mean([r["instability"] for r in results])),
            "time":        float(np.mean([r["time"]        for r in results])),
        }

    return objective


# ── Pareto plot ───────────────────────────────────────────────────────────────

def plot_pareto(smac, out_path: str) -> None:
    """
    Save a figure with three pairwise scatter plots of the three objectives:
      (rmse, instability), (rmse, time), (instability, time)

    All evaluated configs are shown as small transparent blue dots; Pareto-front
    incumbents are overlaid as large red crosses.
    """
    PAIRS = [(0, 1), (0, 2), (1, 2)]
    LABELS = ["RMSE (final)", "Instability", "Time (s)"]
    PENALTY = 1e5

    incumbents = set(smac.intensifier.get_incumbents())

    # Iterate over every individual trial in the run history.
    # runhistory.items() yields (TrialKey, TrialValue); TrialValue.cost is the
    # raw multi-objective cost list for that single run.
    bg_pts, fg_pts = [], []
    for trial_key, trial_value in smac.runhistory.items():
        cost = trial_value.cost
        if cost is None:
            continue
        if not hasattr(cost, "__len__") or len(cost) < 3:
            continue
        arr = [float(v) for v in cost]
        if any(v >= PENALTY for v in arr):
            continue  # failed / penalised trial
        cfg = smac.runhistory.get_config(trial_key.config_id)
        (fg_pts if cfg in incumbents else bg_pts).append(arr)

    bg = np.array(bg_pts) if bg_pts else np.empty((0, 3))
    fg = np.array(fg_pts) if fg_pts else np.empty((0, 3))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (i, j) in zip(axes, PAIRS):
        if bg.size:
            ax.scatter(bg[:, i], bg[:, j], c="steelblue", alpha=0.25, s=18,
                       linewidths=0, label="evaluated", zorder=2)
        if fg.size:
            ax.scatter(fg[:, i], fg[:, j], c="red", alpha=1.0, s=60,
                       marker="x", linewidths=1.5, label="Pareto front", zorder=3)
            # Connect the Pareto front with a step line sorted by the x-axis objective
            order = np.argsort(fg[:, i])
            ax.step(fg[order, i], fg[order, j], where="post",
                    color="red", lw=0.8, alpha=0.5, zorder=3)
        ax.set_xlabel(LABELS[i])
        ax.set_ylabel(LABELS[j])
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Meta-BO Pareto front  (3 objectives)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[bo_meta] Pareto plot saved to {out_path}")


# Order of axes in the 3D scatter plot — edit to taste.
# Must be a permutation of the three objective names: "rmse", "instability", "time".
PLOT_3D_AXES = ["time", "rmse", "instability"]

_OBJ_INDEX = {"rmse": 0, "instability": 1, "time": 2}
_OBJ_LABEL = {"rmse": "RMSE (final)", "instability": "Instability", "time": "Time (s)"}


def plot_pareto_3d(smac, out_path: str) -> None:
    """
    Save a 3D scatter plot of all trials in objective space.
    The axis order is controlled by PLOT_3D_AXES at the top of this section.
    Pareto-front incumbents are shown as red stars; all other trials as small
    transparent blue dots.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    xi, yi, zi = [_OBJ_INDEX[k] for k in PLOT_3D_AXES]
    xl, yl, zl = [_OBJ_LABEL[k] for k in PLOT_3D_AXES]
    PENALTY = 1e5

    incumbents = set(smac.intensifier.get_incumbents())

    bg_pts, fg_pts = [], []
    for trial_key, trial_value in smac.runhistory.items():
        cost = trial_value.cost
        if cost is None:
            continue
        if not hasattr(cost, "__len__") or len(cost) < 3:
            continue
        arr = [float(v) for v in cost]
        if any(v >= PENALTY for v in arr):
            continue
        cfg = smac.runhistory.get_config(trial_key.config_id)
        (fg_pts if cfg in incumbents else bg_pts).append(arr)

    bg = np.array(bg_pts) if bg_pts else np.empty((0, 3))
    fg = np.array(fg_pts) if fg_pts else np.empty((0, 3))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if bg.size:
        ax.scatter(bg[:, xi], bg[:, yi], bg[:, zi],
                   c="steelblue", alpha=0.25, s=20, linewidths=0, label="evaluated")
    if fg.size:
        ax.scatter(fg[:, xi], fg[:, yi], fg[:, zi],
                   c="red", alpha=1.0, s=80, marker="*", linewidths=0,
                   label="Pareto front")

    ax.set_xlabel(xl, labelpad=8)
    ax.set_ylabel(yl, labelpad=8)
    ax.set_zlabel(zl, labelpad=8)
    ax.legend(fontsize=9)
    fig.suptitle("Meta-BO trials — 3D objective space", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[bo_meta] 3D plot saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", required=True, metavar="FILE[.mlir]",
        help="MLIR input file (the .mlir extension is added if omitted)",
    )
    ap.add_argument(
        "--problem", required=True, metavar="NAME",
        help="Problem (function name in the input file) used to compute the objectives",
    )
    ap.add_argument(
        "--scale", default="log10",
        metavar="SCALE",
        help="Objective transform used by BO: linear, log2, log10, ln, sqrt, cbrt "
             "(must match the surrogate's training scale; default: log10)",
    )
    ap.add_argument(
        "--max-evals", type=int, default=60,
        help="BO evaluations per cinm-opt trial (controls how long each trial runs; "
             "default: 60)",
    )
    ap.add_argument(
        "--n-validation", type=int, default=30,
        help="Size of the held-out validation set used to measure surrogate RMSE "
             "(default: 30)",
    )
    ap.add_argument(
        "--validation-interval", type=int, default=5,
        help="Record a surrogate snapshot on the validation set every N BO iterations "
             "(default: 5)",
    )
    ap.add_argument(
        "--eval-timeout-ms", type=int, default=1000,
        help="Simulation timeout (milliseconds)"
             "(default: 1000)",
    )
    ap.add_argument(
        "--simulator", default="cycleaccurate",
        choices=["cycleaccurate", "opcount"],
        help="DPU cost simulator used inside cinm-opt (default: cycleaccurate)",
    )
    ap.add_argument(
        "--n-seeds", type=int, default=2,
        help="Number of BO seeds averaged per SMAC trial to reduce noise (default: 2)",
    )
    ap.add_argument(
        "--n-trials", type=int, default=80,
        help="SMAC meta-optimization budget in number of trials (default: 80)",
    )
    ap.add_argument(
        "--workers", type=int, default=max(1, ((os.cpu_count() or 2)- 1) // 2),
        help="Parallel SMAC trial workers.  Each worker spawns n_seeds cinm-opt "
             "processes, so total concurrency = workers × n_seeds.  Default: ncpu/2.",
    )
    ap.add_argument(
        "--cinm-opt", default="cinm-opt", metavar="PATH",
        help="Path to cinm-opt binary (default: cinm-opt found on PATH)",
    )
    ap.add_argument(
        "--smac-dir", default="data/bo_meta_smac", metavar="DIR",
        help="Directory for SMAC output / run history (default: data/bo_meta_smac)",
    )
    ap.add_argument(
        "--no-split-input-file", action="store_true", dest="no_split_input_file",
        help="Do NOT pass --split-input-file to cinm-opt. "
             "By default it is always passed (as run_seeds.sh does).",
    )
    ap.add_argument(
        "--trial-timeout", type=float, default=60.0, metavar="SECONDS",
        help="Kill cinm-opt if a single seed takes longer than this many seconds "
             "(default: 60). Timed-out trials receive maximum penalty on all objectives.",
    )
    ap.add_argument(
        "--extra-opts", nargs="*", default=[], metavar="KEY=VAL",
        help="Additional key=value options forwarded verbatim to "
             "--upmem-infer-accelerator (e.g. kappa=2.5 n-ensemble=7)",
    )
    args = ap.parse_args()

    mlir_file = args.input
    if not mlir_file.endswith(".mlir"):
        mlir_file += ".mlir"
    if not os.path.exists(mlir_file):
        print(f"ERROR: input file not found: {mlir_file}", file=sys.stderr)
        sys.exit(1)
    # Use absolute path so dask/multiprocessing workers find the file regardless
    # of their working directory.
    mlir_file = os.path.abspath(mlir_file)

    if args.n_validation < 2:
        print("ERROR: --n-validation must be at least 2 to compute RMSE", file=sys.stderr)
        sys.exit(1)

    objective = make_objective(
        mlir_file=mlir_file,
        scale=args.scale,
        max_evals=args.max_evals,
        n_validation=args.n_validation,
        validation_interval=args.validation_interval,
        simulator=args.simulator,
        cinm_opt=args.cinm_opt,
        n_seeds=args.n_seeds,
        split_input=not args.no_split_input_file,
        extra_opts=args.extra_opts or [],
        problem_name=args.problem,
        eval_timeout_ms=args.eval_timeout_ms,
        trial_timeout=args.trial_timeout,
    )

    cs = build_config_space()
    scenario = Scenario(
        cs,
        objectives=["rmse", "instability", "time"],
        n_trials=args.n_trials,
        n_workers=args.workers,
        seed=1,
        output_directory=Path(args.smac_dir),
    )

    smac = HPOFacade(
        scenario=scenario,
        target_function=objective,
        logging_level=10,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(

            scenario,
            # Weights for RMSE, instability, and time
            objective_weights=[0.5, 1, 0.6],
        ),
        overwrite=True,
    )

    print("[bo_meta] Starting meta-optimization")
    print(f"  input:               {mlir_file}")
    print(f"  scale:               {args.scale}")
    print(f"  simulator:           {args.simulator}")
    print(f"  max_evals / trial:   {args.max_evals}")
    print(f"  n_validation:        {args.n_validation}")
    print(f"  validation_interval: {args.validation_interval}")
    print(f"  n_seeds / trial:     {args.n_seeds}")
    print(f"  trial_timeout:       {args.trial_timeout}s")
    print(f"  smac_trials:         {args.n_trials}")
    print(f"  smac_workers:        {args.workers}")
    print()

    incumbents = smac.optimize()

    print("\n── Pareto-optimal configurations " + "─" * 40)
    header = f"{'rmse':>10}  {'instability':>12}  {'time':>8}  config"
    print(header)
    print("─" * len(header))
    for cfg in incumbents:
        costs = smac.runhistory.average_cost(cfg)
        if hasattr(costs, "__len__") and len(costs) >= 3:
            rmse, instab, t = float(costs[0]), float(costs[1]), float(costs[2])
            cfg_str = "  ".join(f"{k}={v}" for k, v in sorted(dict(cfg).items()))
            print(f"  {rmse:10.4f}  {instab:12.4f}  {t:7.1f}s  {cfg_str}")
        else:
            print(f"  cost={costs}  {dict(cfg)}")

    plot_pareto(smac, os.path.join(args.smac_dir, "pareto_front.png"))
    plot_pareto_3d(smac, os.path.join(args.smac_dir, "pareto_front_3d.png"))

    runner = getattr(smac, "_runner", None)
    if runner is not None:
        try:
            runner.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

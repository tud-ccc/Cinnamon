"""doit tasks for the paper evaluation (docs/EvaluationImplementationPlan.md).

This pipeline collects the measurements behind the paper's §8: the shared
uniform sample (B1), the space dumps the transcription workflow reads (B0),
and -- as later phases land -- the top-k, search, transcription-point,
cinm1-sweep and RQ4 stacks. Structure and conventions follow
cinm1comparison/dodo.py: stages connected by files, fallible-per-config
compiles, ONE strict hardware-bench chain, retry tasks. The shared
machinery lives in cinm_experiments.doit_blocks.

Phase 0 scope (this file today): B0 `space` + B1 `sample` -> compile ->
bench, with retries. See the plan's §5 for the stacks still to come and
the layout they will occupy.

Usage:
  doit space             # dump every benchmark's space.json (no simulator)
  doit sample            # draw the shared uniform sample (simulator only)
  doit                   # everything up to bench_sample (hardware!)
  doit retry_failed_compiles && doit
  doit retry_failed_bench && doit bench_sample
"""

from __future__ import annotations

import pathlib
import sys

from doit import create_after
from doit.reporter import ProgressBarReporter

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from cinm_experiments import cinmopt, compile_run, doit_blocks, pools, ALL_PRIMS  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402

# ── the paper's constants ────────────────────────────────────────────────────
# These values appear verbatim in the paper (CI arithmetic, A3, RQ3), so they
# are set once here and never inline; the rationale for each is the OPTS
# table in docs/EvaluationImplementationPlan.md §5.
OPTS = dict(
    # 3/n rule of three: 0-of-300 beaten => the point is in the top 1% of the
    # feasible space (95% CI) -- the number the paper quotes.
    n_sample=300,
    # Fixed draw: the sample is THE shared sample (E1(b), RQ3's ground truth,
    # §2's best-vs-median, A2's accepted-but-fails). Changing the seed
    # invalidates every measured row, so it is part of the methodology.
    sample_seed=1815,
    k_top=200,  # B3 (not yet wired): measured ground truth for top-k overlap
    n_seeds=32,  # B4 (not yet wired): matches the cinm1comparison campaign
    iters=6,  # measurement repetitions per hardware run
    # The hybrid model's switching point, same value as the search stack
    # (cinm1comparison): cycle-accurate within the budget, fast model past
    # it. Part of the model's definition, not a performance knob -- B1's
    # predicted costs and every search must use the same value.
    eval_timeout_ms=400,
)

WORKLOADS = list(ALL_PRIMS)
DATA_DIR = HERE / "data"

DOIT_CONFIG = {
    "default_tasks": ["bench_sample"],
    "verbosity": 2,
    "continue": True,
    "reporter": ProgressBarReporter,
}


# ── paths ────────────────────────────────────────────────────────────────────


def source_mlir(bench: str) -> pathlib.Path:
    return EXPERIMENTS_DIR / f"{bench}.mlir"


def split_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "_split"


def split_module(bench: str, fn_name: str) -> pathlib.Path:
    return split_dir(bench) / f"{fn_name}.mlir"


def space_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "space"


def space_json(bench: str, fn_name: str) -> pathlib.Path:
    return space_dir(bench) / f"infer_{fn_name}" / "space.json"


def sample_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "sample"


def sample_pool_csv(bench: str, fn_name: str) -> pathlib.Path:
    return sample_dir(bench) / f"infer_{fn_name}" / "pool.csv"


def sample_roots(bench: str) -> doit_blocks.MeasureRoots:
    return doit_blocks.MeasureRoots(
        compile_root=sample_dir(bench) / "compiled",
        run_root=sample_dir(bench) / "run",
    )


# ── split ────────────────────────────────────────────────────────────────────


def _split_one(src: pathlib.Path, out_dir: pathlib.Path) -> bool:
    split_source(src, out_dir)
    return True


def task_split():
    """Split each benchmark's source into one module per function."""
    for bench in WORKLOADS:
        src = source_mlir(bench)
        fns = list_functions(src)
        yield {
            "name": bench,
            "file_dep": [str(src)],
            "targets": [str(split_module(bench, fn)) for fn in fns],
            "actions": [(_split_one, [src, split_dir(bench)])],
        }


# ── B0: space dumps ──────────────────────────────────────────────────────────


def _dump_space(bench: str) -> bool:
    cinmopt.dump_space(source_mlir(bench), space_dir(bench))
    return True


def task_space():
    """B0: dump every function's space.json -- sizes for tab:sufficiency,
    per-param docs + permutation tables for the manual ATiM transcription
    (plan §7). No simulator runs; safe anywhere."""
    for bench in WORKLOADS:
        yield {
            "name": bench,
            "file_dep": [str(source_mlir(bench))],
            "targets": [
                str(space_json(bench, fn)) for fn in list_functions(source_mlir(bench))
            ],
            "actions": [(_dump_space, [bench])],
        }


# ── B1: the shared uniform sample ────────────────────────────────────────────


def _draw_sample(bench: str) -> bool:
    cinmopt.random_sample(
        source_mlir(bench),
        sample_dir(bench),
        n_samples=OPTS["n_sample"],
        seed=OPTS["sample_seed"],
        infer_opts={
            # EXACTLY the model the search stack consumes -- simulator and
            # timeout together, because they are one model: hybrid runs
            # cycle-accurate under eval-timeout-ms and answers with the fast
            # model when it fires (UpmemPythonSimulator.cpp, SimMode::HYBRID)
            # -- with no timeout, "hybrid" is just cycle-accurate. RQ3's
            # fidelity claim is about what the flow chooses with, so the
            # sample's predicted costs must come from the same pairing the
            # search uses. The timeout does NOT censor the sample: a
            # timed-out config is priced by the fallback, not dropped.
            "simulator": "hybrid",
            "eval-timeout-ms": OPTS["eval_timeout_ms"],
            # What WOULD censor the sample is the cost cap, which rejects
            # and resamples: off, or the E1 percentile (rule of three needs
            # a uniform draw) and RQ3's rank correlation are void.
            "sample-max-cost-ms": 0,
        },
    )
    return True


def task_sample():
    """B1: draw the shared uniform sample -- ONE set of n_sample configs per
    function, drawn uniformly at random from the enumerated feasible set
    with a fixed seed, predicted costs recorded. Serves E1(b), RQ3's
    fidelity ground truth, §2's best-vs-median, and A2's accepted-but-fails
    check all at once (plan §2-B1); nothing downstream may redraw it."""
    for bench in WORKLOADS:
        yield {
            "name": bench,
            "file_dep": [str(source_mlir(bench))],
            "targets": [
                str(sample_pool_csv(bench, fn))
                for fn in list_functions(source_mlir(bench))
            ],
            "actions": [(_draw_sample, [bench])],
        }


# ── B2 on the sample: compile + bench every sampled config ──────────────────


def _sample_configs(bench: str) -> list[compile_run.Config]:
    """One Config per sampled row, labelled by its row position ("s000",
    "s001", ...) -- positions are stable because the sample itself is (fixed
    seed, never redrawn)."""
    configs = []
    for fn_name in list_functions(source_mlir(bench)):
        pool_csv = sample_pool_csv(bench, fn_name)
        for i, params in enumerate(pools.sample_rows(pool_csv)):
            configs.append(
                compile_run.Config(
                    system="sample",
                    fn_name=fn_name,
                    label=f"s{i:03d}",
                    params=params,
                    fn_module=split_module(bench, fn_name),
                    prim=bench.removeprefix("prim_"),
                    lower=cinmopt.eval_solution_lowerer(),
                )
            )
    return configs


def _bench_task_name(bench: str, fn_name: str, label: str) -> str:
    return f"bench_sample:{bench}:{fn_name}:{label}"


@create_after(executed="sample", creates=["compile_sample", "bench_sample"])
def task_compile_sample():
    """Compile + bench every sampled config. Compiles are fallible per
    config; benches form ONE strict chain across all benchmarks (hardware
    timing, see doit_blocks.BenchChain). An eval-solution compile failure
    here is A2 DATA (accepted-but-fails must be 0), not just noise -- keep
    the logs."""
    chain = doit_blocks.BenchChain()
    per_bench: list[tuple[str, compile_run.Config]] = []
    for bench in WORKLOADS:
        for config in _sample_configs(bench):
            per_bench.append((bench, config))
            chain.register(_bench_task_name(bench, config.fn_name, config.label))

    for bench, config in per_bench:
        roots = sample_roots(bench)
        marker = roots.compile_marker_of(config)
        pool_csv = sample_pool_csv(bench, config.fn_name)
        yield {
            "basename": "compile_sample",
            "name": f"{bench}:{config.fn_name}:{config.label}",
            "file_dep": [str(pool_csv), str(config.fn_module)],
            "targets": [str(marker)],
            "actions": [(doit_blocks.compile_one, [config, roots, marker])],
        }

        bench_marker = roots.bench_marker_of(config)
        yield {
            "basename": "bench_sample",
            "name": f"{bench}:{config.fn_name}:{config.label}",
            # pool.csv too: a redrawn sample (parameter change) must
            # invalidate the measurements, not silently re-attribute them.
            "file_dep": [str(marker), str(pool_csv)],
            "task_dep": chain.prev_of(
                _bench_task_name(bench, config.fn_name, config.label)
            ),
            "targets": [str(bench_marker)],
            "actions": [
                (
                    doit_blocks.bench_one_config,
                    [config, roots],
                    dict(iters=OPTS["iters"], bench_marker=bench_marker),
                )
            ],
        }


# ── retries ──────────────────────────────────────────────────────────────────


def _retry_failed_compiles() -> bool:
    for bench in WORKLOADS:
        doit_blocks.clear_failed_compiles(_sample_configs(bench), sample_roots(bench))
    return True


def task_retry_failed_compiles():
    """Not part of the default pipeline; run explicitly after fixing the
    cause, then rerun `doit`."""
    return {"actions": [_retry_failed_compiles], "uptodate": [False]}


def _retry_failed_bench() -> bool:
    for bench in WORKLOADS:
        doit_blocks.clear_failed_bench(_sample_configs(bench), sample_roots(bench))
    return True


def task_retry_failed_bench():
    """Not part of the default pipeline; run explicitly after fixing the
    cause, then rerun `doit bench_sample`."""
    return {"actions": [_retry_failed_bench], "uptodate": [False]}


# ── still to come (plan §5) ──────────────────────────────────────────────────
# B3 topk:    exhaustive predicted sweep (eval-timeout-ms=300) -> pools.top_k
#             -> same B2 shape as the sample stack.
# B4 search:  bo_multiseed(n_seeds) x {default, ablated spaces, simulators};
#             timings.csv feeds RQ2.
# B5 points:  points/atim/{bench}.json + points/cinm1rule/{bench}.json ->
#             eval-solution runs + invariants report (plan §7).
# cinm1:      the (D,T) sweep with coverage accounting (plan §4.4).
# assemble/plot: results/*.csv, missing-tolerant (plan §2-B6, §6).

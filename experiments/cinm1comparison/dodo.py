"""doit tasks for the CINM 1.0 vs CINM 2.0 comparison experiment.

Same pipeline as experiment.py's module docstring (working groups -> CINM1
configs / CINM2 search -> compile -> run -> compare -> plot), but driven by
doit instead of a plain top-to-bottom script: each stage declares its file
inputs/outputs, so `doit` only reruns what's actually stale -- e.g. if a
CINM1 compile fails and you fix cinm1.py and rerun, the CINM2 Bayesian
search (expensive, already done) is not repeated.

Usage:
  doit list              # show all tasks (some only appear after upstream
                          # tasks that create them dynamically have run once)
  doit                   # run everything up to the plots
  doit compile_cinm1     # just compile CINM 1.0's configs
  doit forget pairs      # force the working groups to be re-enumerated
  doit retry_failed_compiles && doit  # clear + retry configs that failed to compile
  doit retry_failed_bench && doit bench_cinm1 bench_cinm2  # clear + retry configs that failed on hardware

Stages are connected by files on disk, not in-memory state, since doit may
skip any stage in a given invocation: pairs.csv (the working groups, see
task_pairs) and the CINM 2.0 search's pool.csv files are the source of truth
read back by every downstream stage.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys

import pandas as pd
from doit import create_after
from doit.tools import result_dep
from doit.reporter import ProgressBarReporter  # noqa: E402
# from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))


from cinm_experiments import cinm1, cinmopt, compile_run, measurements, pools, ALL_PRIMS  # noqa: E402
from cinm_experiments import doit_blocks  # noqa: E402
from cinm_experiments import prims as prim_defs  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402

from plot import (
    geomean,
    plot_best_speedup,
    plot_speedup,
    plot_speedup_violin,
    print_summary,
)  # noqa: E402

## TODO reenable mmtv and ttv after optimizations
PRIMS = set(ALL_PRIMS).difference(("prim_gemv",))
DATA_DIR = HERE / "data"

OPTS = dict(
    n_seeds=32,
    iters=6,
    workers=os.cpu_count(),
)

# Seed k in a multiseed run uses k*31+offset for k in 1..n_seeds; offsets must
# be spaced by more than n_seeds*31 apart so different pairs' seed_* dirs
# never collide within the same {fn_name}/ results directory.
_OFFSET_STRIDE = 4096
_BASE_OFFSET = 67

DOIT_CONFIG = {
    "default_tasks": ["plot"],
    "verbosity": 2,
    "continue": True,
    # Pipelines here run into the tens of thousands of leaf tasks (BO
    # search seeds x working groups x functions); the default console
    # reporter's one-line-per-task log is unreadable at that scale.
    # Override with `doit -r console` for a single invocation if you need
    # the full per-task log back (e.g. while debugging a specific task).
    "reporter": ProgressBarReporter,
}


@dataclasses.dataclass(frozen=True)
class Paths:
    """Every on-disk path this pipeline reads or writes. Centralized here
    because stages are wired together by files on disk, not in-memory state
    (see module docstring) -- the same paths would otherwise be re-derived
    piecemeal in nearly every task. config_dir/compile_marker/bench_bin/
    run_output_dir key off a compile_run.Config's (system, fn_name, label)
    -- not its `prim` field, which is the unrelated bench PRIM= op name
    ("gemv"/"red"), not the "prim_gemv"/"prim_red" directory prefix used
    here."""

    experiments_dir: pathlib.Path
    data_dir: pathlib.Path

    def source_mlir(self, prim: str) -> pathlib.Path:
        return self.experiments_dir / f"{prim}.mlir"

    def prim_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim

    def split_dir(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "_split"

    def split_module(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.split_dir(prim) / f"{fn_name}.mlir"

    def pairs_csv(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.prim_dir(prim) / "pairs" / f"{fn_name}.csv"

    def cinm2_results_dir(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "cinm2_results"

    def cinm2_search_marker(
        self, prim: str, fn_name: str, dpus: int, tasklets: int
    ) -> pathlib.Path:
        return (
            self.prim_dir(prim)
            / "cinm2_search_markers"
            / fn_name
            / f"D{dpus}_T{tasklets}.done"
        )

    def cinm2_unconstrained_results_dir(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "cinm2_unconstrained_results"

    def roots(self, prim: str) -> doit_blocks.MeasureRoots:
        """The per-prim compile/run layout, as the shared doit_blocks
        machinery consumes it. The bench marker it derives is per-config and
        touched whether the run succeeded or not (bench_one_config), so an
        interrupted `doit bench` resumes config-by-config."""
        return doit_blocks.MeasureRoots(
            compile_root=self.prim_dir(prim) / "compiled",
            run_root=self.prim_dir(prim) / "run",
        )

    def compile_root(self, prim: str) -> pathlib.Path:
        return self.roots(prim).compile_root

    def compile_marker(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.roots(prim).compile_marker_of(config)

    def bench_bin(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.roots(prim).bench_bin_of(config)

    def run_root(self, prim: str) -> pathlib.Path:
        return self.roots(prim).run_root

    def run_output_dir(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.roots(prim).run_output_dir_of(config)

    def bench_marker(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.roots(prim).bench_marker_of(config)

    def comparison_csv(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "comparison.csv"

    def comparison_best_csv(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "comparison_best.csv"

    def plots_dir(self) -> pathlib.Path:
        return self.data_dir / "plots"


PATHS = Paths(EXPERIMENTS_DIR, DATA_DIR)


# ── split ────────────────────────────────────────────────────────────────────


def _split_one(prim_mlir: pathlib.Path, split_dir: pathlib.Path) -> bool:
    split_source(
        prim_mlir, split_dir
    )  # dict return value isn't JSON-picklable for doit's DB
    return True


def task_split():
    """Split each prim's source into one module per function."""
    for prim in PRIMS:
        prim_mlir = PATHS.source_mlir(prim)
        split_dir = PATHS.split_dir(prim)
        fns = list_functions(prim_mlir)
        yield {
            "name": prim,
            "file_dep": [str(prim_mlir)],
            "targets": [str(split_dir / f"{fn}.mlir") for fn in fns],
            "actions": [(_split_one, [prim_mlir, split_dir])],
        }


# ── working groups ───────────────────────────────────────────────────────────


def _prim_def(prim: str) -> prim_defs.Prim:
    """The Prim behind a "prim_<name>" source-module stem."""
    return prim_defs.PRIMS[prim.removeprefix("prim_")]


def _write_pairs(prim: str, fn_name: str, out_csv: pathlib.Path) -> bool:
    p = _prim_def(prim)
    max_dpus, max_tasklets = prim_defs.platform_limits(p)
    pairs = p.working_groups(fn_name, max_dpus=max_dpus, max_tasklets=max_tasklets)
    if not pairs:
        raise RuntimeError(f"no feasible working group for {prim}:{fn_name}")
    print(
        f"  {fn_name:20s}  parallel extent {p.parallel_extent(fn_name)}"
        f" -> {len(pairs)} (dpus,tasklets) pairs"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs, columns=["dpus", "tasklets"]).to_csv(out_csv, index=False)
    return True


def task_pairs():
    """Enumerate the (dpus, tasklets) working groups to measure: every group
    whose worker count divides the function's parallel extent, i.e. every one
    the kernel can be distributed over at all (Prim.working_groups).

    This replaced a screening stage that swept CINM 2.0's cost model over the
    whole configuration space and kept the best-predicted 10% of working
    groups. Nothing here consults a cost model: which groups are worth
    measuring is exactly the question the experiment asks, and screening on a
    prediction answered part of it in advance -- with CINM 2.0's own model,
    the very thing under test. The condition left is structural (does the work
    divide evenly?), so the set is the whole feasible space rather than a
    sample of it, at the cost of measuring groups that turn out to be slow.

    Cheap and deterministic -- it reads the problem dimensions and the
    platform's limits, and runs no compiler -- so unlike the screening it
    replaced there is nothing to preserve across runs; `doit forget pairs`
    costs nothing. One subtask per function, whose extents are its own."""
    for prim in PRIMS:
        for fn_name in list_functions(PATHS.source_mlir(prim)):
            pairs_csv = PATHS.pairs_csv(prim, fn_name)
            yield {
                "name": f"{prim}:{fn_name}",
                # The source module, not the split one: the extents and the
                # #upmem.platform limits both come from it.
                "file_dep": [str(PATHS.source_mlir(prim))],
                "targets": [str(pairs_csv)],
                "actions": [(_write_pairs, [prim, fn_name, pairs_csv])],
            }


# ── CINM 2.0 search ──────────────────────────────────────────────────────────


def _cinm2_search_one(
    prim: str,
    fn_name: str,
    fn_module: pathlib.Path,
    *,
    n_seeds: int,
    offset: int,
    extra_infer_opts: dict,
    results_dir: pathlib.Path,
    label: str,
) -> bool:
    """Run one CINM 2.0 BO search (n_seeds independent runs sharing the
    config-space setup). Shared by both search shapes _cinm2_search_groups
    yields: extra_infer_opts pins (dpus, tasklets) for the matched sweep, or
    is empty to leave them free for the unconstrained sweep."""
    print(f"  {fn_name} {label}: BO search ({n_seeds} seeds, offset={offset})")
    cinmopt.bo_multiseed(
        fn_module,
        results_dir,
        n_seeds=n_seeds,
        offset=offset,
        workers=OPTS["workers"],
        infer_opts={
            **extra_infer_opts,
            "simulator": "hybrid",
            "eval-timeout-ms": 400,
            "max-evals": 100,
            "n-init": 10,
        },
    )
    return True


def get_offset(pair_idx):
    return _BASE_OFFSET + pair_idx * _OFFSET_STRIDE


def gen_seeds(pair_idx):
    offset = get_offset(pair_idx)
    return (31 * k + offset for k in range(0, OPTS["n_seeds"]))


@dataclasses.dataclass(frozen=True)
class _Cinm2SearchGroup:
    """One CINM 2.0 BO search (n_seeds independent runs) that will run for
    one function: either dpus/tasklets pinned to one of the enumerated (dpus,
    tasklets) working groups (the matched sweep, comparable 1:1 against CINM
    1.0's own config -- see compare()), or left free (the unconstrained sweep,
    comparable against CINM 1.0's best-ever config -- see
    task_compare_best/plot_best_speedup). _cinm2_search_groups yields these;
    task_cinm2_search builds identical search/compile/bench tasks from
    either kind, since they only differ in what's in this dataclass.

    system must be "cinm2" + <the basename suffix task_cinm2_search should
    append to "cinm2_search"/"compile_cinm2"/"bench_cinm2">, e.g. "cinm2" (no
    suffix) or "cinm2_unconstrained" ("_unconstrained" suffix) -- not an
    independently-chosen tag -- because task_cinm2_search derives that suffix
    back out of the system string (system.removeprefix("cinm2")), which is
    also how config_ids identifies the config's run directory."""

    system: str  # "cinm2" | "cinm2_unconstrained"
    task_label: str  # e.g. "D8_T4" | "unconstrained"
    extra_infer_opts: dict
    offset: int
    results_dir: pathlib.Path
    seeds: tuple[int, ...]


def _cinm2_search_groups(prim: str, fn_name: str):
    pairs_csv = PATHS.pairs_csv(prim, fn_name)
    if pairs_csv.exists():
        pairs = pd.read_csv(pairs_csv)
        for pair_idx, (dpus, tasklets) in enumerate(
            pairs[["dpus", "tasklets"]].itertuples(index=False)
        ):
            dpus, tasklets = int(dpus), int(tasklets)
            yield _Cinm2SearchGroup(
                system="cinm2",
                task_label=f"D{dpus}_T{tasklets}",
                extra_infer_opts={"fixed-dpus": dpus, "fixed-tasklets": tasklets},
                offset=get_offset(pair_idx),
                results_dir=PATHS.cinm2_results_dir(prim),
                seeds=tuple(gen_seeds(pair_idx)),
            )
    # Unconstrained sweep: dpus/tasklets left free, one search per function
    # instead of one per working group. Reuses gen_seeds/get_offset(0) as-is
    # (same seed count and offset as the matched sweep's pair 0) rather than
    # a separate scheme -- safe to reuse the same offset because results
    # land in cinm2_unconstrained_results_dir, never cinm2_results_dir, so
    # there's no seed_<k>/pool.csv path collision with pair 0's matched
    # search either way.
    yield _Cinm2SearchGroup(
        system="cinm2_unconstrained",
        task_label="unconstrained",
        extra_infer_opts={},
        offset=get_offset(0),
        results_dir=PATHS.cinm2_unconstrained_results_dir(prim),
        seeds=tuple(gen_seeds(0)),
    )


def _config_ids():
    """Yield (prim, system, fn_name, label, dpus, tasklets, pair_idx) for
    every config that will exist once task_pairs has written pairs.csv, for
    every prim in PRIMS. Unlike a compile_run.Config's params/lower, these
    identities are fully known as soon as the working groups are --
    compile/run directories are keyed off (system, fn_name, label) alone,
    and CINM 2.0's dpus/tasklets are pinned to the pair's before its search
    even starts -- so this is the single place task_compile_cinm1 derives
    the set of CINM 1.0 configs (and their bench_cinm1 tasks) from, without
    waiting on search or compile results."""
    for prim in PRIMS:
        for fn_name in list_functions(PATHS.source_mlir(prim)):
            pairs_csv = PATHS.pairs_csv(prim, fn_name)
            if not pairs_csv.exists():
                continue
            pairs = pd.read_csv(pairs_csv)
            for pair_idx, (dpus, tasklets) in enumerate(
                pairs[["dpus", "tasklets"]].itertuples(index=False)
            ):
                dpus, tasklets = int(dpus), int(tasklets)
                yield (
                    prim,
                    "cinm1",
                    fn_name,
                    f"D{dpus}_T{tasklets}",
                    dpus,
                    tasklets,
                    pair_idx,
                )
                for seed in gen_seeds(pair_idx):
                    yield prim, "cinm2", fn_name, str(seed), dpus, tasklets, pair_idx

        for fn_name in list_functions(PATHS.source_mlir(prim)):
            for seed in gen_seeds(0):
                yield prim, "cinm2_unconstrained", fn_name, str(seed), None, None, None


@create_after(
    executed="pairs",
    creates=[
        "cinm2_search",
        "compile_cinm2",
        "bench_cinm2",
        "cinm2_search_unconstrained",
        "compile_cinm2_unconstrained",
        "bench_cinm2_unconstrained",
    ],
)
def task_cinm2_search():
    """Run CINM 2.0's Bayesian search for every function, in both shapes
    _cinm2_search_groups yields: once per enumerated (dpus, tasklets) working
    group with that pair pinned (matched sweep, MRAM tiling enabled -- CINM
    2.0's normal codegen), and once more with dpus/tasklets left free
    (unconstrained sweep, feeds task_compare_best/plot_best_speedup's
    steelmanned-CINM-1.0 comparison). Every seed's own best-in-pool config
    gets compiled and benched individually, in both sweeps -- not just the
    group's single best -- so seed-to-seed variance stays visible in
    comparison.csv/comparison_best.csv and their violin plots."""
    if OPTS["n_seeds"] * 31 >= _OFFSET_STRIDE:
        raise RuntimeError(
            f"n_seeds {OPTS['n_seeds']} too large for offset stride {_OFFSET_STRIDE}"
        )

    for prim in PRIMS:
        op = prim.removeprefix("prim_")
        roots = PATHS.roots(prim)
        for fn_name in list_functions(PATHS.source_mlir(prim)):
            fn_module = PATHS.split_module(prim, fn_name)
            for group in _cinm2_search_groups(prim, fn_name):
                basename_suffix = group.system.removeprefix("cinm2")
                yield {
                    "basename": "cinm2_search" + basename_suffix,
                    "name": f"{prim}:{fn_name}:{group.task_label}",
                    # The function's own source, not the pairs.csv this
                    # group's (dpus, tasklets) came from: enumerating one
                    # more working group must not invalidate the searches
                    # already done for the other groups in that same file
                    # (each search's result depends on its own pinned pair,
                    # which is in its task name, not on the rest of the set).
                    "file_dep": [str(fn_module)],
                    "targets": [
                        str(
                            group.results_dir
                            / f"infer_{fn_name}"
                            / f"seed_{seed}"
                            / "pool.csv"
                        )
                        for seed in group.seeds
                    ],
                    "actions": [
                        (
                            _cinm2_search_one,
                            [prim, fn_name, fn_module],
                            dict(
                                n_seeds=len(group.seeds),
                                offset=group.offset,
                                extra_infer_opts=group.extra_infer_opts,
                                results_dir=group.results_dir,
                                label=group.task_label,
                            ),
                        )
                    ],
                }

                for seed in group.seeds:
                    seed = str(seed)
                    config = compile_run.Config(
                        system=group.system,
                        fn_name=fn_name,
                        label=seed,
                        # Will be replaced once we know which config params are the best
                        params={},
                        fn_module=fn_module,
                        prim=op,
                        lower=cinmopt.eval_solution_lowerer(),
                    )
                    pool_csv = (
                        group.results_dir
                        / f"infer_{fn_name}"
                        / f"seed_{seed}"
                        / "pool.csv"
                    )
                    marker = PATHS.compile_marker(prim, config)
                    yield {
                        "basename": "compile_cinm2" + basename_suffix,
                        "name": f"{prim}:{fn_name}:{group.task_label}:{seed}",
                        "file_dep": [str(pool_csv)],
                        "targets": [str(marker)],
                        "actions": [
                            (
                                doit_blocks.compile_best,
                                [config, pool_csv, roots, marker],
                            )
                        ],
                    }

                    bench_marker = PATHS.bench_marker(prim, config)
                    yield {
                        "basename": "bench_cinm2" + basename_suffix,
                        "name": f"{prim}:{fn_name}:{seed}",
                        # On the search's pool.csv as well as the compile
                        # marker, because the marker cannot express this: it
                        # is an empty file, so recompiling a seed whose search
                        # found a *different* best config leaves it byte-identical
                        # and the measurement of the config it replaced would
                        # stay on disk, attributed to the new one. A cinm2 run
                        # directory is keyed by seed, and which working group a
                        # seed belongs to is the pair's position in pairs.csv
                        # (get_offset) -- so enumerating a different set of
                        # working groups is exactly the case where this
                        # happens, and the stale measurement would be one of
                        # another group entirely.
                        "file_dep": [str(marker), str(pool_csv)],
                        # Real hardware: never beside anything else.
                        "exclusive": True,
                        "targets": [str(bench_marker)],
                        "actions": [
                            (
                                doit_blocks.bench_one_config,
                                [config, roots],
                                dict(
                                    iters=OPTS["iters"],
                                    bench_marker=bench_marker,
                                ),
                            )
                        ],
                    }


# ── compile ──────────────────────────────────────────────────────────────────


# Compile/bench actions live in cinm_experiments.doit_blocks (compile_one,
# compile_best, bench_one_config) -- shared with the other experiment dodos.


@create_after(executed="pairs", creates=["bench_cinm1", "compile_cinm1"])
def task_compile_cinm1():
    """Compile CINM 1.0 once per enumerated working group -- no search, its
    tile sizes are inferred deterministically. Also generates that config's
    bench_cinm1 task (basename "bench_cinm1", see doit_blocks.bench_one_config)
    right here, so the set of CINM 1.0 configs is derived exactly once instead
    of separately for compile and bench."""
    for prim, system, fn_name, label, dpus, tasklets, _ in _config_ids():
        if system != "cinm1":
            continue
        op = prim.removeprefix("prim_")
        roots = PATHS.roots(prim)
        fn_module = PATHS.split_module(prim, fn_name)
        config = compile_run.Config(
            system="cinm1",
            fn_name=fn_name,
            label=label,
            params={"dpus": dpus, "tasklets": tasklets},
            fn_module=fn_module,
            prim=op,
            lower=cinm1.lowerer(),
        )
        marker = PATHS.compile_marker(prim, config)
        yield {
            "basename": "compile_cinm1",
            "name": f"{prim}:{fn_name}:{label}",
            # The function's own source, not the pairs.csv this config's
            # (dpus, tasklets) came from -- same reason as cinm2_search's
            # file_dep: this config is identified by its label, so enumerating
            # one more working group must not recompile the others.
            "file_dep": [str(fn_module)],
            "targets": [str(marker)],
            "actions": [(doit_blocks.compile_one, [config, roots, marker])],
        }

        bench_marker = PATHS.bench_marker(prim, config)
        yield {
            "basename": "bench_cinm1",
            "name": f"{prim}:{fn_name}:{label}",
            "file_dep": [str(marker)],
            # Real hardware: never beside anything else.
            "exclusive": True,
            "targets": [str(bench_marker)],
            "actions": [
                (
                    doit_blocks.bench_one_config,
                    [config, roots],
                    dict(
                        iters=OPTS["iters"],
                        bench_marker=bench_marker,
                    ),
                )
            ],
        }


# ── run (sequential -- accurate wall-clock timing) ──────────────────────────


def _discover_cinm2_configs(
    prim: str, results_dir: pathlib.Path, system: str
) -> list[compile_run.Config]:
    """Reconstruct every CINM 2.0 Config for a prim from one BO search's
    results dir -- one Config per seed, from that seed's best-in-pool config
    (pools.best_per_seed). `results_dir`/`system` select which sweep:
    cinm2_results_dir(prim)/"cinm2" for the matched sweep, or
    cinm2_unconstrained_results_dir(prim)/"cinm2_unconstrained" for the
    unconstrained one (see _cinm2_search_groups)."""
    if not results_dir.exists():
        return []
    op = prim.removeprefix("prim_")
    configs = []
    for fn_name, seed, params in pools.best_per_seed(results_dir):
        fn_module = PATHS.split_module(prim, fn_name)
        configs.append(
            compile_run.Config(
                system=system,
                fn_name=fn_name,
                label=seed,
                params=params,
                fn_module=fn_module,
                prim=op,
                lower=cinmopt.eval_solution_lowerer(),
            )
        )
    return configs


def _discover_configs(prim: str) -> list[compile_run.Config]:
    """Reconstruct every Config for a prim from what pairs/cinm2_search
    already wrote to disk (mirrors build_cinm1_configs/build_cinm2_configs
    in experiment.py, but reading state back instead of computing it) --
    CINM 1.0's matched sweep plus both of CINM 2.0's sweeps, matched and
    unconstrained."""
    op = prim.removeprefix("prim_")
    configs = []
    for fn_name in list_functions(PATHS.source_mlir(prim)):
        pairs_csv = PATHS.pairs_csv(prim, fn_name)
        if not pairs_csv.exists():
            continue
        fn_module = PATHS.split_module(prim, fn_name)
        pairs = pd.read_csv(pairs_csv)
        for dpus, tasklets in pairs[["dpus", "tasklets"]].itertuples(index=False):
            dpus, tasklets = int(dpus), int(tasklets)
            configs.append(
                compile_run.Config(
                    system="cinm1",
                    fn_name=fn_name,
                    label=f"D{dpus}_T{tasklets}",
                    params={"dpus": dpus, "tasklets": tasklets},
                    fn_module=fn_module,
                    prim=op,
                    lower=cinm1.lowerer(),
                )
            )

    configs += _discover_cinm2_configs(prim, PATHS.cinm2_results_dir(prim), "cinm2")
    configs += _discover_cinm2_configs(
        prim, PATHS.cinm2_unconstrained_results_dir(prim), "cinm2_unconstrained"
    )
    return configs


def task_bench():
    return {
        "actions": None,
        "task_dep": ["bench_cinm1", "bench_cinm2", "bench_cinm2_unconstrained"],
    }


# ── retry failed benches ────────────────────────────────────────────────────


def _retry_failed_bench() -> bool:
    """Per-prim doit_blocks.clear_failed_bench: clears the bench.done marker
    of every config that failed on hardware (compile succeeded, no
    measurable result), so the next `doit bench_cinm1 bench_cinm2` (or plain
    `doit`) retries just those; successfully benched configs are
    untouched."""
    for prim in PRIMS:
        doit_blocks.clear_failed_bench(_discover_configs(prim), PATHS.roots(prim))
    return True


def task_retry_failed_bench():
    """Not part of the default pipeline. Run explicitly (`doit
    retry_failed_bench`) after fixing whatever caused some configs to fail
    on hardware, then rerun `doit bench_cinm1 bench_cinm2` (or `doit`) to
    pick them back up."""
    return {
        "actions": [_retry_failed_bench],
        "uptodate": [False],
    }


# ── retry failed compiles ───────────────────────────────────────────────────


def _retry_failed_compiles() -> bool:
    """Per-prim doit_blocks.clear_failed_compiles: removes the compile
    output of every config recorded as failed (marker without binary), so
    the next `doit` sees a missing target and retries just those; compiled
    configs are untouched."""
    for prim in PRIMS:
        doit_blocks.clear_failed_compiles(_discover_configs(prim), PATHS.roots(prim))
    return True


def task_retry_failed_compiles():
    """Not part of the default pipeline. Run explicitly (`doit
    retry_failed_compiles`) after fixing whatever caused some configs to
    fail to compile, then rerun `doit` to pick them back up."""
    return {
        "actions": [_retry_failed_compiles],
        "uptodate": [False],
    }


# ── compare + plot ───────────────────────────────────────────────────────────


def compare(
    cinm1_results: list[compile_run.RunResult],
    cinm2_results: list[compile_run.RunResult],
) -> pd.DataFrame:
    """Merge CINM 1.0 (one point per working group) with CINM 2.0 (n_seeds
    points per working group -> seed-median) into a speedup table keyed by
    (fn_name, dpus, tasklets)."""
    cinm1 = measurements.results_to_frame(cinm1_results).drop(columns=["label"])
    cinm1 = cinm1.rename(columns={"net_time_ms": "cinm1_ms"})
    if cinm1.empty:
        # results_to_frame() only guarantees fn_name/label/net_time_ms when
        # empty -- dpus/tasklets come from cfg.params, which needs at least
        # one row to appear at all.
        cinm1["dpus"] = cinm1["tasklets"] = pd.Series(dtype=object)

    cinm2_raw = measurements.results_to_frame(cinm2_results)
    if cinm2_raw.empty:
        cinm2_summary = pd.DataFrame(
            columns=[
                "fn_name",
                "dpus",
                "tasklets",
                "cinm2_ms",
                "cinm2_ms_geomean",
                "cinm2_p25",
                "cinm2_p75",
                "cinm2_n",
            ]
        )
    else:
        cinm2_summary = (
            cinm2_raw.groupby(["fn_name", "dpus", "tasklets"])["net_time_ms"]
            .agg(
                cinm2_ms="median",
                cinm2_ms_geomean=geomean,
                cinm2_p25=lambda s: s.quantile(0.25),
                cinm2_p75=lambda s: s.quantile(0.75),
                cinm2_n="count",
            )
            .reset_index()
        )

    merged = cinm1.merge(cinm2_summary, on=["fn_name", "dpus", "tasklets"], how="inner")
    missing = set(zip(cinm1.fn_name, cinm1.dpus, cinm1.tasklets)) - set(
        zip(merged.fn_name, merged.dpus, merged.tasklets)
    )
    if missing:
        print(
            f"  WARNING: {len(missing)} (fn_name,dpus,tasklets) pairs have CINM1 "
            f"but no CINM2 data: {sorted(missing)[:5]}...",
            file=sys.stderr,
        )
    merged["speedup"] = merged["cinm1_ms"] / merged["cinm2_ms"]
    # Per-config speedup with CINM2's seed noise collapsed by geomean instead
    # of by seed-median -- the population plot_speedup_violin draws one violin
    # point per (dpus,tasklets) config from, so config-to-config spread stays
    # visible instead of also being averaged away.
    merged["speedup_seed_geomean"] = merged["cinm1_ms"] / merged["cinm2_ms_geomean"]
    return merged


def _compare_prim(prim: str) -> bool:
    configs = _discover_configs(prim)
    compiled = compile_run.discover_compiled(
        configs, compile_root=PATHS.compile_root(prim)
    )

    # Reconstruct RunResults by pointing at the output dirs bench_* already
    # wrote to -- bench.done guarantees a bench was *attempted* (never that
    # it succeeded, see _bench_one_config), so this ok=output_dir.exists()
    # only really rules out compile failures (run_config mkdir's output_dir
    # unconditionally once compile succeeds, even if the run then fails). A
    # failed run still gets filtered out, just downstream in
    # results_to_frame() via net_time_ms(...) is None.
    results = []
    for c in compiled:
        output_dir = PATHS.run_output_dir(prim, c.config)
        results.append(compile_run.RunResult(c, output_dir.exists(), output_dir))

    cinm1_results = [r for r in results if r.compiled.config.system == "cinm1"]
    cinm2_results = [r for r in results if r.compiled.config.system == "cinm2"]
    cmp_df = compare(cinm1_results, cinm2_results)
    cmp_df["prim"] = prim
    cmp_df.to_csv(PATHS.comparison_csv(prim), index=False)
    return True


# @create_after(executed="pairs")
def task_compare():
    """Geomean speedup of CINM 2.0 (seed-median) over CINM 1.0 per working
    group, aggregated per benchmark."""
    for prim in PRIMS:
        yield {
            "name": prim,
            "uptodate": [
                result_dep(f"bench_cinm1:{prim}"),
                result_dep(f"bench_cinm2:{prim}"),
            ],
            "targets": [str(PATHS.comparison_csv(prim))],
            "actions": [(_compare_prim, [prim])],
        }


def _cinm1_best_per_fn(prim: str) -> pd.DataFrame:
    """CINM 1.0's lowest net time reached anywhere in its matched-config
    sweep, per fn_name -- the steelmanned baseline comparison_best.csv uses,
    as opposed to comparison.csv's per-(dpus,tasklets) matched one."""
    configs = [c for c in _discover_configs(prim) if c.system == "cinm1"]
    compiled = compile_run.discover_compiled(
        configs, compile_root=PATHS.compile_root(prim)
    )
    results = [
        compile_run.RunResult(
            c,
            PATHS.run_output_dir(prim, c.config).exists(),
            PATHS.run_output_dir(prim, c.config),
        )
        for c in compiled
    ]
    frame = measurements.results_to_frame(results)
    return (
        frame.groupby("fn_name")["net_time_ms"]
        .agg(cinm1_best_ms="min", cinm1_n_configs="count")
        .reset_index()
    )


def _compare_best_prim(prim: str) -> bool:
    cinm1_best = _cinm1_best_per_fn(prim)

    configs = [c for c in _discover_configs(prim) if c.system == "cinm2_unconstrained"]
    compiled = compile_run.discover_compiled(
        configs, compile_root=PATHS.compile_root(prim)
    )
    results = [
        compile_run.RunResult(
            c,
            PATHS.run_output_dir(prim, c.config).exists(),
            PATHS.run_output_dir(prim, c.config),
        )
        for c in compiled
    ]
    cinm2_unc = measurements.results_to_frame(results).rename(
        columns={"net_time_ms": "cinm2_unc_ms", "label": "seed"}
    )

    merged = cinm2_unc.merge(cinm1_best, on="fn_name", how="inner")
    missing = set(cinm2_unc.fn_name) - set(merged.fn_name)
    if missing:
        print(
            f"  WARNING: {len(missing)} fn_name(s) have CINM2 unconstrained results "
            f"but no CINM1 baseline: {sorted(missing)}",
            file=sys.stderr,
        )
    merged["speedup_vs_cinm1_best"] = merged["cinm1_best_ms"] / merged["cinm2_unc_ms"]
    merged["prim"] = prim
    merged.to_csv(PATHS.comparison_best_csv(prim), index=False)
    return True


# @create_after(executed="pairs")
def task_compare_best():
    """Best-vs-best comparison, one row per (fn_name, seed): CINM 1.0's best
    time anywhere in its matched-config sweep vs CINM 2.0's unconstrained
    search (see _cinm2_search_groups in task_cinm2_search), seed indexing
    CINM 2.0's independent search runs. Not part of the default pipeline --
    depends on the unconstrained sweep's hardware benches, which a plain
    `doit` doesn't run."""
    for prim in PRIMS:
        yield {
            "name": prim,
            "uptodate": [
                result_dep(f"bench_cinm1:{prim}"),
                result_dep(f"bench_cinm2_unconstrained:{prim}"),
            ],
            "targets": [str(PATHS.comparison_best_csv(prim))],
            "actions": [(_compare_best_prim, [prim])],
        }


def _plot_all() -> bool:
    comparison = pd.concat(
        [
            pd.read_csv(PATHS.comparison_csv(prim))
            for prim in PRIMS
            if PATHS.comparison_csv(prim).exists()
        ],
        ignore_index=True,
    )
    out_dir = PATHS.plots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    print_summary(comparison)
    plot_speedup(comparison, out_dir)
    plot_speedup_violin(comparison, out_dir)
    return True


# @create_after(executed="compare")
def task_plot():
    comparison_csvs = [PATHS.comparison_csv(prim) for prim in PRIMS]
    return {
        "file_dep": [str(p) for p in comparison_csvs if p.exists()],
        "targets": [
            str(PATHS.plots_dir() / "cinm1_vs_cinm2_speedup.pdf"),
            str(PATHS.plots_dir() / "cinm1_vs_cinm2_speedup_violin.pdf"),
        ],
        "actions": [_plot_all],
    }


def _plot_best_all() -> bool:
    comparison_best = pd.concat(
        [
            pd.read_csv(PATHS.comparison_best_csv(prim))
            for prim in PRIMS
            if PATHS.comparison_best_csv(prim).exists()
        ],
        ignore_index=True,
    )
    out_dir = PATHS.plots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_best.to_csv(out_dir / "comparison_best.csv", index=False)
    plot_best_speedup(comparison_best, out_dir)
    return True


def task_plot_best():
    """Not part of the default pipeline -- run explicitly (`doit
    plot_best`), since it depends on the unconstrained CINM 2.0 sweep's
    hardware benches (see _cinm2_search_groups in task_cinm2_search), which a
    plain `doit` doesn't run."""
    comparison_best_csvs = [PATHS.comparison_best_csv(prim) for prim in PRIMS]
    return {
        "file_dep": [str(p) for p in comparison_best_csvs if p.exists()],
        "targets": [
            str(PATHS.plots_dir() / "cinm1_best_vs_cinm2_speedup_violin.pdf"),
            str(PATHS.plots_dir() / "cinm1_best_vs_cinm2_speedup_bar.pdf"),
        ],
        "actions": [_plot_best_all],
    }

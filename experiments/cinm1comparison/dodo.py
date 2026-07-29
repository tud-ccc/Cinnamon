"""doit tasks for the CINM 1.0 vs CINM 2.0 comparison experiment.

Same pipeline as experiment.py's module docstring (screen -> CINM1 configs /
CINM2 search -> compile -> run -> compare -> plot), but driven by doit
instead of a plain top-to-bottom script: each stage declares its file
inputs/outputs, so `doit` only reruns what's actually stale -- e.g. if a
CINM1 compile fails and you fix cinm1.py and rerun, the CINM2 Bayesian
search (expensive, already done) is not repeated.

Usage:
  doit list              # show all tasks (some only appear after upstream
                          # tasks that create them dynamically have run once)
  doit                   # run everything up to the plots
  doit compile_cinm1     # just compile CINM 1.0's configs
  doit forget screen     # force screening to rerun next time
  doit retry_failed_compiles && doit  # clear + retry configs that failed to compile
  doit retry_failed_bench && doit bench_cinm1 bench_cinm2  # clear + retry configs that failed on hardware

Stages are connected by files on disk, not in-memory state, since doit may
skip any stage in a given invocation: pairs.csv (screen's output) and the
CINM 2.0 search's pool.csv files are the source of truth read back by every
downstream stage.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import sys

import pandas as pd
from doit import create_after
from doit.tools import result_dep
from doit.reporter import ProgressBarReporter  # noqa: E402
# from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))


from cinm_experiments import cinm1, cinmopt, compile_run, measurements, pools  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402

from plot import (
    geomean,
    plot_best_speedup,
    plot_speedup,
    plot_speedup_violin,
    print_summary,
)  # noqa: E402

PRIMS = ["prim_gemv", "prim_red"]
DATA_DIR = HERE / "data"

OPTS = dict(
    top_frac=0.10,
    min_configs=200,
    n_seeds=32,
    iters=6,
    screen_sim="cycle-accurate",
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

    def screen_dir(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "screen"

    def pairs_csv(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.screen_dir(prim) / fn_name / "pairs.csv"

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

    def compile_root(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "compiled"

    def config_dir(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.compile_root(prim) / config.system / config.fn_name / config.label

    def compile_marker(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.config_dir(prim, config) / "compile.done"

    def bench_bin(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.config_dir(prim, config) / "bin" / f"bench_{config.fn_name}"

    def run_root(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "run"

    def run_config_dir_id(
        self, prim: str, system: str, fn_name: str, label: str
    ) -> pathlib.Path:
        return self.run_root(prim) / system / fn_name / label

    def run_config_dir(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.run_config_dir_id(prim, config.system, config.fn_name, config.label)

    def run_output_dir(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.run_config_dir(prim, config) / "output"

    def bench_marker_id(
        self, prim: str, system: str, fn_name: str, label: str
    ) -> pathlib.Path:
        """Per-config bench-attempted marker (touched whether the run
        succeeded or not -- see _bench_one_config), sibling of that config's
        output/ dir. Lets doit save bench progress config-by-config instead
        of only per-prim, so an interrupted `doit bench` resumes where it
        left off. Takes the bare (system, fn_name, label) identity rather
        than a full Config so bench_cinm1/bench_cinm2 tasks can reference
        each other's markers (to chain hardware runs into one global
        sequence, see task_compile_cinm1/task_cinm2_search) without needing
        each other's Config objects (fn_module/lower/params)."""
        return self.run_config_dir_id(prim, system, fn_name, label) / "bench.done"

    def bench_marker(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.bench_marker_id(prim, config.system, config.fn_name, config.label)

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


# ── screen ───────────────────────────────────────────────────────────────────


def _screen_one(prim: str, fn_name: str, fn_module: pathlib.Path) -> bool:
    screen_dir = PATHS.screen_dir(prim)
    cinmopt.exhaustive_search(
        fn_module,
        screen_dir,
        workers=OPTS["workers"],
        infer_opts={"use-mram-tiling": False, "simulator": OPTS["screen_sim"]},
    )

    # exhaustive_search names the dump dir after its own NameInventor
    # ("infer_" prefix); rename it to the plain function name so every
    # downstream reader can use one consistent path.
    infer_dir = screen_dir / f"infer_{fn_name}"
    fn_dir = screen_dir / fn_name
    if fn_dir.exists():
        shutil.rmtree(fn_dir)
    infer_dir.rename(fn_dir)

    pool_csv = fn_dir / "pool.csv"
    top, n_valid, n_kept = pools.select_best(
        pool_csv, top_frac=OPTS["top_frac"], min_configs=OPTS["min_configs"]
    )
    if top.empty:
        raise RuntimeError(f"no valid configs for {prim}:{fn_name}")
    pairs = (
        top[["dpus", "tasklets"]]
        .drop_duplicates()
        .sort_values(["dpus", "tasklets"])
        .reset_index(drop=True)
    )
    print(
        f"  {fn_name:20s}  kept {n_kept} of {n_valid} valid rows"
        f" -> {len(pairs)} (dpus,tasklets) pairs"
    )
    pairs.to_csv(PATHS.pairs_csv(prim, fn_name), index=False)
    return True


# @create_after(executed="split")
def task_screen():
    """Sweep CINM 2.0's configuration space with MRAM tiling disabled (CINM
    1.0's own best configs are known to lie in this constrained subspace),
    and select the (dpus, tasklets) working groups worth real hardware
    measurements. One subtask per function -- its config space (problem
    size, valid tile shapes) is its own."""
    for prim in PRIMS:
        for fn_name in list_functions(PATHS.source_mlir(prim)):
            fn_module = PATHS.split_module(prim, fn_name)
            yield {
                "name": f"{prim}:{fn_name}",
                "file_dep": [str(fn_module)],
                "targets": [str(PATHS.pairs_csv(prim, fn_name))],
                "actions": [(_screen_one, [prim, fn_name, fn_module])],
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
    one function: either dpus/tasklets pinned to a screened (dpus, tasklets)
    working group (the matched sweep, comparable 1:1 against CINM 1.0's own
    config -- see compare()), or left free (the unconstrained sweep,
    comparable against CINM 1.0's best-ever config -- see
    task_compare_best/plot_best_speedup). _cinm2_search_groups yields these;
    task_cinm2_search builds identical search/compile/bench tasks from
    either kind, since they only differ in what's in this dataclass.

    system must be "cinm2" + <the basename suffix task_cinm2_search should
    append to "cinm2_search"/"compile_cinm2"/"bench_cinm2">, e.g. "cinm2" (no
    suffix) or "cinm2_unconstrained" ("_unconstrained" suffix) -- not an
    independently-chosen tag -- because _prev_bench_task_dep reconstructs a
    predecessor config's bench basename as f"bench_{system}" purely from its
    system string (config_ids doesn't carry the suffix separately)."""

    system: str  # "cinm2" | "cinm2_unconstrained"
    task_label: str  # e.g. "D8_T4" | "unconstrained"
    extra_infer_opts: dict
    offset: int
    results_dir: pathlib.Path
    search_file_dep: pathlib.Path
    seeds: tuple[int, ...]


def _cinm2_search_groups(prim: str, fn_name: str, fn_module: pathlib.Path):
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
                search_file_dep=pairs_csv,
                seeds=tuple(gen_seeds(pair_idx)),
            )
    # Unconstrained sweep: dpus/tasklets left free, one search per function
    # instead of one per screened pair. Reuses gen_seeds/get_offset(0) as-is
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
        search_file_dep=fn_module,
        seeds=tuple(gen_seeds(0)),
    )


def _config_ids():
    """Yield (prim, system, fn_name, label, dpus, tasklets, pair_idx) for
    every config that will exist once screening has written pairs.csv, for
    every prim in PRIMS -- one flat sequence spanning ALL prims, in PRIMS
    order (flip PRIMS to reverse it), not one sequence per prim. Unlike a
    compile_run.Config's params/lower, these identities are fully known
    right after screening -- compile/run directories are keyed off (system,
    fn_name, label) alone, and CINM 2.0's dpus/tasklets are pinned to the
    pair's before its search even starts -- so this is the single place
    task_compile_cinm1 and task_cinm2_search each derive the same set of
    configs (and their bench_cinm1/bench_cinm2 tasks) from, instead of
    re-deriving it (or, for CINM 2.0, waiting on search/compile results)
    independently.

    Spanning every prim in one sequence (rather than scoping this per prim,
    as it used to) matters for _prev_bench_task_dep: each prim's benches
    must run strictly one at a time (real hardware, wall-clock timing), but
    with two independent per-prim chains -- each starting its own unchained
    i==0 -- nothing stopped doit's dispatcher from interleaving them (both
    chains' heads become ready around the same time, and each completion
    re-races its chain's next link against whatever the other chain already
    had waiting), which is exactly what happened before this was one
    sequence. The unconstrained (dpus/tasklets free) cinm2_unconstrained
    configs are appended after every matched-config cinm1/cinm2 entry for
    that same prim -- keeping them all in this one flat sequence chains the
    unconstrained sweep's hardware benches onto the tail of the matched
    sweep's for that prim, so all three systems' real-hardware runs still
    execute strictly one at a time, in prim order."""
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


def _config_index(config_ids: list[tuple]) -> dict[tuple[str, str, str, str], int]:
    return {
        (prim, system, fn_name, label): i
        for i, (prim, system, fn_name, label, *_rest) in enumerate(config_ids)
    }


def _prev_bench_task_dep(
    config_ids: list[tuple],
    index_of: dict[tuple[str, str, str, str], int],
    prim: str,
    system: str,
    fn_name: str,
    label: str,
) -> list[str]:
    """task_dep entry (or none, for the first config overall) on the
    bench_cinm1/bench_cinm2 subtask immediately before (prim, system,
    fn_name, label) in `config_ids`'s order. Chains those tasks -- which,
    unlike compile, must run strictly one at a time so concurrent hardware
    runs don't skew wall-clock timing -- into one global sequence across
    every prim and system, even though they're generated by different task
    creator functions (doit doesn't allow two creators to share a basename,
    so there's no single 'bench' task group to chain within).

    The predecessor can belong to a *different* prim than `prim` (the last
    entry of one prim's sequence chains to the first entry of the next, per
    _config_ids) -- so the returned task name uses the predecessor's own
    prim, not the `prim` argument.

    Must be task_dep, not file_dep on the predecessor's bench.done marker:
    doit's implicit file_dep -> task_dep inference (control.py
    set_implicit_deps) is computed once, right when each delayed creator's
    tasks are generated, against whatever targets are already known at that
    moment -- it does NOT retroactively wire up a file_dep against a target
    a *different*, not-yet-expanded delayed creator produces later. Naming
    the task directly resolves correctly instead (via the loader's
    delayed-placeholder machinery), regardless of which creator happens to
    run first."""
    i = index_of[(prim, system, fn_name, label)]
    if i == 0:
        return []
    prev_prim, prev_system, prev_fn_name, prev_label = config_ids[i - 1][:4]
    return [f"bench_{prev_system}:{prev_prim}:{prev_fn_name}:{prev_label}"]


@create_after(
    executed="screen",
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
    _cinm2_search_groups yields: once per screened (dpus, tasklets) working
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

    config_ids = list(_config_ids())
    index_of = _config_index(config_ids)
    for prim in PRIMS:
        op = prim.removeprefix("prim_")
        compile_root = PATHS.compile_root(prim)
        for fn_name in list_functions(PATHS.source_mlir(prim)):
            fn_module = PATHS.split_module(prim, fn_name)
            for group in _cinm2_search_groups(prim, fn_name, fn_module):
                basename_suffix = group.system.removeprefix("cinm2")
                yield {
                    "basename": "cinm2_search" + basename_suffix,
                    "name": f"{prim}:{fn_name}:{group.task_label}",
                    "file_dep": [str(group.search_file_dep)],
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
                            (_compile_best, [config, pool_csv, compile_root, marker])
                        ],
                    }

                    bench_marker = PATHS.bench_marker(prim, config)
                    yield {
                        "basename": "bench_cinm2" + basename_suffix,
                        "name": f"{prim}:{fn_name}:{seed}",
                        "file_dep": [str(marker)],
                        "task_dep": _prev_bench_task_dep(
                            config_ids, index_of, prim, group.system, fn_name, seed
                        ),
                        "targets": [str(bench_marker)],
                        "actions": [
                            (
                                _bench_one_config,
                                [config],
                                dict(
                                    compile_root=compile_root,
                                    run_root=PATHS.run_root(prim),
                                    iters=OPTS["iters"],
                                    bench_marker=bench_marker,
                                ),
                            )
                        ],
                    }


# ── compile ──────────────────────────────────────────────────────────────────


def _compile_best(
    config: compile_run.Config,
    pool_csv: pathlib.Path,
    compile_root: pathlib.Path,
    marker: pathlib.Path,
) -> bool:
    """Never raises: a config that fails to compile is recorded (printed +
    left out of the marker's sibling bin/) but must not block sibling
    configs' bench task from running -- doit treats a raised exception as a
    hard failure and skips every downstream task that depends on it, which
    is more than we want for one bad config out of many. discover_compiled()
    already treats a missing bench_* binary as a per-config failure, so
    downstream stages tolerate this fine."""
    config.params = pools.best_in_pool(pool_csv)
    if not config.params:
        return False
    return _compile_one(config, compile_root, marker)


def _compile_one(
    config: compile_run.Config, compile_root: pathlib.Path, marker: pathlib.Path
) -> bool:
    """Never raises: a config that fails to compile is recorded (printed +
    left out of the marker's sibling bin/) but must not block sibling
    configs' bench task from running -- doit treats a raised exception as a
    hard failure and skips every downstream task that depends on it, which
    is more than we want for one bad config out of many. discover_compiled()
    already treats a missing bench_* binary as a per-config failure, so
    downstream stages tolerate this fine."""
    compiled = compile_run.compile_config(config, compile_root=compile_root)
    if not compiled.ok:
        print(
            f"  FAIL compile: {config.system} {config.fn_name} {config.label}: {compiled.error}"
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


@create_after(executed="screen", creates=["bench_cinm1", "compile_cinm1"])
def task_compile_cinm1():
    """Compile CINM 1.0 once per selected working group -- no search, its
    tile sizes are inferred deterministically. Also generates that config's
    bench_cinm1 task (basename "bench_cinm1", see _bench_one_config) right
    here, so the set of CINM 1.0 configs is derived exactly once instead of
    separately for compile and bench."""
    config_ids = list(_config_ids())
    index_of = _config_index(config_ids)
    for prim, system, fn_name, label, dpus, tasklets, _ in config_ids:
        if system != "cinm1":
            continue
        op = prim.removeprefix("prim_")
        compile_root = PATHS.compile_root(prim)
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
            "file_dep": [str(PATHS.pairs_csv(prim, fn_name))],
            "targets": [str(marker)],
            "actions": [(_compile_one, [config, compile_root, marker])],
        }

        bench_marker = PATHS.bench_marker(prim, config)
        yield {
            "basename": "bench_cinm1",
            "name": f"{prim}:{fn_name}:{label}",
            "file_dep": [str(marker)],
            "task_dep": _prev_bench_task_dep(
                config_ids, index_of, prim, system, fn_name, label
            ),
            "targets": [str(bench_marker)],
            "actions": [
                (
                    _bench_one_config,
                    [config],
                    dict(
                        compile_root=compile_root,
                        run_root=PATHS.run_root(prim),
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
    """Reconstruct every Config for a prim from what screen/cinm2_search
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


def _bench_one_config(
    config: compile_run.Config,
    *,
    compile_root: pathlib.Path,
    run_root: pathlib.Path,
    iters: int,
    bench_marker: pathlib.Path,
) -> bool:
    """Never raises, like _compile_one/_compile_best: a config whose compile
    failed (compile.done marker present, no bench_* binary -- compile is
    fallible, see _compile_one) is skipped rather than treated as a hard
    doit failure, so it doesn't block sibling configs' bench tasks. The
    bench.done marker is always touched, even when the hardware run itself
    fails, so a flaky config doesn't get retried on every `doit` invocation
    -- rerun it explicitly via `doit retry_failed_bench`."""
    compiled = compile_run.discover_compiled([config], compile_root=compile_root)[0]
    if not compiled.ok:
        print(
            f"  SKIP bench (not compiled): {config.system} {config.fn_name} {config.label}"
        )
    else:
        r = compile_run.run_config(compiled, run_root=run_root, iters=iters)
        if not r.ok and compile_run.is_dpu_allocation_error(r.error):
            # retry
            r = compile_run.run_config(compiled, run_root=run_root, iters=iters)
        if not r.ok:
            print(
                f"  FAIL run: {config.system} {config.fn_name} {config.label}: {r.error[:200]}"
            )
    bench_marker.parent.mkdir(parents=True, exist_ok=True)
    bench_marker.touch()
    return True


def task_bench():
    return {
        "actions": None,
        "task_dep": ["bench_cinm1", "bench_cinm2", "bench_cinm2_unconstrained"],
    }


# ── retry failed benches ────────────────────────────────────────────────────


def _retry_failed_bench() -> bool:
    """Delete the bench.done marker of every config whose compile succeeded
    but whose run didn't leave a measurable result (see _bench_one_config
    and measurements.net_time_ms) -- i.e. every config that failed on
    hardware instead of just being un-benched yet. With the marker gone,
    the next `doit bench_cinm1 bench_cinm2` (or plain `doit`) retries just
    those configs; configs that already benched successfully are
    untouched."""
    n = 0
    for prim in PRIMS:
        for c in _discover_configs(prim):
            bin_path = PATHS.bench_bin(prim, c)
            bench_marker = PATHS.bench_marker(prim, c)
            if not (bin_path.exists() and bench_marker.exists()):
                continue
            output_dir = PATHS.run_output_dir(prim, c)
            if measurements.net_time_ms(output_dir) is None:
                print(f"  retry bench: {c.system} {c.fn_name} {c.label}")
                bench_marker.unlink()
                n += 1
    print(f"cleared {n} failed bench(es)")
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
    """Delete the compile output of every config whose compile.done marker
    exists but whose bench_* binary doesn't -- i.e. every config _compile_one
    recorded as failed (see its docstring) instead of leaving broken. With
    the marker gone, the next `doit compile_cinm1` / `compile_cinm2` (or
    plain `doit`) sees a missing target and retries just those configs;
    configs that already compiled are untouched."""
    n = 0
    for prim in PRIMS:
        for c in _discover_configs(prim):
            marker = PATHS.compile_marker(prim, c)
            bench_bin = PATHS.bench_bin(prim, c)
            if marker.exists() and not bench_bin.exists():
                print(f"  retry: {c.system} {c.fn_name} {c.label}")
                shutil.rmtree(marker.parent)
                n += 1
    print(f"cleared {n} failed compile(s)")
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


# @create_after(executed="screen")
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


# @create_after(executed="screen")
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

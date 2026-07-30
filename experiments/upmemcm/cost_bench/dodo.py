"""doit tasks for cost_bench: benchmark a random sample of valid configs, for
one or more primitives (red, gemv, ...), to validate the cost model (measured
vs predicted -- see plot_cost.py's cost_calibration plot).

search -> split -> compile -> bench -> agg -> plot. The search stage
(task_search) generates each prim's own oracle pool via
cinmopt.random_sample -- evaluating only ORACLE_SAMPLE_N random valid configs
per function instead of every valid config in the space (which can run into
the hundreds of thousands, e.g. ~230k for gemv, and take hours to evaluate
exhaustively) -- so the whole pipeline is self-contained: no oracle needs to
be produced separately beforehand (unlike cinm1comparison's BO-search
comparison, which still has no search stage of its own).

Adding a new benchmark later = one more PRIMS entry (source mlir + config
filter) plus an experiments/bench/<prim>.cpp driver; no other code here needs
to change.

doit tasks are one per (prim, function), not one per config row: even a
sampled oracle pool degenerates back toward the old exhaustive-search scale
if ORACLE_SAMPLE_N is pushed up, and doit's own task-graph bookkeeping (not
the actual compile/bench work) becomes the bottleneck well before that many
doit tasks -- confirmed in practice, both at million-row scale (`doit list`
alone took over a minute) and, after tightening the gemv filter down to
~209k configs total, at that scale too. So each (prim, function) task
internally loops over every filtered config for that function, but --
unlike a plain compile_run.compile_configs/run_configs call -- skips
configs that already have a successful result on disk (see
_compile_fn/_bench_fn), so doit's own per-function tracking plus this inner
skip together give the practical effect of per-config tracking without doit
ever seeing more than a handful of tasks.

Usage:
  doit list                       # show all tasks
  doit                            # search -> split -> compile -> bench -> agg -> plot (default)
  doit single:red:red_4MB         # the whole pipeline, but only for red_4MB
  doit search:red                 # (re)generate just red's oracle pool (no hardware needed)
  doit compile:red:red_4MB        # just compile red_4MB's configs (no hardware needed)
  doit cost:red:red_4MB           # just predict costs for red_4MB's configs (cheaper than
                                   # compile -- no DPU/host compile, no hardware needed)
  doit forget bench:red:red_4MB   # force that function's hardware runs to redo
                                   # (only outstanding/failed configs actually rerun --
                                   # see _bench_one)
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import sys
import os
import random

from doit.reporter import ProgressBarReporter
from doit.tools import PythonInteractiveAction

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from cinm_experiments import (
    compile_run,
    cinmopt,
    aggregate,
    pools,
    measurements,
    failures,
)
from cinm_experiments.split_source import list_functions, split_source
from cinm_experiments.paths import python_bin

DOIT_CONFIG = {
    "default_tasks": ["plot"],
    "verbosity": 2,
    "continue": True,
    "reporter": ProgressBarReporter,
}

ITERS = 5
SYSTEM = "cinm2"  # single fixed system tag -- this pipeline doesn't compare systems

# How many filtered pool rows actually get compiled+benched per function.
SAMPLE_N = 512
# Increase this if the filter throws things away
ORACLE_SAMPLE_N = SAMPLE_N  # * 4


def _red_filter(p: dict) -> bool:
    return True
    # return p["mramCol"] * p["dpus"] >= 64 * 1024 and p["dpus"] <= 512


def _gemv_filter(p: dict) -> bool:
    return True
    # return (
    #     p["M"] == (p["dpus"] / p["dpuCols"]) * p["mramRow"]
    #     and p["K"] == p["dpuCols"] * p["mramCol"]
    #     and p["dpus"] >= 32
    #     and p["wramCol"] > 16
    # )


@dataclasses.dataclass(frozen=True)
class Prim:
    name: str
    source_mlir: pathlib.Path
    config_filter: callable
    dimensions: dict[str, dict]


PRIMS: dict[str, Prim] = {
    "red": Prim(
        name="red",
        source_mlir=EXPERIMENTS_DIR / "prim_red.mlir",
        config_filter=_red_filter,
        dimensions={
            "4MB": dict(K=524288),
            "64MB": dict(K=8388608),
            "256MB": dict(K=34554432),
            "512MB": dict(K=67108864),
        },
    ),
    "gemv": Prim(
        name="gemv",
        source_mlir=EXPERIMENTS_DIR / "prim_gemv.mlir",
        config_filter=_gemv_filter,
        dimensions={
            "4MB": dict(M=1024, K=1024),
            "64MB": dict(M=4096, K=4096),
            "256MB": dict(M=8192, K=8192),
            "512MB": dict(M=8192, K=16394),
        },
    ),
}

DATA_DIR = HERE / "data"


@dataclasses.dataclass(frozen=True)
class Paths:
    data_dir: pathlib.Path

    def split_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "_split"

    def split_module(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.split_dir(prim) / f"{fn_name}.mlir"

    def compile_root(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "compiled"

    def run_root(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "run"

    # Function-level markers (not per-config) -- flat files next to, not
    # inside, compile_root/run_root's system/fn_name/label config dirs, so
    # they can never collide with a config's own label ("row_00007", ...).
    def compile_marker(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.compile_root(prim) / f"{fn_name}.compile.done"

    def cost_marker(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.compile_root(prim) / f"{fn_name}.cost.done"

    def bench_marker(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.run_root(prim) / f"{fn_name}.bench.done"

    # Same flat-file-next-to-the-fn-dir convention as the markers above --
    # written right after _compile_fn/_bench_fn so a post-mortem doesn't
    # depend on having scrolled back through doit's stdout, or on having run
    # the separate (whole-prim) `failures` task afterwards.
    def compile_failures_csv(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.compile_root(prim) / f"{fn_name}.compile_failures.csv"

    def run_failures_csv(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.run_root(prim) / f"{fn_name}.run_failures.csv"

    def bench_bin(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return (
            self.compile_root(prim)
            / config.system
            / config.fn_name
            / config.label
            / "bin"
            / f"bench_{config.fn_name}"
        )

    def cost_csv(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return (
            self.compile_root(prim)
            / config.system
            / config.fn_name
            / config.label
            / "ir"
            / "cost.csv"
        )

    def output_dir(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return (
            self.run_root(prim)
            / config.system
            / config.fn_name
            / config.label
            / "output"
        )

    def agg_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "aggregated"

    def predicted_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "predicted"

    def failures_csv(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "failures.csv"

    def plots_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "plots"

    def pool_measured_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "pool_measured"

    def oracle_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "oracle"


PATHS = Paths(DATA_DIR)


# ── search (generate the oracle) ────────────────────────────────────────────
#
# The oracle pool used to be assumed pre-existing (produced separately by a
# multi-hour cinmopt.exhaustive_search run, or handed to us as a fixed
# external dir -- see git history). cinmopt.random_sample makes generating it
# fast enough to fold into this pipeline directly: one doit subtask per
# (prim, function), each running its own cinm-opt invocation against that
# function's already-split module (task_split's output, used as file_dep here
# the same way task_compile uses it) -- so re-splitting or editing one
# function's source only reruns that function's search, not the whole prim's.


def _search_one(prim_name: str, fn_name: str) -> bool:
    out_dir = PATHS.oracle_dir(prim_name)
    print(
        f"  {prim_name}:{fn_name}: sampling {ORACLE_SAMPLE_N} random valid "
        f"configs -> {out_dir}"
    )
    cinmopt.random_sample(
        PATHS.split_module(prim_name, fn_name),
        out_dir,
        infer_opts={
            "simulator": "hybrid",
            "eval-timeout-ms": "400",
        },
        n_samples=ORACLE_SAMPLE_N,
        workers=os.cpu_count(),
        nice=True,
        seed=0,
    )
    return True


def task_search():
    """(Re)generate one function's oracle pool -- see module docstring
    above. One doit subtask per (prim, function), depending (via file_dep,
    same as task_compile) on that function's split module rather than the
    prim's whole unsplit source_mlir."""
    for name in PRIMS:
        for fn_name in _fn_names(name):
            yield {
                "name": f"{name}:{fn_name}",
                "file_dep": [str(PATHS.split_module(name, fn_name))],
                "targets": [
                    str(PATHS.oracle_dir(name) / f"infer_{fn_name}" / "pool.csv")
                ],
                "actions": [PythonInteractiveAction(_search_one, [name, fn_name])],
            }


# ── config discovery ─────────────────────────────────────────────────────────


def _fn_names(prim_name: str) -> list[str]:
    """Every function declared in this prim's source_mlir -- a cheap static
    parse (list_functions), deliberately not derived from the oracle pool
    (unlike _fn_configs below): task_compile/task_bench/etc. call this while
    doit is still just building its task list, before any task has actually
    run search:<prim> yet, so it must work even before the oracle exists."""
    return list_functions(PRIMS[prim_name].source_mlir)


@functools.cache
def _fn_configs(prim_name: str, fn_name: str) -> tuple[compile_run.Config, ...]:
    """Every compile_run.Config to benchmark for one (prim, function): one
    per valid pool row surviving that prim's config_filter. label = the
    row's index in pool.csv (stable identity for joining measured results
    back to the pool later, see plot_cost.py).

    Cached per (prim, fn_name), since task_compile/task_bench each need it
    and oracle pools run into hundreds of thousands of rows (gemv). Uses
    itertuples(), not iterrows() -- ~15x faster on a 230k-row pool, since
    iterrows() boxes every row into a Series.
    """
    prim = PRIMS[prim_name]
    pool_csv = PATHS.oracle_dir(prim_name) / f"infer_{fn_name}" / "pool.csv"
    problem_dims = fn_name.removeprefix(prim.name + "_")
    fn_module = PATHS.split_module(prim_name, fn_name)
    df = pools.load_valid(pool_csv)
    cols = pools.param_cols(df)
    configs = []
    for row in df[cols].itertuples(index=True, name=None):
        idx, values = row[0], row[1:]
        params = dict(zip(cols, (int(v) for v in values)))
        # The filter also has access to the problem dimensions
        full_parms = params | prim.dimensions[problem_dims]
        if not prim.config_filter(full_parms):
            continue
        configs.append(
            compile_run.Config(
                system=SYSTEM,
                fn_name=fn_name,
                label=f"row_{idx:05d}",
                params=params,
                fn_module=fn_module,
                prim=prim_name,
                lower=cinmopt.eval_solution_lowerer(),
            )
        )
    random.seed(0)
    return tuple(random.sample(configs, k=min(SAMPLE_N, len(configs))))


# ── split ────────────────────────────────────────────────────────────────────


def _split_one(source_mlir: pathlib.Path, split_dir: pathlib.Path) -> bool:
    split_source(
        source_mlir, split_dir
    )  # dict return value isn't JSON-picklable for doit's DB
    return True


def task_split():
    """Split each prim's source into one module per function."""
    for name, prim in PRIMS.items():
        split_dir = PATHS.split_dir(name)
        fns = list_functions(prim.source_mlir)
        yield {
            "name": name,
            "file_dep": [str(prim.source_mlir)],
            "targets": [str(split_dir / f"{fn}.mlir") for fn in fns],
            "actions": [(_split_one, [prim.source_mlir, split_dir])],
        }


# ── compile ──────────────────────────────────────────────────────────────────


def _compile_fn(prim_name: str, fn_name: str, marker: pathlib.Path) -> bool:
    """Compile every filtered pool row for one function, in parallel
    (compile_run.compile_configs) -- except configs that already have a
    compiled binary on disk, which are skipped up front so re-running this
    (e.g. after `doit forget compile:<prim>:<fn>`, or resuming an
    interrupted previous run) only (re)compiles what's outstanding or
    previously failed instead of recompiling the whole function's sweep. A
    config that fails to compile is printed and skipped (never raises), it
    doesn't block its siblings."""
    configs = list(_fn_configs(prim_name, fn_name))
    pending = [c for c in configs if not PATHS.bench_bin(prim_name, c).exists()]
    print(
        f"  {prim_name}:{fn_name}: {len(pending)}/{len(configs)} configs need compiling"
    )
    compile_run.compile_configs(
        pending,
        compile_root=PATHS.compile_root(prim_name),
        workers=os.cpu_count(),
        label=fn_name,
    )
    fn_failures = failures.collect_compile_failures(
        PATHS.compile_root(prim_name) / SYSTEM, only_fn=fn_name
    )
    # Written unconditionally (even when empty) so the file's mere presence
    # means "checked as of this compile run", not just "had failures".
    fn_failures.to_csv(PATHS.compile_failures_csv(prim_name, fn_name), index=False)
    if not fn_failures.empty:
        print(f"  {prim_name}:{fn_name}: {len(fn_failures)} compile failures")
        print(failures.summarize(fn_failures))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def task_compile():
    """One doit subtask per (prim, function) -- see module docstring for why
    not per config row. Compiling needs no hardware."""
    for name in PRIMS:
        for fn_name in _fn_names(name):
            marker = PATHS.compile_marker(name, fn_name)
            yield {
                "name": f"{name}:{fn_name}",
                "task_dep": [f"search:{name}:{fn_name}"],
                "file_dep": [str(PATHS.split_module(name, fn_name))],
                "targets": [str(marker)],
                # PythonInteractiveAction, not a plain (callable, args) tuple:
                # a normal PythonAction captures stdout into a StringIO (see
                # doit/action.py's PythonAction.execute), which is what
                # mangles compile_configs' own tqdm bar through the
                # ProgressBarReporter -- Interactive actions never capture.
                "actions": [
                    PythonInteractiveAction(_compile_fn, [name, fn_name, marker])
                ],
            }


# ── cost (predicted costs only -- no DPU/host compile, no hardware) ────────
#
# Standalone alternative to task_compile for when only the cost model's
# predicted breakdown is needed: compute_costs runs just cinm-opt's
# --upmem-annotate-costs pass (Makefile's costs-only target) per config,
# skipping DPU kernel compilation and host object/link entirely. Writes into
# the same compile_root layout task_compile does (config.csv + ir/cost.csv,
# no bin/), so aggregate_predicted_costs/task_agg_predictions pick up
# whichever configs have a cost.csv on disk regardless of which of these two
# tasks produced it. Not wired into task_agg_predictions/task_plot/
# task_single's task_dep chain -- those still depend on compile (which
# already produces ir/cost.csv as a side effect of bench-single, so nothing
# extra would be gained there); this is for getting cost predictions for a
# sweep on its own, without paying for full compile+link.


def _cost_fn(prim_name: str, fn_name: str, marker: pathlib.Path) -> bool:
    """Predict costs for every filtered pool row for one function, in
    parallel (compile_run.compute_costs) -- except configs that already
    have a cost.csv on disk, which are skipped up front the same way
    _compile_fn skips already-compiled configs."""
    configs = list(_fn_configs(prim_name, fn_name))
    pending = [c for c in configs if not PATHS.cost_csv(prim_name, c).exists()]
    print(
        f"  {prim_name}:{fn_name}: {len(pending)}/{len(configs)} configs need cost prediction"
    )
    compile_run.compute_costs(
        pending,
        compile_root=PATHS.compile_root(prim_name),
        workers=os.cpu_count(),
        label=fn_name,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def task_cost():
    """`doit cost:<prim>:<fn>` -- one doit subtask per (prim, function), like
    task_compile. Needs no hardware."""
    for name in PRIMS:
        for fn_name in _fn_names(name):
            marker = PATHS.cost_marker(name, fn_name)
            yield {
                "name": f"{name}:{fn_name}",
                "task_dep": [f"search:{name}:{fn_name}"],
                "file_dep": [str(PATHS.split_module(name, fn_name))],
                "targets": [str(marker)],
                # PythonInteractiveAction -- see task_compile's comment.
                "actions": [PythonInteractiveAction(_cost_fn, [name, fn_name, marker])],
            }


# ── bench (sequential -- accurate wall-clock timing) ────────────────────────


def _bench_fn(prim_name: str, fn_name: str, marker: pathlib.Path) -> bool:
    """Benchmark every filtered pool row for one function, sequentially
    (compile_run.run_configs -- real hardware, wall-clock timing; concurrent
    hardware runs would contend for host/DPU resources and skew it).

    Configs that already have a measurable result are skipped up front, so
    re-running this (e.g. after `doit forget bench:<prim>:<fn>`, or resuming
    an interrupted previous run) only (re)does what's outstanding or
    previously failed instead of re-benching the whole function's sweep."""
    configs = list(_fn_configs(prim_name, fn_name))
    compiled = compile_run.discover_compiled(
        configs, compile_root=PATHS.compile_root(prim_name)
    )
    # --- Subsample for a quicker run. TODO remove this
    random.seed(0)
    compiled = random.sample(compiled, k=256)
    # ---
    pending = [
        c
        for c in compiled
        if c.ok
        and measurements.net_time_ms(PATHS.output_dir(prim_name, c.config)) is None
    ]
    print(
        f"  {prim_name}:{fn_name}: {len(pending)}/{len(compiled)} configs need a bench run"
    )
    compile_run.run_configs(pending, run_root=PATHS.run_root(prim_name), iters=ITERS)
    fn_failures = failures.collect_run_failures(
        PATHS.run_root(prim_name) / SYSTEM,
        PATHS.compile_root(prim_name) / SYSTEM,
        only_fn=fn_name,
    )
    fn_failures.to_csv(PATHS.run_failures_csv(prim_name, fn_name), index=False)
    if not fn_failures.empty:
        print(f"  {prim_name}:{fn_name}: {len(fn_failures)} run failures")
        print(failures.summarize(fn_failures))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def task_bench():
    """One doit subtask per (prim, function) -- see module docstring. Runs
    within _bench_fn are strictly sequential; across (prim, function) pairs,
    doit's default same-priority yield-order keeps every hardware run
    overall sequential too (PRIMS' fixed dict order), matching every other
    dodo.py in this repo -- no extra task_dep chaining needed since there's
    only one task-generator function producing these, unlike
    cinm1comparison's cross-creator chaining."""
    for name in PRIMS:
        for fn_name in _fn_names(name):
            marker = PATHS.bench_marker(name, fn_name)
            yield {
                "name": f"{name}:{fn_name}",
                "file_dep": [str(PATHS.compile_marker(name, fn_name))],
                "targets": [str(marker)],
                # see task_compile's comment -- run_configs has its own tqdm
                # bar too.
                "actions": [
                    PythonInteractiveAction(_bench_fn, [name, fn_name, marker])
                ],
            }


# ── aggregate ────────────────────────────────────────────────────────────────


def _agg_one(prim_name: str) -> bool:
    # aggregate.aggregate_run expects run_dir/fn_name/config_dir/output (2
    # levels); this pipeline only ever uses one `system` tag, so pointing it
    # at .../{system}/ lines the depth up with no changes needed there.
    run_dir = PATHS.run_root(prim_name) / SYSTEM
    compile_dir = PATHS.compile_root(prim_name) / SYSTEM
    aggregate.aggregate_run(run_dir, compile_dir, PATHS.agg_dir(prim_name))
    return True


def task_agg():
    """Merge every config's raw per-iteration output CSVs into one CSV per
    measurement type. Cheap (pandas concat over what bench already wrote to
    disk), so unlike compile/bench it just always reruns rather than tracking
    which of its dynamically-named output files are stale."""
    for name in PRIMS:
        yield {
            "name": name,
            "task_dep": [f"bench:{name}:{fn}" for fn in _fn_names(name)],
            "uptodate": [False],
            "actions": [(_agg_one, [name])],
        }


def _agg_predictions_one(prim_name: str) -> bool:
    compile_dir = PATHS.compile_root(prim_name) / SYSTEM
    aggregate.aggregate_predicted_costs(compile_dir, PATHS.predicted_dir(prim_name))
    return True


def task_agg_predictions():
    """Merge every config's ir/cost.csv (the cost-model's per-op category
    breakdown, written at compile time -- see aggregate.
    aggregate_predicted_costs) into one predicted_costs.csv. Depends only on
    compile (not bench), since the breakdown is a compile-time artifact that
    doesn't need hardware. Like task_agg, always reruns rather than tracking
    staleness."""
    for name in PRIMS:
        yield {
            "name": name,
            "task_dep": [f"compile:{name}:{fn}" for fn in _fn_names(name)],
            "uptodate": [False],
            "actions": [(_agg_predictions_one, [name])],
        }


def _failures_one(prim_name: str) -> bool:
    df = failures.collect_failures(
        PATHS.run_root(prim_name) / SYSTEM, PATHS.compile_root(prim_name) / SYSTEM
    )
    df.to_csv(PATHS.failures_csv(prim_name), index=False)
    print(f"  {prim_name}: {len(df)} failures -> {PATHS.failures_csv(prim_name)}")
    print(failures.summarize(df))
    return True


def task_failures():
    """Collect every compile/run failure left on disk across this prim's
    functions into one failures.csv (see cinm_experiments.failures) --
    depends on bench (not just compile), since run failures need bench to
    have been attempted too. Like task_agg, always reruns."""
    for name in PRIMS:
        yield {
            "name": name,
            "task_dep": [f"bench:{name}:{fn}" for fn in _fn_names(name)],
            "uptodate": [False],
            "actions": [(_failures_one, [name])],
        }


# ── plot ─────────────────────────────────────────────────────────────────────


def _plot_cmd(prim_name: str) -> str:
    """The plot_cost.py invocation for one prim -- factored out so
    task_single can run the exact same command directly (see its docstring
    for why), without duplicating the CLI string by hand."""
    return (
        f"{python_bin()} plot_cost.py "
        f"--in-dir {PATHS.agg_dir(prim_name)} --out-dir {PATHS.plots_dir(prim_name)} "
        f"--oracle {PATHS.oracle_dir(prim_name)} "
        f"--pool-out-dir {PATHS.pool_measured_dir(prim_name)} "
        f"--predicted-dir {PATHS.predicted_dir(prim_name)} "
        f"--failures-csv {PATHS.failures_csv(prim_name)}"
    )


def task_plot():
    """Regenerate this prim's plots/ (per-metric-vs-dim scatter plots, plus
    the cost_calibration measured-vs-predicted plot). Shells out to
    plot_cost.py (a standalone CLI script, also runnable on its own) rather
    than importing it, so its own --filter/--oracle CLI stays the single
    source of truth for what it does. Like task_agg, always reruns -- cheap
    relative to compile/bench, and its output set is dynamic (one plot per
    function/metric found)."""
    for name in PRIMS:
        yield {
            "name": name,
            "task_dep": [f"agg:{name}", f"agg_predictions:{name}", f"failures:{name}"],
            "uptodate": [False],
            "actions": [_plot_cmd(name)],
        }


# ── single (whole pipeline for one function) ────────────────────────────────


def task_single():
    """`doit single:<prim>:<fn>` -- the whole pipeline (search -> split ->
    compile -> bench -> agg -> agg_predictions -> failures -> plot) for one
    function.

    task_dep only covers search/compile/bench for exactly this function --
    NOT agg:<prim>/agg_predictions:<prim>/failures:<prim>/plot:<prim>
    themselves, since those task_dep on *every* function of the prim (see
    their own docstrings), which would silently force search+compile+bench
    of every sibling function too, defeating the point of "single". Instead
    the aggregate+plot step is called directly as this task's own actions --
    the exact same functions/command task_agg/task_agg_predictions/
    task_failures/task_plot use, just invoked here rather than depended on --
    which is safe because all four are already unconditional/"uptodate:
    False" (cheap, always-rerun-from-whatever's-on-disk), not incremental.
    One consequence worth knowing: that rerun still aggregates/plots every
    function of the prim that happens to already have data on disk, not
    narrowly just this one -- plot_cost.py has no per-function scope today."""
    for name in PRIMS:
        for fn_name in _fn_names(name):
            yield {
                "name": f"{name}:{fn_name}",
                "task_dep": [
                    f"search:{name}:{fn_name}",
                    f"compile:{name}:{fn_name}",
                    f"bench:{name}:{fn_name}",
                ],
                "uptodate": [False],
                "actions": [
                    (_agg_one, [name]),
                    (_agg_predictions_one, [name]),
                    (_failures_one, [name]),
                    _plot_cmd(name),
                ],
            }

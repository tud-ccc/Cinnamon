"""doit tasks for the paperplots experiment: for each (prim, BO-search
target) source, run CINM 2.0's Bayesian search (or CINM 1.0's tile-size
inference, itself modeled below as a "target" with MRAM tiling disabled)
with a large seed count, compile+benchmark every seed's best config on real
UPMEM hardware, and plot net-time distributions/speedups across sources.
Same doit-over-Makefile rationale as cinm1comparison/dodo.py.

PREFIX plays the same role the old Makefile's PREFIX variable did: it
namespaces a whole run of this pipeline under data/{PREFIX}/ so different BO
hyperparameter choices don't clobber each other's data. It is a constant
here, not something doit generates tasks for -- change it and rerun to
start a new namespaced run; old ones are left untouched on disk.

IMPORTANT -- this pipeline's most expensive stages (BO search: CPU-days
across all 14 sources; compile+run: real UPMEM hardware time) were
previously driven by this directory's Makefile (see git history), and most
of that data already exists on disk under data/. doit has no record of
those runs -- they predate doit -- so on this dodo.py's FIRST use, run:

    doit reset-dep

This recomputes doit's file-dependency bookkeeping from what's already on
disk WITHOUT re-executing anything, so already-completed sources are
recognized as up-to-date instead of doit redoing them from scratch (see
`doit help reset-dep`). The one exception is the bench task: its target is a
new per-source marker that doesn't exist in the legacy layout, so it will
run once per already-completed source even after reset-dep -- but its
action skips any config that already has benchmark output on disk, so that
first pass is cheap, not a real re-benchmark (see _run_source).

Usage:
  doit list                     # show all tasks
  doit                          # run everything up to the plots
  doit bo:prim_gemv_cinm2_ca    # just that source's BO search
  doit retry_failed_compiles && doit  # clear + retry configs that failed to compile
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import sys

from doit import create_after

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from cinm_experiments import aggregate, cinmopt, compile_run, parallel, pools  # noqa: E402
from cinm_experiments.paths import DEFAULT_CINM_OPT  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402

from plot_best_configs import run_plots  # noqa: E402

PRIMS = ["prim_gemv", "prim_red"]

# Same BO "target" -> extra --upmem-infer-accelerator options as the old
# Makefile's per-target BO_RULE invocations. Dict order is preserved as the
# order sources are plotted in (matches the old Makefile's TARGETS var).
TARGET_INFER_OPTS = {
    "cinm2_ca400":     {"simulator": "cycle-accurate", "eval-timeout-ms": 400},
    "cinm2_fast":      {"simulator": "fast", "eval-timeout-ms": 60000},
    "cinm2_hybrid200": {"simulator": "hybrid", "eval-timeout-ms": 200},
    "cinm2_hybrid400": {"simulator": "hybrid", "eval-timeout-ms": 400},
    "cinm1":           {"simulator": "hybrid", "eval-timeout-ms": 400, "use-mram-tiling": False},
    "cinm2_ca":        {"simulator": "cycle-accurate", "eval-timeout-ms": 60000},
    "cinm2_ca200":     {"simulator": "cycle-accurate", "eval-timeout-ms": 200},
}
TARGETS = list(TARGET_INFER_OPTS)
SOURCES = [f"{prim}_{target}" for prim in PRIMS for target in TARGETS]

PREFIX = "prim_largesurrogate"
CINM_OPT = pathlib.Path(os.environ.get("CINM_OPT", DEFAULT_CINM_OPT))

BASE_BO_INFER_OPTS = {
    "n-validation": 600, "validation-interval": 1, "max-evals": 120,
    "dump-full-pool": False, "hidden-depth": 6, "epochs": 1810, "hidden-width": 32,
}
N_SEEDS = 128
BO_OFFSET = 67  # fixed: every source has its own results dir, so seeds never collide across sources
WORKERS = 64
ITERS = 10
DPU_CAP = 1024

DOIT_CONFIG = {"default_tasks": ["plot"], "verbosity": 2, "continue": True}


@dataclasses.dataclass(frozen=True)
class Paths:
    """Every on-disk path this pipeline reads or writes, keyed by `source`
    ("{prim}_{target}", e.g. "prim_gemv_cinm2_ca") and/or a compile_run.Config.
    Matches the legacy Makefile-produced layout exactly (config_dir has no
    extra "system" nesting -- see task_compile) so already-completed sources
    are recognized by `doit reset-dep` instead of doit wanting to redo them."""
    experiments_dir: pathlib.Path
    data_dir: pathlib.Path  # data/{PREFIX}

    def source_mlir(self, prim: str) -> pathlib.Path:
        return self.experiments_dir / f"{prim}.mlir"

    def source_dir(self, source: str) -> pathlib.Path:
        return self.data_dir / source

    def results_dir(self, source: str) -> pathlib.Path:
        return self.source_dir(source) / "results"

    def bo_out(self, source: str) -> pathlib.Path:
        return self.results_dir(source) / "out.mlir"

    def compile_root(self, source: str) -> pathlib.Path:
        return self.source_dir(source) / "compiled"

    def split_dir(self, source: str) -> pathlib.Path:
        return self.compile_root(source) / "_split"

    def config_dir(self, source: str, config: compile_run.Config) -> pathlib.Path:
        return self.compile_root(source) / config.fn_name / config.label

    def config_csv(self, source: str, config: compile_run.Config) -> pathlib.Path:
        return self.config_dir(source, config) / "config.csv"

    def bench_bin(self, source: str, config: compile_run.Config) -> pathlib.Path:
        return self.config_dir(source, config) / "bin" / f"bench_{config.fn_name}"

    def run_root(self, source: str) -> pathlib.Path:
        return self.source_dir(source) / "run"

    def run_output_dir(self, source: str, config: compile_run.Config) -> pathlib.Path:
        return self.run_root(source) / config.fn_name / config.label / "output"

    def aggregated_dir(self, source: str) -> pathlib.Path:
        return self.source_dir(source) / "aggregated"

    def bo_timings_dir(self, source: str) -> pathlib.Path:
        return self.source_dir(source) / "bo_timings"

    def run_done(self, source: str) -> pathlib.Path:
        return self.source_dir(source) / "run.done"

    def plots_dir(self) -> pathlib.Path:
        return self.data_dir / "plots"


PATHS = Paths(EXPERIMENTS_DIR, HERE / "data" / PREFIX)


def _discover_configs(source: str, op: str) -> list[compile_run.Config]:
    """Reconstruct every Config for `source` from its BO results (mirrors
    cinm1comparison/dodo.py's _discover_configs: read state back from disk
    instead of recomputing it, so compile/run/aggregate all agree on the
    same config list). system="" so config_dir lands on
    compiled/{fn_name}/{label}/ with no extra nesting, matching the legacy
    layout (compile_run.Config always inserts a system/ level; Path("x") /
    "" / "y" == Path("x") / "y", so an empty system is the way to opt out)."""
    results_dir = PATHS.results_dir(source)
    if not results_dir.exists():
        return []
    split_dir = PATHS.split_dir(source)
    return [
        compile_run.Config(
            system="", fn_name=fn_name, label=f"seed_{seed}", params=params,
            fn_module=split_dir / f"{fn_name}.mlir", prim=op,
            lower=cinmopt.eval_solution_lowerer(params, cinm_opt=CINM_OPT),
        )
        for fn_name, seed, params in pools.best_per_seed(results_dir)
    ]


# ── BO search ────────────────────────────────────────────────────────────────

def _bo_one(prim_mlir: pathlib.Path, results_dir: pathlib.Path, infer_opts: dict) -> bool:
    cinmopt.bo_multiseed(
        prim_mlir, results_dir, n_seeds=N_SEEDS, offset=BO_OFFSET, workers=WORKERS,
        infer_opts={**BASE_BO_INFER_OPTS, **infer_opts}, nice=True, cinm_opt=CINM_OPT,
    )
    return True


def task_bo():
    """Run CINM 2.0's Bayesian search (n_seeds=128) on the whole prim module
    at once -- cinm-opt's --split-input-file handles per-function splitting
    internally here (unlike compile below, this doesn't need one config per
    function pinned; every function in the module gets its own
    {fn_name}/seed_{k}/ dump under one results dir)."""
    for prim in PRIMS:
        prim_mlir = PATHS.source_mlir(prim)
        for target in TARGETS:
            source = f"{prim}_{target}"
            yield {
                "name": source,
                "file_dep": [str(prim_mlir)],
                "targets": [str(PATHS.bo_out(source))],
                "actions": [(_bo_one, [prim_mlir, PATHS.results_dir(source), TARGET_INFER_OPTS[target]])],
            }


# ── split (compile needs one function per module; bo above doesn't) ────────

def _split_one(prim_mlir: pathlib.Path, split_dir: pathlib.Path) -> bool:
    split_source(prim_mlir, split_dir)
    return True


def task_split():
    """Split each source's prim module into one file per function. One
    subtask per source (not just per prim) to match the legacy
    compiled/_split/ layout, even though the split content is identical for
    every target of a given prim."""
    for prim in PRIMS:
        prim_mlir = PATHS.source_mlir(prim)
        fns = list_functions(prim_mlir)
        for target in TARGETS:
            source = f"{prim}_{target}"
            split_dir = PATHS.split_dir(source)
            yield {
                "name": source,
                "file_dep": [str(prim_mlir)],
                "targets": [str(split_dir / f"{fn}.mlir") for fn in fns],
                "actions": [(_split_one, [prim_mlir, split_dir])],
            }


# ── compile ──────────────────────────────────────────────────────────────────

def _compile_one(config: compile_run.Config, compile_root: pathlib.Path) -> bool:
    """Never raises: compile_config() already catches subprocess failures
    and always writes config.csv (this action's doit target) before
    attempting the failure-prone lower/make steps, so one config failing to
    compile can't block sibling configs' bench task the way a raised
    exception (blocking every downstream file_dep on this task) would."""
    compiled = compile_run.compile_config(config, compile_root=compile_root)
    if not compiled.ok:
        print(f"  FAIL compile: {config.fn_name} {config.label}: {compiled.error}")
    return True


@create_after(executed="bo")
def task_compile():
    """Compile every seed's best (lowest-cost visited) config for real, one
    subtask per (source, fn_name, seed)."""
    for prim in PRIMS:
        op = prim.removeprefix("prim_")
        for target in TARGETS:
            source = f"{prim}_{target}"
            compile_root = PATHS.compile_root(source)
            results_dir = PATHS.results_dir(source)
            for config in _discover_configs(source, op):
                pool_csv = results_dir / f"infer_{config.fn_name}" / config.label / "pool.csv"
                yield {
                    "name": f"{source}:{config.fn_name}:{config.label}",
                    "file_dep": [str(pool_csv), str(config.fn_module)],
                    "targets": [str(PATHS.config_csv(source, config))],
                    "actions": [(_compile_one, [config, compile_root])],
                }


def _retry_failed_compiles() -> bool:
    """Delete the compile output of every config whose config.csv exists but
    whose bench_* binary doesn't -- i.e. every config _compile_one recorded
    as failed. With config.csv gone, the next `doit compile` (or plain
    `doit`) sees a missing target and retries just those configs; configs
    that already compiled are untouched."""
    n = 0
    for prim in PRIMS:
        op = prim.removeprefix("prim_")
        for target in TARGETS:
            source = f"{prim}_{target}"
            for config in _discover_configs(source, op):
                config_dir = PATHS.config_dir(source, config)
                if (config_dir / "config.csv").exists() and not PATHS.bench_bin(source, config).exists():
                    print(f"  retry: {source} {config.fn_name} {config.label}")
                    shutil.rmtree(config_dir)
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


# ── run (DPU-cap-concurrent -- thousands of configs, sequential is too slow) ─

def _run_source(source: str, op: str, marker: pathlib.Path) -> bool:
    """Benchmark every compiled config for a source concurrently, capped at
    DPU_CAP DPUs in flight (unlike cinm1comparison's task_bench, which runs
    strictly sequentially for wall-clock accuracy -- with O(1000) configs
    per source here, sequential execution would take far too long;
    paperplots accepts the extra timing noise from sharing hardware
    concurrently, matching the old run_best_configs.py's behavior).

    Skips any config that already has benchmark output on disk. This is
    what makes adopting the legacy Makefile-produced data cheap: run.done
    (this task's target) doesn't exist in that legacy layout, so this
    action runs once per already-completed source even after `doit
    reset-dep` -- but every config it looks at is already benchmarked, so
    that pass is a fast no-op, not a real re-run on real hardware."""
    configs = _discover_configs(source, op)
    compiled = compile_run.discover_compiled(configs, compile_root=PATHS.compile_root(source))
    ok = [c for c in compiled if c.ok]
    n_failed = len(compiled) - len(ok)
    if n_failed:
        print(f"  {n_failed}/{len(compiled)} configs not compiled for {source}, skipping them")

    def _already_run(c: compile_run.CompiledConfig) -> bool:
        out = PATHS.run_output_dir(source, c.config)
        return (out / f"{c.config.fn_name}_total.csv").exists()

    todo = [c for c in ok if not _already_run(c)]
    if len(todo) < len(ok):
        print(f"  {len(ok) - len(todo)}/{len(ok)} configs already benchmarked for {source}, skipping")

    run_root = PATHS.run_root(source)
    results = parallel.run_resource_capped(
        todo,
        lambda c: compile_run.run_config(c, run_root=run_root, iters=ITERS),
        cost_fn=lambda c: c.num_dpus,
        cap=DPU_CAP, workers=WORKERS, desc=f"run {source}",
        should_retry=lambda r: not r.ok and compile_run.is_dpu_allocation_error(r.error),
    )
    for r in results:
        if not r.ok:
            print(f"  FAIL run: {source} {r.compiled.config.fn_name} {r.compiled.config.label}: {r.error[:200]}")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


@create_after(executed="compile")
def task_bench():
    """One subtask per source (not per config): the DPU-cap scheduler needs
    every config for a source in hand at once to pack the shared budget."""
    for prim in PRIMS:
        op = prim.removeprefix("prim_")
        for target in TARGETS:
            source = f"{prim}_{target}"
            config_csvs = [str(PATHS.config_csv(source, c)) for c in _discover_configs(source, op)]
            if not config_csvs:
                continue
            marker = PATHS.run_done(source)
            yield {
                "name": source,
                "file_dep": config_csvs,
                "targets": [str(marker)],
                "actions": [(_run_source, [source, op, marker])],
            }


# ── aggregate ────────────────────────────────────────────────────────────────

def _aggregate_one(source: str) -> bool:
    aggregate.aggregate_run(PATHS.run_root(source), PATHS.compile_root(source), PATHS.aggregated_dir(source))
    return True


@create_after(executed="bench")
def task_aggregate():
    """Merge every config's per-iteration output CSVs into one CSV per
    measurement type (scatter/gather/alloc/free/total/...) -- see
    cinm_experiments.aggregate.aggregate_run."""
    for prim in PRIMS:
        for target in TARGETS:
            source = f"{prim}_{target}"
            marker = PATHS.run_done(source)
            if not marker.exists():
                continue
            yield {
                "name": source,
                "file_dep": [str(marker)],
                "targets": [str(PATHS.aggregated_dir(source) / "total.csv")],
                "actions": [(_aggregate_one, [source])],
            }


def _aggregate_bo_timings_one(source: str) -> bool:
    aggregate.aggregate_bo_timings(PATHS.results_dir(source), PATHS.bo_timings_dir(source))
    return True


@create_after(executed="bo")
def task_bo_timings():
    """Merge every seed's raw BO-search timings.csv into one per-source
    file -- independent of compile/run, only needs the BO search done."""
    for prim in PRIMS:
        for target in TARGETS:
            source = f"{prim}_{target}"
            if not PATHS.bo_out(source).exists():
                continue
            yield {
                "name": source,
                "file_dep": [str(PATHS.bo_out(source))],
                "targets": [str(PATHS.bo_timings_dir(source) / "timings.csv")],
                "actions": [(_aggregate_bo_timings_one, [source])],
            }


# ── plot ─────────────────────────────────────────────────────────────────────

def _plot_all() -> bool:
    run_plots(PATHS.data_dir, SOURCES, PATHS.plots_dir())
    return True


@create_after(executed="aggregate")
def task_plot():
    """Violin/bar/speedup plots across every source -- see
    plot_best_configs.py (kept local to this experiment; the pipeline
    plumbing above is generic, the figures aren't). Gated on "aggregate"
    (not also "bo_timings", which create_after can't express two of): by
    the time the much heavier compile+run+aggregate chain finishes for a
    source, its bo_timings (which only needs the much-earlier "bo" stage)
    is already done in practice."""
    agg_totals = [PATHS.aggregated_dir(s) / "total.csv" for s in SOURCES]
    bo_timings = [PATHS.bo_timings_dir(s) / "timings.csv" for s in SOURCES]
    file_dep = [str(p) for p in agg_totals + bo_timings if p.exists()]
    return {
        "file_dep": file_dep,
        "targets": [str(PATHS.plots_dir() / "net_times.csv")],
        "actions": [_plot_all],
    }

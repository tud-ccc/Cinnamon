"""doit tasks for cost_bench: exhaustively benchmark every valid config
already dumped in an existing cost-model oracle pool.csv, for one or more
primitives (red, gemv, ...), to validate the cost model (measured vs
predicted -- see plot_cost.py's cost_calibration plot). Unlike
cinm1comparison's BO-search comparison, there's no search stage here --
pool.csv is assumed to already exist under experiments/data/ (produced by
cinmopt.exhaustive_search or similar, run separately).

Adding a new benchmark later = one more PRIMS entry (source mlir + oracle
dir + config filter) plus an experiments/bench/<prim>.cpp driver; no other
code here needs to change.

doit tasks are one per (prim, function), not one per config row: oracle
pools run into hundreds of thousands of valid rows (gemv), and doit's own
task-graph bookkeeping (not the actual compile/bench work) becomes the
bottleneck well before that many doit tasks -- confirmed in practice, both
at million-row scale (`doit list` alone took over a minute) and, after
tightening the gemv filter down to ~209k configs total, at that scale too.
So each (prim, function) task internally loops over every filtered config
for that function, but -- unlike a plain compile_run.compile_configs/
run_configs call -- skips configs that already have a successful result on
disk (see _compile_fn/_bench_fn), so doit's own per-function tracking plus
this inner skip together give the practical effect of per-config tracking
without doit ever seeing more than a handful of tasks.

Usage:
  doit list                       # show all tasks
  doit                            # split -> compile -> bench -> agg -> plot (default)
  doit compile:red:red_4MB        # just compile red_4MB's configs (no hardware needed)
  doit forget bench:red:red_4MB   # force that function's hardware runs to redo
                                   # (only outstanding/failed configs actually rerun --
                                   # see _bench_one)
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import sys

from doit.reporter import ProgressBarReporter

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from cinm_experiments import compile_run, cinmopt, aggregate, pools, measurements  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402
from cinm_experiments.paths import python_bin  # noqa: E402

DOIT_CONFIG = {
    "default_tasks": ["plot"],
    "verbosity": 2,
    "continue": True,
    "reporter": ProgressBarReporter,
}

ITERS = 5
SYSTEM = "cinm2"  # single fixed system tag -- this pipeline doesn't compare systems


def _red_filter(p: dict) -> bool:
    return p["mramCol"] * p["dpus"] >= 64 * 1024 and p["dpus"] <= 512


def _gemv_filter(p: dict) -> bool:
    # TODO: needs a real filter before running the gemv sweep. gemv's oracle
    # pool is ~230k valid configs per function (vs red's ~19k), and dpus<=512
    # alone still leaves ~2.5M surviving configs for gemv_4MB. Needs either a
    # tighter structural constraint (mirroring red's `mramCol*dpus >= 64k`
    # total-coverage idea, using gemv's own tile dims) and/or an explicit cap
    # on how many configs to keep per function.
    return p["M"] == (p["dpus"] / p["dpuCols"]) * p["mramRow"] and p["K"] == p["dpuCols"] * p["mramCol"]


@dataclasses.dataclass(frozen=True)
class Prim:
    name: str
    source_mlir: pathlib.Path
    oracle_dir: pathlib.Path
    config_filter: callable
    dimensions: dict[str, dict]


PRIMS: dict[str, Prim] = {
    "red": Prim(
        name="red",
        source_mlir=EXPERIMENTS_DIR / "prim_red.mlir",
        oracle_dir=EXPERIMENTS_DIR / "data" / "prim_red_oracle",
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
        oracle_dir=HERE / "data" / "gemv_hybrid_400_oracle",
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

    def bench_marker(self, prim: str, fn_name: str) -> pathlib.Path:
        return self.run_root(prim) / f"{fn_name}.bench.done"

    def bench_bin(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return (
            self.compile_root(prim)
            / config.system
            / config.fn_name
            / config.label
            / "bin"
            / f"bench_{config.fn_name}"
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

    def plots_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "plots"

    def pool_measured_dir(self, prim: str) -> pathlib.Path:
        return self.data_dir / prim / "pool_measured"


PATHS = Paths(DATA_DIR)


# ── config discovery ─────────────────────────────────────────────────────────


def _oracle_pools(oracle_dir: pathlib.Path):
    """Yield (fn_name, pool_csv) for every infer_<fn>/pool.csv under oracle_dir."""
    for sub in sorted(oracle_dir.iterdir()):
        pool_csv = sub / "pool.csv"
        if sub.is_dir() and pool_csv.exists():
            yield sub.name.removeprefix("infer_"), pool_csv


def _fn_names(prim_name: str) -> list[str]:
    return [fn for fn, _ in _oracle_pools(PRIMS[prim_name].oracle_dir)]


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
    pool_csv = prim.oracle_dir / f"infer_{fn_name}" / "pool.csv"
    problem_dims = fn_name.removeprefix(prim.name + "_")
    fn_module = PATHS.split_module(prim_name, fn_name)
    df = pools.load_valid(pool_csv)
    cols = pools.param_cols(df)
    configs = []
    for row in df[cols].itertuples(index=True, name=None):
        idx, values = row[0], row[1:]
        params = dict(zip(cols, (int(v) for v in values)))
        params |= prim.dimensions[problem_dims]
        if not prim.config_filter(params):
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
    return tuple(configs)


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
    print(f"  {prim_name}:{fn_name}: {len(pending)}/{len(configs)} configs need compiling")
    compile_run.compile_configs(pending, compile_root=PATHS.compile_root(prim_name))
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
                "file_dep": [str(PATHS.split_module(name, fn_name))],
                "targets": [str(marker)],
                "actions": [(_compile_fn, [name, fn_name, marker])],
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
    compiled = compile_run.discover_compiled(configs, compile_root=PATHS.compile_root(prim_name))
    pending = [
        c for c in compiled
        if c.ok and measurements.net_time_ms(PATHS.output_dir(prim_name, c.config)) is None
    ]
    print(f"  {prim_name}:{fn_name}: {len(pending)}/{len(compiled)} configs need a bench run")
    compile_run.run_configs(pending, run_root=PATHS.run_root(prim_name), iters=ITERS)
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
                "actions": [(_bench_fn, [name, fn_name, marker])],
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


# ── plot ─────────────────────────────────────────────────────────────────────


def task_plot():
    """Regenerate this prim's plots/ (per-metric-vs-dim scatter plots, plus
    the cost_calibration measured-vs-predicted plot). Shells out to
    plot_cost.py (a standalone CLI script, also runnable on its own) rather
    than importing it, so its own --filter/--oracle CLI stays the single
    source of truth for what it does. Like task_agg, always reruns -- cheap
    relative to compile/bench, and its output set is dynamic (one plot per
    function/metric found)."""
    for name, prim in PRIMS.items():
        cmd = (
            f"{python_bin()} plot_cost.py "
            f"--in-dir {PATHS.agg_dir(name)} --out-dir {PATHS.plots_dir(name)} "
            f"--oracle {prim.oracle_dir} --pool-out-dir {PATHS.pool_measured_dir(name)}"
        )
        yield {
            "name": name,
            "task_dep": [f"agg:{name}"],
            "uptodate": [False],
            "actions": [cmd],
        }

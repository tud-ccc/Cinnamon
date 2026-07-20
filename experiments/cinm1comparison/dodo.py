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

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from cinm_experiments import cinm1, cinmopt, compile_run, measurements, pools  # noqa: E402
from cinm_experiments.paths import DEFAULT_CINM_OPT  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402

from plot import plot_speedup, print_summary  # noqa: E402

PRIMS = ["prim_gemv", "prim_red"]
DATA_DIR = HERE / "data"
CINM_OPT = pathlib.Path(os.environ.get("CINM_OPT", DEFAULT_CINM_OPT))

OPTS = dict(
    top_frac=0.10,
    min_configs=200,
    n_seeds=32,
    iters=10,
    screen_sim="cycle-accurate",
    workers=os.cpu_count(),
)

# Seed k in a multiseed run uses k*31+offset for k in 1..n_seeds; offsets must
# be spaced by more than n_seeds*31 apart so different pairs' seed_* dirs
# never collide within the same {fn_name}/ results directory.
_OFFSET_STRIDE = 4096
_BASE_OFFSET = 67

DOIT_CONFIG = {"default_tasks": ["plot"], "verbosity": 2, "continue": True}


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

    def cinm2_pool_csv(self, prim: str, fn_name: str, seed: str) -> pathlib.Path:
        return self.cinm2_results_dir(prim) / f"infer_{fn_name}" / f"seed_{seed}" / "pool.csv"

    def cinm2_search_marker(self, prim: str, fn_name: str, dpus: int, tasklets: int) -> pathlib.Path:
        return self.prim_dir(prim) / "cinm2_search_markers" / fn_name / f"D{dpus}_T{tasklets}.done"

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

    def run_output_dir(self, prim: str, config: compile_run.Config) -> pathlib.Path:
        return self.run_root(prim) / config.system / config.fn_name / config.label / "output"

    def bench_done(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "bench.done"

    def comparison_csv(self, prim: str) -> pathlib.Path:
        return self.prim_dir(prim) / "comparison.csv"

    def plots_dir(self) -> pathlib.Path:
        return self.data_dir / "plots"


PATHS = Paths(EXPERIMENTS_DIR, DATA_DIR)


# ── split ────────────────────────────────────────────────────────────────────

def _split_one(prim_mlir: pathlib.Path, split_dir: pathlib.Path) -> bool:
    split_source(prim_mlir, split_dir)  # dict return value isn't JSON-picklable for doit's DB
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
        fn_module, screen_dir, workers=OPTS["workers"],
        infer_opts={"use-mram-tiling": False, "simulator": OPTS["screen_sim"]},
        cinm_opt=CINM_OPT,
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
    print(f"  {fn_name:20s}  kept {n_kept} of {n_valid} valid rows"
          f" -> {len(pairs)} (dpus,tasklets) pairs")
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
    dpus: int,
    tasklets: int,
    offset: int,
) -> bool:
    results_dir = PATHS.cinm2_results_dir(prim)
    print(f"  {fn_name} dpus={dpus} tasklets={tasklets}: BO search "
          f"({OPTS['n_seeds']} seeds, offset={offset})")
    cinmopt.bo_multiseed(
        fn_module,
        results_dir,
        n_seeds=OPTS["n_seeds"],
        offset=offset,
        workers=OPTS["workers"],
        infer_opts={"fixed-dpus": dpus, "fixed-tasklets": tasklets},
        cinm_opt=CINM_OPT,
    )
    return True

def get_offset(pair_idx):
    return _BASE_OFFSET + pair_idx * _OFFSET_STRIDE

def gen_seeds(pair_idx):
    offset = get_offset(pair_idx)
    return (31 * k + offset for k in range(0, OPTS["n_seeds"]))

@create_after(executed="screen", creates=["compile_cinm2", "cinm2_search"])
def task_cinm2_search():
    """For each selected working group, run CINM 2.0's Bayesian search
    n_seeds times with (dpus, tasklets) pinned (MRAM tiling enabled -- CINM
    2.0's normal codegen)."""
    if OPTS["n_seeds"] * 31 >= _OFFSET_STRIDE:
        raise RuntimeError(f"n_seeds {OPTS['n_seeds']} too large for offset stride {_OFFSET_STRIDE}")

    for prim in PRIMS:
        for fn_name in list_functions(PATHS.source_mlir(prim)):
            pairs_csv = PATHS.pairs_csv(prim, fn_name)
            if not pairs_csv.exists():
                continue
            fn_module = PATHS.split_module(prim, fn_name)
            pairs = pd.read_csv(pairs_csv)
            for pair_idx, (dpus, tasklets) in enumerate(pairs[["dpus", "tasklets"]].itertuples(index=False)):
                yield {
                    "basename": "cinm2_search",
                    "name": f"{prim}:{fn_name}:D{dpus}_T{tasklets}",
                    "file_dep": [str(pairs_csv)],
                    "targets": [
                        str(PATHS.cinm2_pool_csv(prim, fn_name, seed)) for seed in gen_seeds(pair_idx)
                    ],
                    "actions": [
                        (
                            _cinm2_search_one,
                            [
                                prim,
                                fn_name,
                                fn_module,
                                int(dpus),
                                int(tasklets),
                                get_offset(pair_idx),
                            ],
                        )
                    ],
                }

                for seed in gen_seeds(pair_idx):
                  seed = str(seed)
                  config = compile_run.Config(
                      system="cinm2",
                      fn_name=fn_name,
                      label=seed,
                      # Will be replaced once we know which config params are the best
                      params={}, 
                      fn_module=fn_module,
                      prim=prim.removeprefix("prim_"),
                      lower=None # lower also gets replaced
                  )
                  compile_root = PATHS.compile_root(prim)
                  marker = PATHS.compile_marker(prim, config)
                  pool_csv = PATHS.cinm2_pool_csv(prim, fn_name, seed)
                  yield {
                      "basename": "compile_cinm2",
                      "name": f"{prim}:{fn_name}:D{dpus}_T{tasklets}:{seed}",
                      "file_dep": [str(pool_csv)],
                      "targets": [str(marker)],
                      "actions": [(_compile_best, [config, pool_csv, compile_root, marker])],
                  }

# ── compile ──────────────────────────────────────────────────────────────────

def _compile_best(config: compile_run.Config, pool_csv: pathlib.Path, compile_root: pathlib.Path, marker: pathlib.Path) -> bool:
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
    config.lower=cinmopt.eval_solution_lowerer(config.params, cinm_opt=CINM_OPT)
    return _compile_one(config, compile_root, marker)

def _compile_one(config: compile_run.Config, compile_root: pathlib.Path, marker: pathlib.Path) -> bool:
    """Never raises: a config that fails to compile is recorded (printed +
    left out of the marker's sibling bin/) but must not block sibling
    configs' bench task from running -- doit treats a raised exception as a
    hard failure and skips every downstream task that depends on it, which
    is more than we want for one bad config out of many. discover_compiled()
    already treats a missing bench_* binary as a per-config failure, so
    downstream stages tolerate this fine."""
    compiled = compile_run.compile_config(config, compile_root=compile_root)
    if not compiled.ok:
        print(f"  FAIL compile: {config.system} {config.fn_name} {config.label}: {compiled.error}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


@create_after(executed="screen")
def task_compile_cinm1():
    """Compile CINM 1.0 once per selected working group -- no search, its
    tile sizes are inferred deterministically."""
    for prim in PRIMS:
        op = prim.removeprefix("prim_")
        compile_root = PATHS.compile_root(prim)
        for fn_name in list_functions(PATHS.source_mlir(prim)):
            pairs_csv = PATHS.pairs_csv(prim, fn_name)
            if not pairs_csv.exists():
                continue
            fn_module = PATHS.split_module(prim, fn_name)
            pairs = pd.read_csv(pairs_csv)
            for dpus, tasklets in pairs[["dpus", "tasklets"]].itertuples(index=False):
                dpus, tasklets = int(dpus), int(tasklets)
                label = f"D{dpus}_T{tasklets}"
                config = compile_run.Config(
                    system="cinm1", fn_name=fn_name, label=label,
                    params={"dpus": dpus, "tasklets": tasklets},
                    fn_module=fn_module, prim=op,
                    lower=cinm1.lowerer(dpus, tasklets, cinm_opt=CINM_OPT),
                )
                marker = PATHS.compile_marker(prim, config)
                yield {
                    "name": f"{prim}:{fn_name}:{label}",
                    "file_dep": [str(pairs_csv)],
                    "targets": [str(marker)],
                    "actions": [(_compile_one, [config, compile_root, marker])],
                }



# ── run (sequential -- accurate wall-clock timing) ──────────────────────────

def _discover_configs(prim: str) -> list[compile_run.Config]:
    """Reconstruct every Config for a prim from what screen/cinm2_search
    already wrote to disk (mirrors build_cinm1_configs/build_cinm2_configs
    in experiment.py, but reading state back instead of computing it)."""
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
            configs.append(compile_run.Config(
                system="cinm1", fn_name=fn_name, label=f"D{dpus}_T{tasklets}",
                params={"dpus": dpus, "tasklets": tasklets}, fn_module=fn_module, prim=op,
                lower=cinm1.lowerer(dpus, tasklets, cinm_opt=CINM_OPT),
            ))

    results_dir = PATHS.cinm2_results_dir(prim)
    if results_dir.exists():
        for fn_name, seed, params in pools.best_per_seed(results_dir):
            fn_module = PATHS.split_module(prim, fn_name)
            configs.append(compile_run.Config(
                system="cinm2", fn_name=fn_name, label=seed, params=params,
                fn_module=fn_module, prim=op,
                lower=cinmopt.eval_solution_lowerer(params, cinm_opt=CINM_OPT),
            ))
    return configs


def _bench_prim(prim: str) -> bool:
    configs = _discover_configs(prim)
    compiled = compile_run.discover_compiled(configs, compile_root=PATHS.compile_root(prim))
    print(f"\n=== {prim}: running {len(compiled)} configs ===")
    compile_run.run_configs(
        compiled, run_root=PATHS.run_root(prim), iters=OPTS["iters"]
    )
    return True


@create_after(executed="screen")
def task_bench():
    """Benchmark every compiled config for a prim, one at a time -- not
    parallel, so concurrent hardware runs don't skew wall-clock timing.
    One task per prim (not per config): running is inherently sequential
    here, so there is no parallelism for doit to schedule within it, and a
    single task per prim keeps the dependency graph simple (it just needs
    every compile_cinm1/compile_cinm2 subtask for that prim to be done).

    Depends on each config's compile.done marker rather than its bench_*
    binary: the marker is always written, even when that one config failed
    to compile, so one bad config doesn't stop doit from running bench for
    every config that did compile (discover_compiled treats a missing
    binary as a per-config failure, not a fatal one)."""
    for prim in PRIMS:
        # compile_markers = [str(PATHS.compile_marker(prim, c)) for c in _discover_configs(prim)]
        # if not compile_markers:
        #     continue
        # marker = PATHS.bench_done(prim)
        yield {
            "name": prim,
            "uptodate": [
                result_dep(f"compile_cinm2:{prim}"),
                result_dep(f"compile_cinm1:{prim}"),
            ],
            # "file_dep": compile_markers,
            "actions": [(_bench_prim, [prim])],
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



def compare(cinm1_results: list[compile_run.RunResult],
            cinm2_results: list[compile_run.RunResult]) -> pd.DataFrame:
    """Merge CINM 1.0 (one point per working group) with CINM 2.0 (n_seeds
    points per working group -> seed-median) into a speedup table keyed by
    (fn_name, dpus, tasklets)."""
    cinm1 = measurements.results_to_frame(cinm1_results).drop(columns=["label"])
    cinm1 = cinm1.rename(columns={"net_time_ms": "cinm1_ms"})

    cinm2_raw = measurements.results_to_frame(cinm2_results)
    cinm2_summary = (
        cinm2_raw.groupby(["fn_name", "dpus", "tasklets"])["net_time_ms"]
        .agg(cinm2_ms="median",
             cinm2_p25=lambda s: s.quantile(0.25),
             cinm2_p75=lambda s: s.quantile(0.75),
             cinm2_n="count")
        .reset_index()
    )

    merged = cinm1.merge(cinm2_summary, on=["fn_name", "dpus", "tasklets"], how="inner")
    missing = set(zip(cinm1.fn_name, cinm1.dpus, cinm1.tasklets)) - set(
        zip(merged.fn_name, merged.dpus, merged.tasklets)
    )
    if missing:
        print(f"  WARNING: {len(missing)} (fn_name,dpus,tasklets) pairs have CINM1 "
              f"but no CINM2 data: {sorted(missing)[:5]}...", file=sys.stderr)
    merged["speedup"] = merged["cinm1_ms"] / merged["cinm2_ms"]
    return merged


def _compare_prim(prim: str) -> bool:
    configs = _discover_configs(prim)
    compiled = compile_run.discover_compiled(configs, compile_root=PATHS.compile_root(prim))

    # Reconstruct RunResults by pointing at the already-written output dirs
    # (no need to re-run bench_* -- bench.done guarantees they exist).
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


# @create_after(executed="bench")
@create_after(executed="screen")
def task_compare():
    """Geomean speedup of CINM 2.0 (seed-median) over CINM 1.0 per working
    group, aggregated per benchmark."""
    for prim in PRIMS:
        yield {
            "name": prim,
            "uptodate": [result_dep(f"bench:{prim}")],
            "targets": [str(PATHS.comparison_csv(prim))],
            "actions": [(_compare_prim, [prim])],
        }


def _plot_all() -> bool:
    comparison = pd.concat(
        [pd.read_csv(PATHS.comparison_csv(prim)) for prim in PRIMS
         if PATHS.comparison_csv(prim).exists()],
        ignore_index=True,
    )
    out_dir = PATHS.plots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    print_summary(comparison)
    plot_speedup(comparison, out_dir)
    return True


# @create_after(executed="compare")
def task_plot():
    comparison_csvs = [PATHS.comparison_csv(prim) for prim in PRIMS]
    return {
        "file_dep": [str(p) for p in comparison_csvs if p.exists()],
        "targets": [str(PATHS.plots_dir() / "cinm1_vs_cinm2_speedup.pdf")],
        "actions": [_plot_all],
    }

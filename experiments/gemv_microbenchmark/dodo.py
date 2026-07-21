"""doit tasks for the gemv_64MB CINM 1.0 vs CINM 2.0 comparison: one compile
task per config, one run task per config (each depending on its own
compile), then a plot task depending on both runs.

Config objects hold a `lower` Callable (the cinm-opt lowering pipeline),
which can never be JSON-encoded, so they can't be passed between tasks via
doit's getargs. Instead CONFIGS is just a module-level Python list that every
task closure captures directly (no serialization needed -- it's all one
process), and stages are connected by marker files on disk instead, matching
cinm1comparison/dodo.py: `doit` then only reruns what's actually stale, e.g.
rerunning after a plot tweak doesn't recompile or re-benchmark on hardware.

Usage:
  doit list         # show all tasks
  doit              # compile + run + plot (default task)
  doit compile      # just compile both configs
  doit forget run   # force both hardware runs to redo next time
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from doit.reporter import ProgressBarReporter  # noqa: E402
from cinm_experiments import compile_run, cinm1, cinmopt, measurements  # noqa: E402

DOIT_CONFIG = {"default_tasks": ["plot"], "reporter": ProgressBarReporter}

COMPILE_ROOT = HERE / "data"
RUN_ROOT = HERE / "run"
ITERS = 10

cinm2parms = {
    "dpus": 256,
    "tasklets": 4,
    "wramRow": 1,
    "wramCol": 1024,
    "dpuCols": 1,
    "mramRow": 4,
    "mramCol": 1024,
}

CONFIGS = [
    compile_run.Config(
        system="cinm2as1",
        fn_name="gemv_64MB",
        label="foo",
        params=cinm2parms,
        fn_module="/home/clement.fournier/Work/cinm-mlir/experiments/cinm1comparison/data/prim_gemv/_split/gemv_64MB.mlir",
        prim="gemv",
        lower=cinmopt.eval_solution_lowerer(params=cinm2parms),
    ),
    compile_run.Config(
        system="cinm1",
        fn_name="gemv_64MB",
        label="foo",
        params={"dpus": 256, "tasklets": 4},
        fn_module="/home/clement.fournier/Work/cinm-mlir/experiments/cinm1comparison/data/prim_gemv/_split/gemv_64MB.mlir",
        prim="gemv",
        lower=cinm1.lowerer(dpus=256, tasklets=4),
    ),
]


def _config_dir(root: pathlib.Path, config: compile_run.Config) -> pathlib.Path:
    return root / config.system / config.fn_name / config.label


def _output_dir(config: compile_run.Config) -> pathlib.Path:
    return _config_dir(RUN_ROOT, config) / "output"


def _compile_one(config: compile_run.Config, marker: pathlib.Path) -> bool:
    """Never raises: a config that fails to compile is recorded (printed,
    marker left untouched) but must not stop doit from at least attempting
    the sibling config -- a raised exception would abort the whole doit run,
    not just this task."""
    compiled = compile_run.compile_config(config, compile_root=COMPILE_ROOT)
    if not compiled.ok:
        print(f"  FAIL compile: {config.system}: {compiled.error}")
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def _run_one(config: compile_run.Config, marker: pathlib.Path) -> bool:
    """Never raises, mirrors _compile_one. A hardware run failure is printed
    but the marker is still touched, so a flaky run isn't silently retried
    on every `doit` invocation -- rerun it explicitly with `doit forget
    run:<system> && doit run:<system>`."""
    compiled = compile_run.discover_compiled([config], compile_root=COMPILE_ROOT)[0]
    result = compile_run.run_config(compiled, run_root=RUN_ROOT, iters=ITERS)
    if not result.ok and compile_run.is_dpu_allocation_error(result.error):
        result = compile_run.run_config(compiled, run_root=RUN_ROOT, iters=ITERS)
    if not result.ok:
        print(f"  FAIL run: {config.system}: {result.error[:200]}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def task_compile():
    """Compile each config to a upmem binary."""
    for config in CONFIGS:
        marker = _config_dir(COMPILE_ROOT, config) / "compile.done"
        yield {
            "name": config.system,
            "file_dep": [config.fn_module],
            "targets": [str(marker)],
            "actions": [(_compile_one, [config, marker])],
        }


def task_bench():
    """Benchmark each compiled config on hardware. doit runs same-priority
    tasks in the order they're yielded, so with CONFIGS' fixed order this
    stays sequential across configs -- concurrent hardware runs would
    contend for host/DPU resources and skew wall-clock timing."""
    for config in CONFIGS:
        compile_marker = _config_dir(COMPILE_ROOT, config) / "compile.done"
        run_marker = _config_dir(RUN_ROOT, config) / "run.done"
        yield {
            "name": config.system,
            "file_dep": [str(compile_marker)],
            "targets": [str(run_marker)],
            "actions": [(_run_one, [config, run_marker])],
        }


def task_plot():
    """Print + plot the net-time breakdown (scatter/gather/copy/launch/
    unaccounted) for both systems."""

    def action():
        breakdowns = {}
        for config in CONFIGS:
            output_dir = _output_dir(config)
            net_ms = measurements.net_time_ms(output_dir)
            if net_ms is None:
                print(f"{config.system}: no results (run failed?)")
                continue
            breakdown = measurements.net_breakdown_ms(output_dir)
            breakdowns[config.system] = breakdown
            parts = "  ".join(f"{k}={v:.3f}ms" for k, v in breakdown.items())
            print(f"{config.system}: net_time={net_ms:.3f}ms  {parts}")

        if not breakdowns:
            return

        import matplotlib.pyplot as plt

        systems = list(breakdowns.keys())
        categories = measurements.NET_BREAKDOWN_CATEGORIES
        colors = [plt.get_cmap("Dark2")(i) for i in range(len(categories))]

        fig, ax = plt.subplots(figsize=(5, 5))
        bottoms = [0.0] * len(systems)
        for cat, color in zip(categories, colors):
            values = [breakdowns[s][cat] for s in systems]
            totals = [sum(breakdowns[s].values()) for s in systems]
            bars = ax.bar(systems, values, bottom=bottoms, label=cat, color=color,
                           edgecolor="white", linewidth=1)
            for bar, v, total in zip(bars, values, totals):
                if v > 0.03 * total:  # skip labeling slivers -- they'd just overlap
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + v / 2,
                            f"{v:.1f}", ha="center", va="center", fontsize=8, color="white")
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_ylabel("time (ms)")
        ax.set_title("gemv_64MB net-time breakdown")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        out_path = HERE / "breakdown.png"
        fig.savefig(out_path, dpi=150)
        print(f"wrote {out_path}")

    return {
        "file_dep": [str(_config_dir(RUN_ROOT, c) / "run.done") for c in CONFIGS],
        "targets": [str(HERE / "breakdown.png")],
        "actions": [action],
    }

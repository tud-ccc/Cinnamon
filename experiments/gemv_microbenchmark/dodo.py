"""doit tasks for the mtv_64MB CINM 1.0 vs CINM 2.0 comparison: one compile
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

from doit.tools import check_timestamp_unchanged
from doit.reporter import ProgressBarReporter  # noqa: E402
from cinm_experiments import compile_run, cinmopt, measurements  # noqa: E402
from cinm_experiments.paths import DEFAULT_CINM_OPT

DOIT_CONFIG = {
    "default_tasks": ["plot"],
    "verbosity": 2,
    "reporter": ProgressBarReporter,
}

DATA_ROOT = HERE / "data"
ITERS = 10

# mtv 64MB: 4096x4096

source = "/home/clement.fournier/Work/cinm-mlir/experiments/cinm1comparison/data/prim_mtv/_split/mtv_64MB.mlir"
CONFIGS = [
    # compile_run.Config(
    # system="cinm2",
    # fn_name="mtv_64MB",
    # label="default",
    # # FIXME: not expressible in the generic search space yet.
    # # This configuration relies on *sequential outer trips*: with
    # # dpus=256 tasklets=4 (1024 leaves) it covers the 4096x4096 problem in
    # # 4 trips over M and 4 over K. The generic space requires the tile
    # # counts to fill the workgroup exactly and has no notion of trips --
    # # design §G2 parked them in --cinm-tiling, which the generic pipeline
    # # no longer runs. Left in the old parameter names deliberately so it
    # # fails loudly rather than being silently reinterpreted.
    # params={
    # "dpus": DPUS,
    # "tasklets": TASKLETS,
    # "taskletCols": 1,
    # "wramRow": 1,
    # "wramCol": 1024,
    # "dpuCols": 1,
    # "mramRow": 4,
    # "mramCol": 1024,
    # },
    # fn_module=source,
    # prim="mtv",
    # lower=cinmopt.eval_solution_lowerer(),
    # ),
    compile_run.Config(
        system="cinm2",
        fn_name="mtv_64MB",
        # An ad-hoc configuration using only 512 DPUs,
        # I use this for comparing the templates vs
        # generic flow bc other researchers are using
        # the DPU array and won't let me allocate 2048 DPUs.
        label="dpu512",
        params={
            "dpus": 512,
            "tasklets": 8,
            "gemv.M0": 8,  # mramRow * taskletCols / tasklets
            "gemv.K0": 512,  # mramCol / taskletCols
            "gemv.M1": 8,  # wramRow
            "gemv.K1": 64,  # wramCol
            # The workgroup mapping: k-tile index outermost, so the tasklets of
            # a DPU split rows and share the vector (taskletCols = 1).
            "gemv.order": 0,
        },
        fn_module=source,
        prim="mtv",
        lower=cinmopt.eval_solution_lowerer(),
    ),
    compile_run.Config(
        system="cinm2",
        fn_name="mtv_64MB",
        # This one is the atim2048 optimum,
        # expressed as a point in the cinm2
        # search space. It corresponds precisely
        # to the ATiM optimum because of the
        # workgroup mapping strategy, which is
        # now the gemv.order parameter below;
        # 0 is the strategy CINM2 used to assume.
        label="atim2048optimum",
        params={
            "dpus": 2048,
            "tasklets": 8,
            "gemv.M0": 8,  # mramRow * taskletCols / tasklets
            "gemv.K0": 128,  # mramCol / taskletCols
            "gemv.M1": 8,  # wramRow
            "gemv.K1": 64,  # wramCol
            # The workgroup mapping: k-tile index outermost, so the tasklets of
            # a DPU split rows and share the vector (taskletCols = 1).
            "gemv.order": 0,
        },
        fn_module=source,
        prim="mtv",
        lower=cinmopt.eval_solution_lowerer(),
    ),
    compile_run.Config(
        system="cinm2",
        fn_name="mtv_64MB",
        # This is just another CINM2 config that gives rise
        # to a partial reduction on the host. Not directly
        # comparable to the atim2048 optimum.
        label="partialred2048",
        params={
            "dpus": 2048,
            "tasklets": 8,
            "taskletCols": 1,
            "wramRow": 1,
            "wramCol": 1024,
            "dpuCols": 2,
            "mramRow": 4,
            "mramCol": 1024,
        },
        fn_module=source,
        prim="mtv",
        lower=cinmopt.eval_solution_lowerer(),
    ),
    # These two are the config picked by cinm1 when
    # looking with the parameters of the atim2048
    # optimum. They are comparable to the atim2048
    # optimum but may find different tile sizes and
    # workgroup mapping. CINM2 optimizations are
    # disabled for them, and they use a CINM1-like
    # lowering flow.
    # compile_run.Config(
    #     system="cinm1",
    #     fn_name="mtv_64MB",
    #     label="atim2048optimum",
    #     params={"dpus": 2048, "tasklets": 8},
    #     fn_module=source,
    #     prim="mtv",
    #     lower=cinm1.lowerer(),
    # ),
    # compile_run.Config(
    #     system="cinm1_with_sg",
    #     fn_name="mtv_64MB",
    #     label="atim2048optimum",
    #     params={"dpus": 2048, "tasklets": 8},
    #     fn_module=source,
    #     prim="mtv",
    #     lower=cinm1.lowerer(use_upmem_scatter_api=True),
    # ),
]


def _config_dir(root: pathlib.Path, config: compile_run.Config) -> pathlib.Path:
    return root / config.system / config.fn_name / config.label


def _compile_one(config: compile_run.Config, marker) -> bool:
    # This should not fail
    compiled = compile_run.compile_config(config, compile_root=DATA_ROOT)
    if not compiled.ok:
        print(f"  FAIL compile: {config.system}: {compiled.error}")
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def _run_one(config: compile_run.Config, marker) -> bool:
    compiled = compile_run.discover_compiled([config], compile_root=DATA_ROOT)[0]
    result = compile_run.run_config(compiled, run_root=DATA_ROOT, iters=ITERS)
    if not result.ok:
        print(f"  FAIL run: {config.system}: {result.error[:200]}")
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def task_compile():
    """Compile each config to a upmem binary."""
    for config in CONFIGS:
        compile_marker = config.dir(DATA_ROOT) / "compile.done"
        yield {
            "name": config.label + ":" + config.system,
            # todo add directory check?
            # "uptodate": [check_timestamp_unchanged(compile_marker)],
            "file_dep": [config.fn_module, DEFAULT_CINM_OPT],
            "targets": [compile_marker],
            "actions": [(_compile_one, [config, compile_marker])],
        }


def task_bench():
    """Benchmark each compiled config on hardware. doit runs same-priority
    tasks in the order they're yielded, so with CONFIGS' fixed order this
    stays sequential across configs -- concurrent hardware runs would
    contend for host/DPU resources and skew wall-clock timing."""
    for config in CONFIGS:
        compile_marker = config.dir(DATA_ROOT) / "compile.done"
        run_marker = config.dir(DATA_ROOT) / "bench.done"
        yield {
            "name": config.label + ":" + config.system,
            "uptodate": [check_timestamp_unchanged(compile_marker)],
            "file_dep": [compile_marker],
            "targets": [run_marker],
            "actions": [(_run_one, [config, run_marker])],
        }


def task_plot():
    """Print + plot the net-time breakdown (scatter/gather/copy/launch/
    unaccounted) for both systems."""

    def action(out_path, label, confs, by_kind):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        breakdowns = {}
        for config in confs:
            output_dir = config.dir(DATA_ROOT) / "output"
            net_ms = measurements.net_time_ms(output_dir)
            if net_ms is None:
                print(f"{config.system}: no results (run failed?)")
                continue
            breakdown = measurements.net_breakdown_ms(output_dir, by_kind=by_kind)
            breakdowns[config.system] = breakdown
            parts = "  ".join(f"{k}={v:.3f}ms" for k, v in breakdown.items())
            print(f"{config.system}: net_time={net_ms:.3f}ms  {parts}")

        if not breakdowns:
            return

        import matplotlib.pyplot as plt

        systems = list(breakdowns.keys())
        categories = sorted(
            {cat for s in systems for cat in breakdowns[s].keys()},
            key=measurements.net_breakdown_sort_ix,
        )
        # Color by each category's fixed, global sort index (not its position
        # in this plot's local `categories` list) so the same category name
        # always gets the same color across both plots, even though the flat
        # and by-kind plots don't show the same set of categories.
        colors = [
            plt.get_cmap("Dark2")(measurements.net_breakdown_color_ix(cat))
            for cat in categories
        ]

        fig, ax = plt.subplots(figsize=(5, 5))
        bottoms = [0.0] * len(systems)
        for cat, color in zip(categories, colors):
            values = [breakdowns[s].get(cat, 0) for s in systems]
            totals = [sum(breakdowns[s].values()) for s in systems]
            bars = ax.bar(
                systems,
                values,
                bottom=bottoms,
                label=cat,
                color=color,
                edgecolor="white",
                linewidth=1,
            )
            for bar, v, total in zip(bars, values, totals):
                if v > 0.03 * total:  # skip labeling slivers -- they'd just overlap
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + v / 2,
                        f"{v:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_ylabel("time (ms)")
        ax.set_title(f"mtv_64MB net-time breakdown ({label})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(systems, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"wrote {out_path}")

    by_label: dict[str, list[compile_run.Config]] = {}
    for c in CONFIGS:
        by_label.setdefault(c.label, []).append(c)

    # print({label: [c.system for c in confs] for label, confs in by_label.items()})
    for label, confs in by_label.items():
        out_path = HERE / "plots" / label / "breakdown.png"
        out_path_by_kind = HERE / "plots" / label / "breakdown_by_kind.png"
        yield {
            "name": label,
            "uptodate": [
                check_timestamp_unchanged(c.dir(DATA_ROOT) / "bench.done", "ctime")
                for c in confs
            ],
            "file_dep": [c.dir(DATA_ROOT) / "bench.done" for c in confs],
            "targets": [out_path, out_path_by_kind],
            "actions": [
                (action, [out_path, label, confs, False]),
                (action, [out_path_by_kind, label, confs, True]),
            ],
        }

"""doit tasks for the CINM 1.0 vs CINM 2.0 comparison: one compile task per
config, one run task per config (each depending on its own compile), then a
plot task depending on both runs.

A config names a prim and one of its functions; the source to split and the
module to compile follow from those, so benchmarking another prim is a new
entry in CONFIGS and nothing else.

Config objects hold a `lower` Callable (the cinm-opt lowering pipeline),
which can never be JSON-encoded, so they can't be passed between tasks via
doit's getargs. Instead CONFIGS is just a module-level Python list that every
task closure captures directly (no serialization needed -- it's all one
process), and stages are connected by marker files on disk instead, matching
cinm1comparison/dodo.py: `doit` then only reruns what's actually stale, e.g.
rerunning after a plot tweak doesn't recompile or re-benchmark on hardware.

Usage:
  doit list         # show all tasks
  doit              # compile + run + plot + fidelity (default tasks)
  doit compile      # just compile both configs
  doit fidelity     # just the predicted-vs-measured plot (no hardware needed
                    # beyond an existing bench run)
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
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402
from cinm_experiments.paths import DEFAULT_CINM_OPT

DOIT_CONFIG = {
    "default_tasks": ["plot", "fidelity"],
    "verbosity": 2,
    "reporter": ProgressBarReporter,
}

DATA_ROOT = HERE / "data"
SPLIT_DIR = DATA_ROOT / "_split"
ITERS = 50


def _prim_source(prim: str) -> pathlib.Path:
    """The shared multi-function source that a prim's functions are split out
    of, named after the prim rather than spelled out per config.

    These sources, rather than another experiment's data directory, are what
    carry `{cinm.static}` on the weight operand; without it the weight
    transfer and any repack of it are charged to every inference instead of
    amortizing."""
    return EXPERIMENTS_DIR / f"prim_{prim}.mlir"


def prim_config(**kwargs) -> compile_run.Config:
    """A compile_run.Config whose `fn_module` is the module task_split writes
    for `fn_name`.

    So a config names the function it wants and the prim it belongs to, and
    the paths follow: adding an experiment for another prim needs no source
    path here and no split task of its own."""
    kwargs.setdefault("fn_module", SPLIT_DIR / f"{kwargs['fn_name']}.mlir")
    return compile_run.Config(**kwargs)


CONFIGS = [
    prim_config(
        system="atim",
        fn_name="mtv_64MB",
        # This one is the atim2048 optimum,
        # expressed as a point in the cinm2
        # search space. It corresponds precisely
        # to the ATiM optimum because of the
        # workgroup mapping strategy, which is
        # now the gemv.order parameter below;
        # 0 is the strategy CINM2 used to assume.
        label="mtv",
        params={
            "dpus": 2048,
            "tasklets": 8,
            "gemv.M.mram": 8,  # mramRow * taskletCols / tasklets
            "gemv.K.mram": 128,  # mramCol / taskletCols
            "gemv.M.wram": 8,  # wramRow
            "gemv.K.wram": 64,  # wramCol
            # The workgroup mapping: k-tile index outermost, so the tasklets of
            # a DPU split rows and share the vector (taskletCols = 1).
            "gemv.order[0]": 2,
            "gemv.order[1]": 1,
        },
        prim="mtv",
        lower=cinmopt.eval_solution_lowerer(
            extra_infer_opts={"simulator": "cycle-accurate", "debug-pipeline": "true"}
        ),
    ),
    prim_config(
        system="cinm2",
        fn_name="mtv_64MB",
        # A point I found via search with 512 evals
        label="mtv",
        params={
            "dpus": 2048,
            "tasklets": 8,
            "gemv.M.mram": 4,
            "gemv.M.wram": 4,
            "gemv.K.mram": 256,
            "gemv.K.wram": 64,
            "gemv.order[0]": 2,
            "gemv.order[1]": 1,
        },
        prim="mtv",
        lower=cinmopt.eval_solution_lowerer(
            extra_infer_opts={"simulator": "cycle-accurate", "debug-pipeline": "true"}
        ),
    ),
    prim_config(
        system="cinm2",
        fn_name="mmtv_4MB",
        # A point I found via search with 512 evals
        label="mmtv",
        params={
            "dpus": 256,
            "tasklets": 16,
            "batch_gemv.B.mram": 1,
            "batch_gemv.B.wram": 1,
            "batch_gemv.M.mram": 2,
            "batch_gemv.M.wram": 2,
            "batch_gemv.K.mram": 128,
            "batch_gemv.K.wram": 128,
            "batch_gemv.order[0]": 2,
            "batch_gemv.order[1]": 3,
            "batch_gemv.order[2]": 1,
        },
        prim="mmtv",
        lower=cinmopt.eval_solution_lowerer(
            extra_infer_opts={"simulator": "cycle-accurate", "debug-pipeline": "true"}
        ),
    ),
    #
    # compile_run.Config(
    #     system="cinm2",
    #     fn_name="mtv_64MB",
    #     # This is just another CINM2 config that gives rise
    #     # to a partial reduction on the host. Not directly
    #     # comparable to the atim2048 optimum.
    #     label="partialred2048",
    #     params={
    #         "dpus": 2048,
    #         "tasklets": 8,
    #         "taskletCols": 1,
    #         "wramRow": 1,
    #         "wramCol": 1024,
    #         "dpuCols": 2,
    #         "mramRow": 4,
    #         "mramCol": 1024,
    #     },
    #     fn_module=source,
    #     prim="mtv",
    #     lower=cinmopt.eval_solution_lowerer(),
    # ),
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


def _split_one(source: pathlib.Path) -> bool:
    split_source(
        source, SPLIT_DIR
    )  # dict return value isn't JSON-picklable for doit's DB
    return True


def task_split():
    """Split every prim source CONFIGS draws on into one module per function.

    One subtask per source, derived from the configs, so a config for a prim
    that has not been benchmarked here before brings its own split with it.
    Each split writes every function its source holds and not just the
    configured ones: which exist is the source's business, and listing them up
    front is what lets task_compile depend on this by file."""
    for source in sorted({_prim_source(c.prim) for c in CONFIGS}):
        yield {
            "name": source.stem,
            "file_dep": [source],
            "targets": [SPLIT_DIR / f"{fn}.mlir" for fn in list_functions(source)],
            "actions": [(_split_one, [source])],
        }


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


def _by_label() -> dict[str, list[compile_run.Config]]:
    """CONFIGS grouped by label -- one plot per label, comparing whichever
    systems were configured for it."""
    by_label: dict[str, list[compile_run.Config]] = {}
    for c in CONFIGS:
        by_label.setdefault(c.label, []).append(c)
    return by_label


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

    # print({label: [c.system for c in confs] for label, confs in by_label.items()})
    for label, confs in _by_label().items():
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


# Two fixed colors, one per side of the comparison -- blue/orange stays
# distinguishable under every common form of color vision deficiency, and the
# side (not the bucket) is what the reader has to tell apart here: the buckets
# are already separated along x and named on the axis.
_MEASURED_COLOR = "#4C72B0"
_PREDICTED_COLOR = "#DD8452"


def _cost_csv(config: compile_run.Config) -> pathlib.Path:
    """The cost model's per-op prediction for a config, written into the
    compile dir by --upmem-annotate-costs (see cinmopt.eval_solution_lowerer)."""
    return config.dir(DATA_ROOT) / "ir" / "cost.csv"


def task_fidelity():
    """Print + plot the cost model's predicted breakdown against the measured
    one, bucket by bucket, with the relative error per bucket.

    The predicted side is ir/cost.csv (written at compile time), folded onto
    the measured breakdown's buckets by measurements.PREDICTED_TO_MEASURED;
    the measured side is the same net_breakdown_ms(by_kind=True) task_plot
    draws. Buckets with no counterpart on one side (e.g. "copy" -- host-side
    repacks the cost model doesn't model as their own category) show up as a
    lone bar rather than being dropped."""

    def action(out_path, csv_path, label, confs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames = {}
        for config in confs:
            df = measurements.breakdown_comparison(
                config.dir(DATA_ROOT) / "output", _cost_csv(config)
            )
            if df is None:
                print(f"{config.system}: no prediction or no results (run failed?)")
                continue
            frames[config.system] = df
            print(f"{config.system} ({label}):")
            for row in df.itertuples():
                err = (
                    "  n/a"
                    if row.rel_error != row.rel_error
                    else f"{row.rel_error:+.0%}"
                )
                print(
                    f"  {row.bucket:<20s} predicted={row.predicted_ms:8.3f}ms  "
                    f"measured={row.measured_ms:8.3f}ms  rel_err={err}"
                )
            totals = df[["predicted_ms", "measured_ms"]].sum()
            print(
                f"  {'TOTAL':<20s} predicted={totals.predicted_ms:8.3f}ms  "
                f"measured={totals.measured_ms:8.3f}ms  rel_err="
                f"{totals.predicted_ms / totals.measured_ms - 1:+.0%}"
            )

        if not frames:
            return

        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        pd.concat(
            [df.assign(system=system) for system, df in frames.items()],
            ignore_index=True,
        ).to_csv(csv_path, index=False)

        systems = list(frames)
        widest = max(len(df) for df in frames.values())
        # Two rows over a shared x: the absolute times can't show the error of
        # the small buckets (one transfer bucket dwarfs the rest on a linear
        # ms axis, and a log axis would misread as a magnitude comparison), so
        # the error gets its own row and its own scale.
        fig, axes = plt.subplots(
            2,
            len(systems),
            figsize=(max(4.5, 1.0 * widest) * len(systems), 6.0),
            squeeze=False,
            sharex="col",
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for col, system in enumerate(systems):
            df = frames[system]
            top, bottom = axes[0][col], axes[1][col]
            x = np.arange(len(df))

            top.bar(
                x - 0.2,
                df["measured_ms"],
                0.38,
                label="measured",
                color=_MEASURED_COLOR,
            )
            top.bar(
                x + 0.2,
                df["predicted_ms"],
                0.38,
                label="predicted",
                color=_PREDICTED_COLOR,
            )
            top.set_title(system)

            errs = df["rel_error"].fillna(0.0) * 100
            bottom.bar(
                x,
                errs,
                0.6,
                # Sign carries meaning, so it gets the color: warm where the
                # model over-predicts, cool where the hardware was slower than
                # the model thought -- same two colors as the row above, whose
                # bar is the taller one in each case.
                color=[_PREDICTED_COLOR if e >= 0 else _MEASURED_COLOR for e in errs],
            )
            bottom.axhline(0, color="black", linewidth=0.8)
            for xi, e, raw in zip(x, errs, df["rel_error"]):
                bottom.text(
                    xi,
                    e + (1 if e >= 0 else -1) * 0.03 * max(abs(errs).max(), 1.0),
                    "n/a" if raw != raw else f"{e:+.0f}%",  # NaN: nothing measured
                    ha="center",
                    va="bottom" if e >= 0 else "top",
                    fontsize=8,
                )
            bottom.margins(y=0.2)
            bottom.set_xticks(x)
            bottom.set_xticklabels(df["bucket"], rotation=45, ha="right")

            for ax in (top, bottom):
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                ax.set_axisbelow(True)

        axes[0][0].set_ylabel("time (ms)")
        axes[1][0].set_ylabel("relative error (%)")
        axes[0][-1].legend(loc="upper right")
        fig.suptitle(
            f"mtv_64MB predicted vs measured breakdown ({label})\n"
            "relative error = (predicted - measured) / measured"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")

    for label, confs in _by_label().items():
        out_path = HERE / "plots" / label / "fidelity.png"
        csv_path = HERE / "plots" / label / "fidelity.csv"
        yield {
            "name": label,
            "uptodate": [
                check_timestamp_unchanged(c.dir(DATA_ROOT) / "bench.done", "ctime")
                for c in confs
            ],
            "file_dep": [c.dir(DATA_ROOT) / "bench.done" for c in confs]
            + [_cost_csv(c) for c in confs],
            "targets": [out_path, csv_path],
            "actions": [(action, [out_path, csv_path, label, confs])],
        }

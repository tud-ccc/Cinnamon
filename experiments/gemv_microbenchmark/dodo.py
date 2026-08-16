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
  doit crosscheck   # price the dumped DPU programs with the Python reference
                    # cost model too, and report the gap (no hardware at all)
  doit ranking      # whether either cost model keeps the order between the
                    # same problem on 4 DPUs and on 2048
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
from cinm_experiments import compile_run, cinmopt, measurements, refmodel  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402
from cinm_experiments.paths import DEFAULT_CINM_OPT

DOIT_CONFIG = {
    "default_tasks": ["plot", "fidelity", "crosscheck", "ranking"],
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


# One legal schedule per primitive, for the functional pass -- see the entries
# they build at the end of CONFIGS. Each came out of a short cycle-accurate
# search over that primitive's 4MB size, pinned to 4 DPUs, and was then checked
# to translate to DPU C: these are points the pipeline accepts end to end and
# not merely points the constraint system admits.
FUNCTIONAL_POINTS: dict[str, dict[str, int]] = {
    "gemv": {
        "dpus": 4,
        "tasklets": 8,
        "gemv.M.mram": 256,
        "gemv.M.wram": 8,
        "gemv.K.mram": 128,
        "gemv.K.wram": 64,
        "gemv.order[0]": 1,
        "gemv.order[1]": 2,
        # gemv fuses the scaling of its result into the same launch, so that
        # op's own tiling is part of the point.
        "generic.D0.mram": 32,
        "generic.D0.wram": 8,
        "fuse.gemv->generic": 1,
    },
    "geva": {
        "dpus": 4,
        "tasklets": 8,
        "generic.D0.mram": 32768,
        "generic.D0.wram": 256,
    },
    "mmtv": {
        "dpus": 4,
        "tasklets": 8,
        "batch_gemv.B.mram": 16,
        "batch_gemv.B.wram": 4,
        "batch_gemv.M.mram": 8,
        "batch_gemv.M.wram": 4,
        "batch_gemv.K.mram": 256,
        "batch_gemv.K.wram": 8,
        "batch_gemv.order[0]": 1,
        "batch_gemv.order[1]": 3,
        "batch_gemv.order[2]": 2,
    },
    "mtv": {
        "dpus": 4,
        "tasklets": 8,
        "gemv.M.mram": 128,
        "gemv.M.wram": 4,
        "gemv.K.mram": 256,
        "gemv.K.wram": 32,
        "gemv.order[0]": 2,
        "gemv.order[1]": 1,
    },
    "red": {
        "dpus": 4,
        "tasklets": 8,
        "generic.D0.mram": 16384,
        "generic.D0.wram": 512,
    },
    "ttv": {
        "dpus": 4,
        "tasklets": 8,
        "generic.D0.mram": 32,
        "generic.D0.wram": 2,
        "generic.D1.mram": 4,
        "generic.D1.wram": 4,
        "generic.D2.mram": 256,
        "generic.D2.wram": 64,
        "generic.order[0]": 3,
        "generic.order[1]": 2,
        "generic.order[2]": 1,
    },
    "va": {
        "dpus": 4,
        "tasklets": 8,
        "elementwise.D0.mram": 32768,
        "elementwise.D0.wram": 128,
    },
}

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
    # prim_config(
    #     system="cinm2",
    #     fn_name="mmtv_4MB",
    #     # A point I found via search with 512 evals
    #     label="mmtv",
    #     params={
    #         "dpus": 256,
    #         "tasklets": 16,
    #         "batch_gemv.B.mram": 1,
    #         "batch_gemv.B.wram": 1,
    #         "batch_gemv.M.mram": 2,
    #         "batch_gemv.M.wram": 2,
    #         "batch_gemv.K.mram": 128,
    #         "batch_gemv.K.wram": 128,
    #         "batch_gemv.order[0]": 2,
    #         "batch_gemv.order[1]": 3,
    #         "batch_gemv.order[2]": 1,
    #     },
    #     prim="mmtv",
    #     lower=cinmopt.eval_solution_lowerer(
    #         extra_infer_opts={"simulator": "cycle-accurate", "debug-pipeline": "true"}
    #     ),
    # ),
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
    #
    # One configuration per primitive, at the smallest size each declares, for
    # the functional pass: what these check is that the pipeline computes the
    # right answer, so the point is coverage of the primitives and not of the
    # schedule space. Each was taken from a short cycle-accurate search and
    # then verified to reach DPU C, so a failure here is the kernel being
    # wrong rather than the configuration being unbuildable.
    #
    # Between them they cover the two shapes that DMA granularity makes
    # awkward: red's accumulator is one scalar per tasklet, which the pooled
    # write-back has to carry, and geva's coefficients are broadcast scalars,
    # which the short-read path has to.
    *[
        prim_config(
            system="cinm2",
            fn_name=f"{prim}_4MB",
            label="functional",
            params=params,
            prim=prim,
            # Each of these also dumps its lowered DPU programs to
            # <config dir>/cnmprog/*.cnmprog.json, with the C++ cost model's
            # own verdict on the same run in cnmprog/cost.csv beside them, so
            # the estimate can be checked against the Python reference model
            # (third-party/cnm-cost-model/Predictor/cnmprog.py). These are the
            # points that are known to run correctly on hardware, which makes
            # them the ones worth pricing twice.
            lower=cinmopt.with_program_dump(
                cinmopt.eval_solution_lowerer(
                    extra_infer_opts={
                        "simulator": "cycle-accurate",
                        "debug-pipeline": "true",
                    }
                )
            ),
        )
        for prim, params in FUNCTIONAL_POINTS.items()
    ],
    #
    # The same problems again at the other end of the machine. What the cost
    # model has to get right here is not the millisecond but the order: a
    # 2048-DPU working group should come out far ahead of the 4-DPU one, and
    # a model that agreed to 5% on every point while ranking these two the
    # wrong way round would be useless for search.
    #
    # The tasklet count is held at the 4-DPU points' 8 rather than searched,
    # so a pair differs in the working group and the tiling it forces, and in
    # nothing else. It also keeps the search out of the tasklets=1 corner,
    # which the space admits and the DPU codegen then rejects for red: one
    # tasklet makes the pooled accumulator write-back a 4-byte DMA, and a
    # transfer length has to be a multiple of 8.
    #
    # gemv is missing on purpose: its space at 2048 DPUs is empty (the fused
    # pair distributes 1024 rows, which cannot fill 2048 DPUs), so there is
    # no configuration to rank rather than a configuration that ranks badly.
    *[
        prim_config(
            system="cinm2",
            fn_name=f"{prim}_4MB",
            label="dpu2048",
            params={"dpus": 2048, "tasklets": 8},
            prim=prim,
            lower=cinmopt.with_program_dump(
                cinmopt.search_lowerer(
                    max_evals=64,
                    extra_infer_opts={
                        "simulator": "cycle-accurate",
                        "debug-pipeline": "true",
                    },
                )
            ),
        )
        for prim in FUNCTIONAL_POINTS
        if prim != "gemv"
    ],
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
            "name": _group_key(config) + ":" + config.system,
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
            "name": _group_key(config) + ":" + config.system,
            "uptodate": [check_timestamp_unchanged(compile_marker)],
            "file_dep": [compile_marker],
            "targets": [run_marker],
            "actions": [(_run_one, [config, run_marker])],
        }


def _group_key(config: compile_run.Config) -> str:
    """What makes two configs comparable: the same function under the same
    label, differing only in the system that compiled them.

    The function has to be part of it. A label is only unique within one
    function -- Config says so -- so keying on the label alone would put two
    primitives that happen to share one in the same plot, and give their
    compile tasks the same name."""
    return f"{config.fn_name}:{config.label}"


def _by_label() -> dict[str, list[compile_run.Config]]:
    """CONFIGS grouped into comparable sets -- one plot each, comparing
    whichever systems were configured for it."""
    by_label: dict[str, list[compile_run.Config]] = {}
    for c in CONFIGS:
        by_label.setdefault(_group_key(c), []).append(c)
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


# One fixed color per source of a number: neutral for the machine, a hue each
# for the two models. Not the fidelity plot's blue for the measured bar -- next
# to the reference model's purple it is barely a different color (7 OKLab units
# apart to normal vision, 2 under protanopia, against ~20 for the pair below),
# and the two would be read as one series.
_HW_COLOR = "#4D4D4D"
_CPP_COLOR = "#DD8452"
_REF_COLOR = "#8172B3"


def _dump_dir(config: compile_run.Config) -> pathlib.Path:
    """Where with_program_dump put this config's DPU programs and the C++
    cost model's verdict on them (see cinmopt.with_program_dump)."""
    return config.dir(DATA_ROOT) / "cnmprog"


def _ms(value):
    return f"{'n/a':>8}  " if value is None else f"{value:8.3f}ms"


def _pct(value, base):
    """`value` against `base` in percent, None if either is missing."""
    if value is None or not base:
        return None
    return (value / base - 1.0) * 100


def _gather(confs, *, quiet: bool = False):
    """Both cost models' verdict on each config's dumped DPU programs, and
    the launch time the hardware took for it.

    Returns (comparisons, measured-by-name). The measurement is optional --
    reading it needs no hardware, only a bench run that may not have
    happened -- so a name can map to None."""
    comparisons, measured = [], {}
    for config in confs:
        c = refmodel.compare(_dump_dir(config), name=config.fn_name)
        if c is None:
            if not quiet:
                print(
                    f"{config.fn_name}: no dumps in {_dump_dir(config)} "
                    "(compiled before program dumping? `doit forget compile`)"
                )
            continue
        comparisons.append(c)
        measured[c.name] = measurements.launch_time_ms(config.dir(DATA_ROOT) / "output")
        if quiet:
            continue
        ratio = "  n/a" if c.ratio is None else f"{c.ratio:.2f}x"
        print(
            f"{c.name:<12s} hw={_ms(measured[c.name])}  cpp={_ms(c.cpp_launch_ms)}  "
            f"ref={_ms(c.ref_launch_ms)}  ref/cpp={ratio}  "
            f"(+{c.overhead_ms:.3f}ms launch, "
            f"{len(c.kernels)} kernel{'s' if len(c.kernels) != 1 else ''})"
        )
        for k in c.kernels:
            if k.ms is None:
                print(f"    {k.kernel}: unpriced -- {k.error}")
            elif len(c.kernels) > 1:
                print(f"    {k.kernel}: {k.ms:.3f}ms")
    return comparisons, measured


def _grouped_bars(ax, x, series, fmt, floor=0.0):
    """One labeled bar per series per x position, side by side. A missing
    value is labeled n/a at the baseline rather than drawn as a zero-height
    bar, which would read as a measurement of zero. Returns the largest
    magnitude drawn, for callers that need to size headroom."""
    import numpy as np

    width = 0.8 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    span = max(
        (abs(v) for _, values, _ in series for v in values if v is not None),
        default=1.0,
    )
    # A label sits a fixed distance clear of its bar's end -- a fraction of
    # the tallest bar on a linear axis, a fraction of its own height on a log
    # one, where a constant offset would leave the short bars' labels floating
    # a decade above them.
    log = ax.get_yscale() == "log"

    def clear_of(end, above):
        if log:
            return end * 1.08 if above else end / 1.08
        return end + (0.03 * span if above else -0.03 * span)

    for (label, values, color), dx in zip(series, offsets):
        ax.bar(
            x + dx,
            [v if v is not None else floor for v in values],
            width * 0.9,
            label=label,
            color=color,
        )
        for xi, v in zip(x, values):
            above = v is None or v >= floor
            ax.text(
                xi + dx,
                clear_of(v if v is not None else floor, above),
                "n/a" if v is None else fmt.format(v),
                ha="center",
                va="bottom" if above else "top",
                fontsize=7,
            )
    return span


# Each group of configs the cross-check draws a figure for: the label they
# carry in CONFIGS, and what that label means in the figure's title.
_CROSSCHECK_GROUPS = {
    "functional": "4 DPUs",
    "dpu2048": "2048 DPUs, tiling searched",
}


def task_crosscheck():
    """Price each point's dumped DPU programs with the Python reference cost
    model and print/plot the gap to the C++ estimate of the same programs,
    with the launch time the hardware actually took beside them where the
    point has been benchmarked. One figure per working-group size.

    The question is how far apart these three are, not whether they agree, so
    the report is the times and the deviations -- nothing here asserts a
    tolerance. A program the reference model refuses to price (it rejects
    what it has no calibrated entry for rather than guessing) is reported as
    such and left out of the ratio, since it is a hole in the comparison
    rather than a difference of zero.

    The measured counterpart of a kernel cost is the launch time, the same
    pairing task_fidelity uses (measurements.PREDICTED_TO_MEASURED). It is
    optional: this task otherwise needs no hardware, so a point that has
    never been benchmarked loses its bar and not its row."""

    def action(out_path, csv_path, confs, group):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        comparisons, measured = _gather(confs)
        if not comparisons:
            return

        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.DataFrame(refmodel.comparison_rows(comparisons))
        df.insert(
            df.columns.get_loc("cpp_total_ms") + 1,
            "measured_launch_ms",
            df["name"].map(measured),
        )
        df.to_csv(csv_path, index=False)

        priced = [c for c in comparisons if c.ratio is not None]
        print(
            f"priced {len(priced)}/{len(comparisons)} configs"
            + (
                f"; ref/cpp in [{min(c.ratio for c in priced):.2f}, "
                f"{max(c.ratio for c in priced):.2f}]"
                if priced
                else ""
            )
        )
        for label, values in (
            ("cpp", [_pct(c.cpp_launch_ms, measured[c.name]) for c in comparisons]),
            ("ref", [_pct(c.ref_launch_ms, measured[c.name]) for c in comparisons]),
        ):
            seen = [v for v in values if v is not None]
            if seen:
                print(
                    f"vs hardware: {label} in "
                    f"[{min(seen):+.0f}%, {max(seen):+.0f}%] over {len(seen)} points"
                )

        x = np.arange(len(comparisons))
        fig, (top, bottom) = plt.subplots(
            2,
            1,
            # Three bars per config, each labeled: the width has to grow with
            # the config count or the labels of adjacent bars run together.
            figsize=(max(6.0, 1.7 * len(comparisons)), 6.0),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        # Measured first: it is the yardstick the two estimates are read
        # against, not a third opinion. Both estimates carry the C++ model's
        # launch overhead, including the reference model's, which has none of
        # its own -- a measured launch pays it whichever engine priced the
        # program, so leaving it off one side would be comparing two
        # different things (see refmodel.Comparison).
        tallest = _grouped_bars(
            top,
            x,
            [
                (
                    "measured (launch)",
                    [measured[c.name] for c in comparisons],
                    _HW_COLOR,
                ),
                ("C++ cost model", [c.cpp_launch_ms for c in comparisons], _CPP_COLOR),
                (
                    "Python reference model",
                    [c.ref_launch_ms for c in comparisons],
                    _REF_COLOR,
                ),
            ],
            "{:.2f}",
        )

        # Absolute times can't show a gap of a few percent -- two bars of the
        # same height -- so the gap gets its own zero-centred panel. Measured
        # is the baseline where there is one, which puts both models' error on
        # one scale and leaves the distance between the two bars reading as
        # the model-to-model gap; with nothing measured, that gap is all there
        # is to show.
        if any(v is not None for v in measured.values()):
            baseline = "measured"
            deviations = [
                (
                    "C++ cost model",
                    [_pct(c.cpp_launch_ms, measured[c.name]) for c in comparisons],
                    _CPP_COLOR,
                ),
                (
                    "Python reference model",
                    [_pct(c.ref_launch_ms, measured[c.name]) for c in comparisons],
                    _REF_COLOR,
                ),
            ]
        else:
            baseline = "C++ cost model"
            deviations = [
                (
                    "Python reference model",
                    [_pct(c.ref_ms, c.cpp_ms) for c in comparisons],
                    _REF_COLOR,
                )
            ]
        _grouped_bars(bottom, x, deviations, "{:+.1f}%")
        bottom.axhline(0.0, color="black", linewidth=0.8)
        bottom.margins(y=0.3)
        bottom.set_xticks(x)
        bottom.set_xticklabels([c.name for c in comparisons], rotation=45, ha="right")

        for ax in (top, bottom):
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
        # Headroom for the value labels, and for the legend to sit over the
        # bars rather than on top of the tallest one.
        top.set_ylim(0, tallest * 1.45)
        top.set_ylabel("launch time (ms)")
        bottom.set_ylabel(f"deviation from\n{baseline} (%)")
        top.legend(loc="upper left")
        fig.suptitle(
            f"Two cost models and the machine, on the same DPU programs ({group})"
            + (
                "\nn/a: the reference model declines to price the program"
                if len(priced) < len(comparisons)
                else ""
            )
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")

    for label, group in _CROSSCHECK_GROUPS.items():
        confs = [c for c in CONFIGS if c.label == label]
        if not confs:
            continue
        out_path = HERE / "plots" / f"crosscheck_{label}.png"
        csv_path = HERE / "plots" / f"crosscheck_{label}.csv"
        yield {
            "name": label,
            "uptodate": [
                check_timestamp_unchanged(c.dir(DATA_ROOT) / "compile.done", "ctime")
                for c in confs
            ],
            "file_dep": [c.dir(DATA_ROOT) / "compile.done" for c in confs]
            + _optional_deps(confs),
            "targets": [out_path, csv_path],
            "actions": [(action, [out_path, csv_path, confs, group])],
        }


def _optional_deps(confs) -> list[pathlib.Path]:
    """Inputs a cross-check reads that a config need not have yet: the dumps'
    cost.csv, which is what changes when they are re-dumped without a
    recompile (a new emitter, say), and bench.done, which is what changes
    when the measured bar does. Depending on them where they exist makes
    those two events redraw; requiring them would make the whole task wait on
    hardware it otherwise does not need."""
    return [d / "cost.csv" for c in confs if (d := _dump_dir(c)).is_dir()] + [
        b for c in confs if (b := c.dir(DATA_ROOT) / "bench.done").exists()
    ]


def task_ranking():
    """How well each cost model preserves the order of the same problem run
    on 4 DPUs and on 2048, which is what a model used for search has to get
    right even where its absolute error does not matter.

    Two readings of that. Per prim: the 4-DPU/2048-DPU speedup each source
    reports, so a model that has the winner right but the margin wrong is
    visibly different from one that has the winner wrong. Over the whole set:
    the fraction of measured pairs whose order a model reproduces, counting
    every pair of points and not just the two that share a prim -- the
    quantity a search actually consumes, since it compares configurations
    across problems too.

    Points the hardware has not run are left out of both: without a measured
    order there is nothing to preserve."""

    def action(out_path, csv_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        by_label = {}
        for label in _CROSSCHECK_GROUPS:
            confs = [c for c in CONFIGS if c.label == label]
            comparisons, measured = _gather(confs, quiet=True)
            # The launch quantities, not the bare programs: what a bigger
            # working group buys you is exactly what the overhead it costs
            # eats into, so a speedup computed without it is not the one the
            # hardware reports.
            by_label[label] = {
                c.name: {
                    "hw": measured[c.name],
                    "cpp": c.cpp_launch_ms,
                    "ref": c.ref_launch_ms,
                }
                for c in comparisons
            }

        sources = ("hw", "cpp", "ref")
        # A prim is comparable only if both of its points priced and ran: a
        # speedup needs both ends, from the same source.
        prims = [
            name
            for name in by_label.get("functional", {})
            if name in by_label.get("dpu2048", {})
            and all(
                by_label[lbl][name][s]
                for lbl in ("functional", "dpu2048")
                for s in sources
            )
        ]
        if not prims:
            print("ranking: no prim has both working-group sizes measured and priced")
            return

        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        speedup = {
            s: [by_label["functional"][p][s] / by_label["dpu2048"][p][s] for p in prims]
            for s in sources
        }

        # Pairwise order agreement over every point of every group, which is
        # what a search consumes -- not only the within-prim pairs the figure
        # draws. Ties in the measured order would have no order to preserve;
        # there are none here (distinct kernels, distinct times).
        points = [
            (f"{name}@{label}", v)
            for label, entries in by_label.items()
            for name, v in entries.items()
            if all(v[s] for s in sources)
        ]
        agreement = {}
        for s in ("cpp", "ref"):
            pairs = [
                (a, b) for i, (_, a) in enumerate(points) for _, b in points[i + 1 :]
            ]
            kept = [
                (a[s] < b[s]) == (a["hw"] < b["hw"])
                for a, b in pairs
                if a["hw"] != b["hw"]
            ]
            agreement[s] = (sum(kept), len(kept))
            print(
                f"pairwise order agreement with hardware ({s}): "
                f"{sum(kept)}/{len(kept)} pairs"
            )

        rows = []
        for i, p in enumerate(prims):
            row = {"name": p}
            for label in ("functional", "dpu2048"):
                for s in sources:
                    row[f"{label}_{s}_ms"] = by_label[label][p][s]
            for s in sources:
                row[f"speedup_{s}"] = speedup[s][i]
            rows.append(row)
            print(
                f"{p:<12s} speedup 4->2048  hw={speedup['hw'][i]:6.2f}x  "
                f"cpp={speedup['cpp'][i]:6.2f}x  ref={speedup['ref'][i]:6.2f}x"
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        x = np.arange(len(prims))
        fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(prims)), 5.0))
        # Log scale, floor at 1: a speedup and a slowdown of the same factor
        # are then the same distance from the no-change line, and the two ends
        # of the machine differ by enough that a linear axis would flatten the
        # small ones against it. Set before the bars so their labels are
        # placed against the scale they will be drawn on.
        ax.set_yscale("log")
        tallest = max(v for s in sources for v in speedup[s])
        ax.set_ylim(1.0, tallest * 4)
        _grouped_bars(
            ax,
            x,
            [
                ("measured (launch)", speedup["hw"], _HW_COLOR),
                ("C++ cost model", speedup["cpp"], _CPP_COLOR),
                ("Python reference model", speedup["ref"], _REF_COLOR),
            ],
            "{:.1f}x",
            floor=1.0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(prims, rotation=45, ha="right")
        ax.set_ylabel("speedup, 4 DPUs -> 2048 DPUs")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        # Above the bars rather than over them: on a log axis with an order of
        # magnitude between the series there is no corner a legend can sit in.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False)
        fig.suptitle(
            "Does the cost model keep the order when the machine grows?\n"
            + "  ".join(
                f"{s}: {n}/{d} pairs ordered as measured"
                for s, (n, d) in agreement.items()
            )
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")

    confs = [c for c in CONFIGS if c.label in _CROSSCHECK_GROUPS]
    out_path = HERE / "plots" / "ranking.png"
    csv_path = HERE / "plots" / "ranking.csv"
    yield {
        "name": "dpu4_vs_dpu2048",
        "uptodate": [
            check_timestamp_unchanged(c.dir(DATA_ROOT) / "compile.done", "ctime")
            for c in confs
        ],
        "file_dep": [c.dir(DATA_ROOT) / "compile.done" for c in confs]
        + _optional_deps(confs),
        "targets": [out_path, csv_path],
        "actions": [(action, [out_path, csv_path])],
    }

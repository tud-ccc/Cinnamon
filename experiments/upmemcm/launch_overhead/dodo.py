"""doit tasks for the launch-overhead microbenchmark.

What a launch costs when the kernel costs nothing: the same empty DPU program
(launch_dpu.c) on every set size, so the only thing varying across the points
is the number of DPUs. The cost model prices instructions and DMAs and would
predict zero for all of these, which makes the whole measurement the term it
is missing -- the one that dominates the gemv_microbenchmark points once the
kernel gets short (see that experiment's crosscheck figures).

Timed calls are the ones the generated host code makes, so the numbers are
comparable to the runtime's own launch.csv rows.

Usage:
  doit list   # show tasks
  doit make   # build bin/launch_dpu + bin/launch_bench
  doit bench  # run the sweep on hardware -> results.csv
  doit plot   # fit the candidate forms and draw plots/launch_overhead.png
  doit        # all three
"""

import os
import pathlib

from doit.tools import Interactive, config_changed

DOIT_CONFIG = {"default_tasks": ["make", "bench", "plot"], "verbosity": 2}

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
PLOT = HERE / "plots" / "launch_overhead.png"

# Powers of two from one DPU to the whole machine. Rank granularity is 64, so
# everything below that shares a rank and everything above adds them: the
# sweep has to cross that boundary for a per-rank term to be separable from a
# per-DPU one, and to reach both of the sizes gemv_microbenchmark uses (4 and
# 2048).
DPUS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Must match what the DPU binary is built with; the Makefile takes it from
# here. 8 is what the benchmarked points use.
TASKLETS = 8

ITERS = 50
WARMUP = 5


def task_make():
    """Build the empty DPU program and the host sweep."""
    return {
        "file_dep": [
            HERE / "launch_dpu.c",
            HERE / "launch_bench.cpp",
            HERE / "Makefile",
        ],
        "targets": [HERE / "bin" / "launch_dpu", HERE / "bin" / "launch_bench"],
        "uptodate": [config_changed(str(TASKLETS))],
        "actions": [f"make -C {HERE} TASKLETS={TASKLETS}"],
    }


def task_bench():
    """Run the sweep on hardware. Needs the DPUs free: every size up to the
    whole machine is allocated in turn, and a size another job is holding is
    skipped with a warning rather than failing the sweep."""
    return {
        "file_dep": [HERE / "bin" / "launch_dpu", HERE / "bin" / "launch_bench"],
        "targets": [RESULTS],
        "uptodate": [
            config_changed(
                {"dpus": DPUS, "tasklets": TASKLETS, "iters": ITERS, "warmup": WARMUP}
            )
        ],
        "actions": [
            Interactive(
                str(HERE / "bin" / "launch_bench"),
                cwd=str(HERE),
                env={
                    **os.environ,
                    "LAUNCH_DPUS": ",".join(str(d) for d in DPUS),
                    "LAUNCH_TASKLETS": str(TASKLETS),
                    "LAUNCH_ITERS": str(ITERS),
                    "LAUNCH_WARMUP": str(WARMUP),
                    "LAUNCH_BINARY": str(HERE / "bin" / "launch_dpu"),
                    "LAUNCH_CSV_OUT": str(RESULTS),
                },
            )
        ],
    }


def task_plot():
    """Fit the candidate overhead forms to the sweep and plot the winner."""

    def action():
        import sys

        sys.path.insert(0, str(HERE))
        import analyze

        analyze.report(RESULTS, PLOT)

    return {
        "file_dep": [RESULTS, HERE / "analyze.py"],
        "targets": [PLOT],
        "actions": [action],
    }

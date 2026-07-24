from doit.tools import config_changed
from doit.tools import Interactive
import os

"""doit tasks for the scatter_cost microbenchmark.

Only wraps the two cheap steps (build, plot) -- generating results.csv means
running ./bin/scatter_bench on real UPMEM hardware (see README.md for the
SCATTER_* env vars), which is a manual step done separately, not modeled here.

Usage:
  doit list   # show tasks
  doit make   # build bin/scatter_dpu + bin/scatter_bench
  doit plot   # analyze results.csv -> plots/
  doit        # both (default task)
"""

DOIT_CONFIG = {"default_tasks": ["make", "plot"], "verbosity": 2}

fns = ["block", "broadcast", "sg", "gather"]


def task_make():
    """Build the DPU kernel and host benchmark binary."""
    return {
        "file_dep": ["scatter_dpu.c", "scatter_bench.cpp", "Makefile"],
        "targets": ["bin/scatter_dpu", *[f"bin/scatter_bench_{fn}" for fn in fns]],
        "actions": ["make"],
    }


env_vars = ["SCATTER_ITERS", "SCATTER_WARMUP"]


def task_bench():
    dense = {"SCATTER_DENSE": ""}
    bench_env = {
        "broadcast": dense,
        "sg": dense,
        "block": dense,
        "gather": dense,
    }
    """Build the DPU kernel and host benchmark binary."""
    for fn in fns:
        yield {
            "name": f"{fn}",
            "targets": [f"plots/{fn}/results.csv"],
            "file_dep": ["bin/scatter_dpu", f"bin/scatter_bench_{fn}"],
            "actions": [
                Interactive(
                    f"bin/scatter_bench_{fn}",
                    env={
                        **os.environ,
                        **bench_env.get(fn, {}),
                        "SCATTER_CSV_OUT": f"plots/{fn}/results.csv",
                        "SCATTER_ITERS": '10',
                    },
                )
            ],
        }


def task_plot():
    bench_splits = {
        "broadcast": "--split 1023 --split-dim block_size",
        "sg": "--split 1023 2047 --split-dim block_size blocks_per_dpu",
        # "sg": "--split 9 13 --split-dim blocks_per_dpu",
        # "sg": "--split 64 --split-dim num_dpus",
        "block": "--split block_size 1023",
    }
    """Analyze results.csv and write plots to plots/."""
    for fn in fns:
        yield {
            "name": f"{fn}",
            "file_dep": ["analyze.py", f"plots/{fn}/results.csv"],
            "targets": [f"plots/{fn}/regression_fit.png"],
            "uptodate": [config_changed(bench_splits.get(fn, ''))],
            "actions": [
                f"python3 analyze.py plots/{fn}/results.csv --out-dir plots/{fn} {bench_splits.get(fn, '')} | tee plots/{fn}/log.log"
            ],
        }

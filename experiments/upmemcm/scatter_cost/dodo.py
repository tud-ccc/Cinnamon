from doit.tools import config_changed
from os import getenv

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


def task_make():
    """Build the DPU kernel and host benchmark binary."""
    return {
        "file_dep": ["scatter_dpu.c", "scatter_bench.cpp", "Makefile"],
        "targets": ["bin/scatter_dpu", "bin/scatter_bench"],
        "actions": ["make"],
    }


env_vars = ["SCATTER_DENSE", "SCATTER_ITERS", "SCATTER_WARMUP"]


def task_bench():
    """Build the DPU kernel and host benchmark binary."""
    return {
        "targets": ["results.csv"],
        "uptodate": [config_changed({var: getenv(var) for var in env_vars})],
        "file_dep": ["bin/scatter_dpu", "bin/scatter_bench"],
        "actions": ["bin/scatter_bench"],
    }


def task_plot():
    """Analyze results.csv and write plots to plots/."""
    return {
        "file_dep": ["results.csv", "analyze.py"],
        "targets": ["plots/regression_fit.png"],
        "actions": ["python3 analyze.py results.csv --out-dir plots --dpu-split 32"],
    }

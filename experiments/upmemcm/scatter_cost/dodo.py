from doit.tools import config_changed
from doit.tools import Interactive
import pathlib
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
        out_path = pathlib.Path(f"plots/{fn}")
        out_path.mkdir(parents=True, exist_ok=True)

        yield {
            "name": f"{fn}",
            "targets": [ out_path / "results.csv"],
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
    splits = "--split block_size 1023 --split num_dpus 16 24 64 128 256 384"

    bench_splits = {
        "sg": splits,#"--split block_size 1023",
        "gather": "--split block_size 1023 --split num_dpus 16 24 64 128",
        "broadcast": splits,
        "block": splits,
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

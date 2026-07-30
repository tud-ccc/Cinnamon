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
  doit agg    # aggregate results.csv -> results_agg.csv (median per config)
  doit plot   # analyze results.csv -> plots/
  doit        # make + agg + plot (default task)
"""

DOIT_CONFIG = {"default_tasks": ["make", "agg", "plot"], "verbosity": 2}

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
            "targets": [out_path / "results.csv"],
            "file_dep": ["bin/scatter_dpu", f"bin/scatter_bench_{fn}"],
            "actions": [
                Interactive(
                    f"bin/scatter_bench_{fn}",
                    env={
                        **os.environ,
                        **bench_env.get(fn, {}),
                        "SCATTER_CSV_OUT": f"plots/{fn}/results.csv",
                        "SCATTER_ITERS": "5",
                        "SCATTER_WARMUP": "1",
                    },
                )
            ],
        }


def _aggregate_one(in_path, out_path):
    import pandas as pd

    DIMS = ["num_dpus", "blocks_per_dpu", "block_size"]

    df = pd.read_csv(in_path)
    df["ms"] = df["ns"] / 1e6
    agg = df.groupby(DIMS)["ms"].median().reset_index()
    agg.to_csv(out_path, index=False)
    # print(f"{in_path}: {len(df):,} rows -> {out_path}: {len(agg):,} configs")


def task_agg():
    """Aggregate each results.csv (one row/iteration) into results_agg.csv
    (median "ms" per config) -- cached by doit so re-opening explore_3d.ipynb
    doesn't re-parse the full per-iteration CSV and re-run groupby every time,
    only when results.csv actually changed."""
    for fn in fns:
        yield {
            "name": fn,
            "file_dep": [f"plots/{fn}/results.csv"],
            "targets": [f"plots/{fn}/results_agg.csv"],
            "actions": [
                (
                    _aggregate_one,
                    [f"plots/{fn}/results.csv", f"plots/{fn}/results_agg.csv"],
                )
            ],
        }


def task_plot():
    splits = "--split block_size 1023 --split num_dpus 16 24 64 128 256 384"

    bench_splits = {
        # --mlp only for sg: it's the one regime where the per-region
        # polynomial families leave real accuracy on the table (see
        # analyze.py's fit_mlp/MLP_TEMPLATE docstrings and
        # residual_vs_blocks.py) -- block/broadcast/gather already fit well
        # with the (much cheaper, C++-exportable) polynomial templates.
        "sg": splits + " --mlp",
        "gather": "--split block_size 1023 --split num_dpus 16 24 64 128",
        "broadcast": splits,
        "block": splits,
    }
    """Analyze results.csv and write plots to plots/."""
    for fn in fns:
        yield {
            "name": f"{fn}",
            "file_dep": ["analyze.py", f"plots/{fn}/results_agg.csv"],
            "targets": [f"plots/{fn}/regression_fit.png"],
            "uptodate": [config_changed(bench_splits.get(fn, ""))],
            "actions": [
                f"python3 analyze.py plots/{fn}/results_agg.csv --out-dir plots/{fn} {bench_splits.get(fn, '')} | tee plots/{fn}/log.log"
            ],
        }

from doit.tools import config_changed
from doit.tools import Interactive
import pathlib
import os

"""doit tasks for the scatter_cost microbenchmark.

Only wraps the two cheap steps (build, plot) -- generating results.csv means
running ./bin/scatter_bench on real UPMEM hardware (see README.md for the
SCATTER_* env vars), which is a manual step done separately, not modeled here.

Usage:
  doit list        # show tasks
  doit make        # build bin/scatter_dpu + bin/scatter_bench
  doit agg         # aggregate results.csv -> results_agg.csv (median per config)
  doit plot        # analyze results.csv -> plots/
  doit model_tree  # fit model trees (model_tree.py) -> plots/*/model_tree*
  doit             # make + agg + plot + model_tree (default task)
"""

DOIT_CONFIG = {
    "default_tasks": ["make", "agg", "plot", "model_tree"],
    "verbosity": 2,
}

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


def task_model_tree():
    # sg: the deep tree with product splits -- the polynomial leaf models
    # only get competitive with the hand-split hybrid there once the tree
    # can cut on bytes-per-DPU-style products and go deeper (relRMSE 19.5%
    # manual hybrid vs. 14.7% at depth 6). The 2-dimension benches are
    # already better than their manual hybrids at depth 4, and restricting
    # to raw cuts keeps every region a plain box (and, empirically, even
    # beats product splits slightly there -- greedy trees aren't monotone
    # in their candidate set).
    tree_args = {
        "sg": "--max-depth 6 --split-terms all",
        "gather": "--max-depth 4 --split-terms raw",
        "broadcast": "--max-depth 4 --split-terms raw",
        "block": "--max-depth 4 --split-terms raw",
    }
    """Fit a model tree (auto-inferred regime cuts, see model_tree.py) per
    benchmark; tree/cuts/C++ report goes to plots/*/model_tree.log."""
    for fn in fns:
        yield {
            "name": f"{fn}",
            # analyze.py is a real dep: model_tree.py imports its problem
            # definition, templates, and plot helpers.
            "file_dep": ["model_tree.py", "analyze.py", f"plots/{fn}/results_agg.csv"],
            "targets": [
                f"plots/{fn}/model_tree_partition.png",
                f"plots/{fn}/model_tree_fit.png",
                f"plots/{fn}/model_tree.log",
            ],
            "uptodate": [config_changed(tree_args[fn])],
            "actions": [
                f"python3 model_tree.py plots/{fn}/results_agg.csv --out-dir plots/{fn} {tree_args[fn]} | tee plots/{fn}/model_tree.log"
            ],
        }

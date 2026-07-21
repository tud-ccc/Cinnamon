"""Turn the raw per-iteration benchmark CSVs written by a bench_* binary
(scatter/gather/alloc/free/total/... one file per measurement type in an
output/ dir) into net compute+transfer time, matching the definition used
throughout the paper pipeline (paperplots/plot_best_configs.py): total
elapsed time minus alloc/free overhead, averaged over iterations."""
from __future__ import annotations

from typing import Union
import pathlib

from math import isnan
import pandas as pd

from .compile_run import RunResult


def _iter_col(df: pd.DataFrame) -> str:
    return "iter" if "iter" in df.columns else "iteration"


def _csv_type(path: pathlib.Path) -> str:
    return path.stem.rsplit("_", 1)[-1]

def _output_dir(obj: Union[pathlib.Path, RunResult]) -> pathlib.Path:
  if isinstance(obj, RunResult):
    return pathlib.Path(obj.output_dir)
  return pathlib.Path(obj)

def net_time_ms(output_dir : Union[pathlib.Path, RunResult]) -> float | None:
    """Mean net time in ms over all iterations recorded in output_dir, or
    None if no total.csv-type file is present."""
    total_df = None
    alloc_ns = pd.Series(dtype=float)
    free_ns = pd.Series(dtype=float)

    for csv_path in _output_dir(output_dir).glob("*.csv"):
        t = _csv_type(csv_path)
        df = pd.read_csv(csv_path)
        df = df.rename(columns={_iter_col(df): "iteration"})
        if t == "total":
            total_df = df
        elif t == "alloc":
            alloc_ns = df.groupby("iteration")["elapsed_ns"].sum()
        elif t == "free":
            free_ns = df.groupby("iteration")["elapsed_ns"].sum()

    if total_df is None or total_df.empty:
        return None

    net = total_df["elapsed_ns"] - total_df["iteration"].map(alloc_ns).fillna(0) \
        - total_df["iteration"].map(free_ns).fillna(0)
    return float(net.mean()) / 1e6


def _sum_time_ms(output_dir : Union[pathlib.Path, RunResult], csv_type: str) -> float | None:
    """Mean per-iteration total time in ms spent in the given csv_type
    (summed over however many calls of that type happen within an
    iteration), or None if no matching csv-type file is present."""
    for csv_path in _output_dir(output_dir).glob("*.csv"):
        if _csv_type(csv_path) != csv_type:
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={_iter_col(df): "iteration"})
        mean = float(df.groupby("iteration")["elapsed_ns"].sum().mean())
        if isnan(mean):
          return None
        return float(mean) / 1e6
    return None


def launch_time_ms(output_dir : Union[pathlib.Path, RunResult]) -> float | None:
    """Mean kernel-launch time in ms over all iterations recorded in
    output_dir, or None if no launch.csv-type file is present."""
    return _sum_time_ms(output_dir, "launch")


def scatter_time_ms(output_dir : Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total host->DPU scatter time in ms per iteration, or None if no
    scatter.csv-type file is present."""
    return _sum_time_ms(output_dir, "scatter")


def gather_time_ms(output_dir : Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total DPU->host gather time in ms per iteration, or None if no
    gather.csv-type file is present."""
    return _sum_time_ms(output_dir, "gather")


def copy_time_ms(output_dir : Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total host-side memrefCopy time in ms per iteration (e.g. the
    strided repack copies feeding upmem.scatter buffers), or None if no
    copy.csv-type file is present."""
    return _sum_time_ms(output_dir, "copy")


# Fixed stacking/legend order for net_breakdown_ms -- keep display code (e.g.
# experiment.py's stacked bar chart) agreeing on category order and colors
# without recomputing it.
NET_BREAKDOWN_CATEGORIES = ["scatter", "gather", "copy", "launch", "unaccounted"]


def net_breakdown_ms(output_dir : Union[pathlib.Path, RunResult]) -> dict[str, float] | None:
    """Split net_time_ms (total - alloc - free) into scatter/gather/copy/
    launch time plus whatever's left over as "unaccounted" -- host-side work
    that happens outside any instrumented runtime call (e.g. computation in
    the generated host loop nest). None if no total.csv-type file is
    present."""
    net = net_time_ms(output_dir)
    if net is None:
        return None
    scatter = scatter_time_ms(output_dir) or 0.0
    gather = gather_time_ms(output_dir) or 0.0
    copy = copy_time_ms(output_dir) or 0.0
    launch = launch_time_ms(output_dir) or 0.0
    unaccounted = net - scatter - gather - copy - launch
    return {"scatter": scatter, "gather": gather, "copy": copy, "launch": launch,
            "unaccounted": unaccounted}


def results_to_frame(results: list[RunResult]) -> pd.DataFrame:
    """Turn a list of compile_run.RunResult into a DataFrame with one row per
    successfully-run config: fn_name, label, every config param, net_time_ms."""
    rows = []
    for res in results:
        if not res.ok:
            continue
        t = net_time_ms(res.output_dir)
        if t is None:
            continue
        cfg = res.compiled.config
        rows.append({"fn_name": cfg.fn_name, "label": cfg.label, **cfg.params, "net_time_ms": t})
    if not rows:
        # pd.DataFrame([]) has no columns at all, since there are no rows to
        # infer them from -- guarantee the fixed columns so callers can rely
        # on e.g. df["fn_name"] / df.drop(columns=["label"]) even when every
        # run in `results` failed or was never attempted.
        return pd.DataFrame(columns=["fn_name", "label", "net_time_ms"])
    return pd.DataFrame(rows)

"""Turn the raw per-iteration benchmark CSVs written by a bench_* binary
(scatter/gather/alloc/free/total/... one file per measurement type in an
output/ dir) into net compute+transfer time, matching the definition used
throughout the paper pipeline (paperplots/plot_best_configs.py): total
elapsed time minus alloc/free overhead, averaged over iterations."""
from __future__ import annotations

import pathlib

import pandas as pd

from .compile_run import RunResult


def _iter_col(df: pd.DataFrame) -> str:
    return "iter" if "iter" in df.columns else "iteration"


def _csv_type(path: pathlib.Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def net_time_ms(output_dir: pathlib.Path) -> float | None:
    """Mean net time in ms over all iterations recorded in output_dir, or
    None if no total.csv-type file is present."""
    total_df = None
    alloc_ns = pd.Series(dtype=float)
    free_ns = pd.Series(dtype=float)

    for csv_path in pathlib.Path(output_dir).glob("*.csv"):
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
    return pd.DataFrame(rows)

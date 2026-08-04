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


def net_time_ms(output_dir: Union[pathlib.Path, RunResult]) -> float | None:
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

    net = (
        total_df["elapsed_ns"]
        - total_df["iteration"].map(alloc_ns).fillna(0)
        - total_df["iteration"].map(free_ns).fillna(0)
    )
    return float(net.mean()) / 1e6


def _sum_time_ms(
    output_dir: Union[pathlib.Path, RunResult], csv_type: str
) -> float | None:
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


def launch_time_ms(output_dir: Union[pathlib.Path, RunResult]) -> float | None:
    """Mean kernel-launch time in ms over all iterations recorded in
    output_dir, or None if no launch.csv-type file is present."""
    return _sum_time_ms(output_dir, "launch")


def scatter_time_ms(output_dir: Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total host->DPU scatter time in ms per iteration, or None if no
    scatter.csv-type file is present."""
    return _sum_time_ms(output_dir, "scatter")


def gather_time_ms(output_dir: Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total DPU->host gather time in ms per iteration, or None if no
    gather.csv-type file is present."""
    return _sum_time_ms(output_dir, "gather")


def copy_time_ms(output_dir: Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total host-side memrefCopy time in ms per iteration (e.g. the
    strided repack copies feeding upmem.scatter buffers), or None if no
    copy.csv-type file is present."""
    return _sum_time_ms(output_dir, "copy")


def _sum_time_ms_by_kind(
    output_dir: Union[pathlib.Path, RunResult], csv_type: str
) -> dict[str, float]:
    """Like _sum_time_ms, but broken down per `kind` column value (e.g.
    "on_array"/"blocks" for scatter, see timers.h's `kind` parameter) instead
    of summed across all of them. Returns {} if no matching csv-type file is
    present, or if it predates the `kind` column."""
    for csv_path in _output_dir(output_dir).glob("*.csv"):
        if _csv_type(csv_path) != csv_type:
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={_iter_col(df): "iteration"})
        if "kind" not in df.columns:
            return {}
        result = {}
        for kind, group in df.groupby("kind"):
            mean = float(group.groupby("iteration")["elapsed_ns"].sum().mean())
            if not isnan(mean):
                result[str(kind)] = mean / 1e6
        return result
    return {}


# The `kind` column now carries the transfer op's own mnemonic. Runs recorded
# before that alignment used "block"/"sg"/"bc"; map them onto the current
# names so old and new output plot the same way.
_LEGACY_KINDS = {
    "scatter:block": "scatter:on_array",
    "scatter:sg": "scatter:blocks",
    "scatter:bc": "scatter:broadcast",
}


# Fixed stacking/legend order for net_breakdown_ms -- keep display code (e.g.
# experiment.py's stacked bar chart) agreeing on category order and colors
# without recomputing it.
_NET_BREAKDOWN_CATEGORIES = [
    "scatter",  # also where scatter:on_array lands, see _canonical_kind
    "scatter:blocks",
    "scatter:broadcast",
    "gather",
    "copy",
    "launch",
    "unaccounted",
]


def _canonical_kind(cat):
    cat = _LEGACY_KINDS.get(cat, cat)
    # The flat per-DPU scatter is the plain "scatter" of a run that wasn't
    # split by kind, so give it the same colour and slot: the two never
    # appear in one plot.
    return "scatter" if cat == "scatter:on_array" else cat


def net_breakdown_color_ix(cat):
    order = [
        "scatter",
        "gather",
        "copy",
        "launch",
        "unaccounted",
        "scatter:blocks",
        "scatter:broadcast",
    ]
    return order.index(_canonical_kind(cat))


def net_breakdown_sort_ix(cat):
    return _NET_BREAKDOWN_CATEGORIES.index(_canonical_kind(cat))


def net_breakdown_ms(
    output_dir: Union[pathlib.Path, RunResult], by_kind: bool = False
) -> dict[str, float] | None:
    """Split net_time_ms (total - alloc - free) into scatter/gather/copy/
    launch time plus whatever's left over as "unaccounted" -- host-side work
    that happens outside any instrumented runtime call (e.g. computation in
    the generated host loop nest). None if no total.csv-type file is
    present.

    If by_kind is True, the "scatter" bucket is instead split into one
    "scatter:<kind>" entry per transfer op kind recorded in scatter.csv's
    `kind` column (e.g. "scatter:on_array", "scatter:blocks") -- note
    these dynamic keys aren't covered by NET_BREAKDOWN_CATEGORIES. Falls back
    to a single "scatter" bucket if the recorded CSV predates the `kind`
    column.
    """
    net = net_time_ms(output_dir)
    if net is None:
        return None
    gather = gather_time_ms(output_dir) or 0.0
    copy = copy_time_ms(output_dir) or 0.0
    launch = launch_time_ms(output_dir) or 0.0

    scatter_by_kind = _sum_time_ms_by_kind(output_dir, "scatter") if by_kind else {}
    if scatter_by_kind:
        scatter_total = sum(scatter_by_kind.values())
        unaccounted = net - scatter_total - gather - copy - launch
        result = {f"scatter:{kind}": t for kind, t in scatter_by_kind.items()}
        result.update(
            {
                "gather": gather,
                "copy": copy,
                "launch": launch,
                "unaccounted": unaccounted,
            }
        )
        return result

    scatter = scatter_time_ms(output_dir) or 0.0
    unaccounted = net - scatter - gather - copy - launch
    return {
        "scatter": scatter,
        "gather": gather,
        "copy": copy,
        "launch": launch,
        "unaccounted": unaccounted,
    }


# Maps a predicted (category, cost_label) pair from aggregate.
# aggregate_predicted_costs' ir/cost.csv rows onto the net_breakdown_ms
# bucket it should be compared against. "cpu"/"other" (host-side work outside
# any instrumented UPMEM runtime call) maps to "unaccounted" rather than
# "copy" -- copy is specifically the strided host-side memrefCopy repacks
# feeding upmem.scatter, which the cost model doesn't currently break out as
# its own predicted category, so there's no predicted counterpart to compare
# it against yet. Falls back to the bare category name for any (category,
# label) pair not listed here, so a new/renamed label doesn't silently
# vanish from aggregation -- it just lands in its own bucket instead of
# being merged into an existing one.
#
# The three "transfer" cost_labels (scatter/scatter_blocks/broadcast -- see
# UpmemOpCountSimulator.cpp's upmem::ScatterOnArrayOp / ScatterBlocksOp /
# BroadcastOp cases) map onto the same
# "scatter:on_array"/"scatter:blocks"/"scatter:broadcast" buckets
# net_breakdown_ms(by_kind=True) already splits the *measured* side into
# (scatter.csv's `kind` column) -- keep both sides on the same three buckets
# rather than merging them back into one "scatter" bucket, so a kernel that
# measures scatter:blocks but predicts scatter:on_array (or vice versa) shows
# up as an error instead of silently cancelling out -- or, before this mapping
# existed, instead of scatter_blocks/broadcast predictions falling through to
# the unlisted "transfer" bucket (not in _BUCKET_ORDER) and vanishing from the
# comparison entirely.
#
# The `kind` values match the op mnemonics; runs recorded before that
# alignment used "block"/"sg"/"bc" and land in their own buckets.
PREDICTED_TO_MEASURED = {
    ("kernel", "kernel"): "launch",
    ("kernel", "launchOverhead"): "launch",
    ("transfer", "array"): "scatter:array",
    ("transfer", "blocks"): "scatter:blocks",
    ("transfer", "broadcast"): "scatter:broadcast",
    ("transfer_back", "array"): "gather:array",
    ("transfer_back", "blocks"): "gather:blocks",
    ("cpu", "other"): "unaccounted",
}


def predicted_bucket(category: str, cost_label: str) -> str:
    """The net_breakdown_ms bucket a predicted (category, cost_label) pair
    (as found in aggregate.aggregate_predicted_costs' output) should be
    compared against -- see PREDICTED_TO_MEASURED."""
    return PREDICTED_TO_MEASURED.get((category, cost_label), category)


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
        rows.append(
            {"fn_name": cfg.fn_name, "label": cfg.label, **cfg.params, "net_time_ms": t}
        )
    if not rows:
        # pd.DataFrame([]) has no columns at all, since there are no rows to
        # infer them from -- guarantee the fixed columns so callers can rely
        # on e.g. df["fn_name"] / df.drop(columns=["label"]) even when every
        # run in `results` failed or was never attempted.
        return pd.DataFrame(columns=["fn_name", "label", "net_time_ms"])
    return pd.DataFrame(rows)

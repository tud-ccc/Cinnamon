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


def _amortizable_index(df: pd.DataFrame) -> pd.Index:
    """The rows of `df` a serving deployment would pay once rather than per
    inference.

    A row qualifies on two counts. Its `tag` must say the data it moves is the
    same on every inference -- `static:<id>`, written by the cnm -> upmem
    conversion, which is the last stage that can still see where the host
    value came from. And its id must appear exactly once in the iteration:
    a transfer inside a loop moves a different tile on every trip, so no
    single load-time transfer replaces it however static its source is. That
    second half cannot be read off the IR, which is why it is asked here.

    Rows with no tag are not amortizable: unattributed means unproven.
    """
    if "tag" not in df.columns:
        return df.index[[]]
    tag = df["tag"].astype("string")
    static = df[tag.notna() & tag.str.startswith("static:")]
    if static.empty:
        return df.index[[]]
    once = static.groupby(["iteration", "tag"]).filter(lambda g: len(g) == 1)
    return once.index


def amortizable_ns(df: pd.DataFrame) -> pd.Series:
    """Per-iteration nanoseconds in `df` that a serving deployment would pay
    once rather than per inference, keyed by iteration. See
    _amortizable_index for which rows those are."""
    rows = df.loc[_amortizable_index(df)]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby("iteration")["elapsed_ns"].sum()


def net_time_ms(
    output_dir: Union[pathlib.Path, RunResult],
    *,
    discount_load: bool = True,
    discount_static_compact: bool = True,
) -> float | None:
    """Mean net time in ms over all iterations recorded in output_dir, or
    None if no total.csv-type file is present.

    Alloc and free are always subtracted (harness overhead). DPU program
    load is subtracted by default -- the amortized convention this function
    has always implemented; the runtime used to time load inside alloc, and
    runs recorded since the split write a separate load.csv, so subtracting
    it here keeps old and new runs comparable. Pass discount_load=False for
    the whole-program (RQ4) analysis, where a load recurring per inference
    is exactly the cost being measured (the per-transfer amortizability
    rule: discounted only if once per workload lifetime).

    Repacks (cnm.compact_buffer, compact.csv) follow the same rule, which is
    why the runtime records them with a `kind`: a repack of a `cinm.static`
    operand happens once per workload lifetime and is subtracted, while one on
    a per-inference operand is a real recurring cost and is not. So do the
    transfers themselves: pinning a weight on the accelerator is the whole
    point of declaring it static, and a scatter of one is paid at load time
    rather than per inference (see amortizable_ns). Pass
    discount_static_compact=False to price the un-amortized case."""
    total_df = None
    alloc_ns = pd.Series(dtype=float)
    free_ns = pd.Series(dtype=float)
    load_ns = pd.Series(dtype=float)
    static_ns: list[pd.Series] = []

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
        elif t == "load" and discount_load:
            load_ns = df.groupby("iteration")["elapsed_ns"].sum()
        elif t == "compact" and discount_static_compact and "kind" in df.columns:
            static = df[df["kind"] == "static"]
            static_ns.append(static.groupby("iteration")["elapsed_ns"].sum())
        elif t == "scatter" and discount_static_compact:
            static_ns.append(amortizable_ns(df))

    if total_df is None or total_df.empty:
        return None

    net = (
        total_df["elapsed_ns"]
        - total_df["iteration"].map(alloc_ns).fillna(0)
        - total_df["iteration"].map(free_ns).fillna(0)
        - total_df["iteration"].map(load_ns).fillna(0)
    )
    for series in static_ns:
        net = net - total_df["iteration"].map(series).fillna(0)
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


def compact_time_ms(output_dir: Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total cnm.compact_buffer repack time in ms per iteration, both
    kinds together, or None if no compact.csv-type file is present. Use
    net_breakdown_ms(by_kind=True) to see the amortizable and per-inference
    repacks apart."""
    return _sum_time_ms(output_dir, "compact")


def load_time_ms(output_dir: Union[pathlib.Path, RunResult]) -> float | None:
    """Mean total DPU program load time in ms per iteration, or None if no
    load.csv-type file is present (runs recorded before the alloc/load timer
    split have it folded into alloc.csv and there is no way to recover it)."""
    return _sum_time_ms(output_dir, "load")


def amortizable_time_ms(
    output_dir: Union[pathlib.Path, RunResult], csv_type: str
) -> float:
    """Mean per-iteration time in ms that amortizable_ns identifies in the
    given csv_type. 0.0 when there is nothing to amortize."""
    for csv_path in _output_dir(output_dir).glob("*.csv"):
        if _csv_type(csv_path) != csv_type:
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={_iter_col(df): "iteration"})
        series = amortizable_ns(df)
        if series.empty:
            return 0.0
        # Iterations with nothing amortizable contribute zero, not nothing:
        # reindexing over every iteration in the file keeps the mean per
        # iteration rather than per iteration that happened to have one.
        iterations = df["iteration"].unique()
        return float(series.reindex(iterations).fillna(0).mean()) / 1e6
    return 0.0


def _sum_time_ms_by_kind(
    output_dir: Union[pathlib.Path, RunResult],
    csv_type: str,
    drop_amortizable: bool = False,
) -> dict[str, float]:
    """Like _sum_time_ms, but broken down per `kind` column value (e.g.
    "on_array"/"blocks" for scatter, see timers.h's `kind` parameter) instead
    of summed across all of them. Returns {} if no matching csv-type file is
    present, or if it predates the `kind` column.

    With drop_amortizable, the rows net_time_ms has already taken out of the
    total are left out here too, so the buckets still sum to net."""
    for csv_path in _output_dir(output_dir).glob("*.csv"):
        if _csv_type(csv_path) != csv_type:
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={_iter_col(df): "iteration"})
        if "kind" not in df.columns:
            return {}
        iterations = df["iteration"].unique()
        if drop_amortizable:
            df = df.drop(index=_amortizable_index(df))
        result = {}
        for kind, group in df.groupby("kind"):
            per_iter = group.groupby("iteration")["elapsed_ns"].sum()
            mean = float(per_iter.reindex(iterations).fillna(0).mean())
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
    "scatter:array",
    "scatter:blocks",
    "scatter:broadcast",
    "gather",
    "gather:array",
    "gather:blocks",
    "copy",
    "compact",  # flat form, when the run predates the compact `kind` column
    "compact:static",  # amortizable; discounted from net unless asked otherwise
    "compact:dyn",  # paid per inference, so never discounted
    "launch",
    "load",  # only present with count_load=True (RQ4's undiscounted view)
    "unaccounted",
]


def _canonical_kind(cat):
    cat = _LEGACY_KINDS.get(cat, cat)
    # The flat per-DPU scatter is the plain "scatter" of a run that wasn't
    # split by kind, so give it the same colour and slot: the two never
    # appear in one plot.
    return "scatter" if cat == "scatter:on_array" else cat


_COLOR_ORDER = [
    "scatter",
    "gather",
    "copy",
    "launch",
    "unaccounted",
    "scatter:array",
    "scatter:blocks",
    "scatter:broadcast",
    "gather:array",
    "gather:blocks",
    "load",
    "compact",
    "compact:static",
    "compact:dyn",
]


def _ix(order: list[str], cat) -> int:
    """Position of `cat` in `order`, or one past the end for a name not
    listed there. Unlisted names do occur by design -- both the per-kind
    breakdown buckets and predicted_bucket's fallback mint bucket names from
    data (a new transfer-op `kind`, an unmapped cost label), and landing in a
    shared last slot keeps them visible in a plot instead of raising."""
    cat = _canonical_kind(cat)
    return order.index(cat) if cat in order else len(order)


def net_breakdown_color_ix(cat):
    return _ix(_COLOR_ORDER, cat)


def net_breakdown_sort_ix(cat):
    return _ix(_NET_BREAKDOWN_CATEGORIES, cat)


def net_breakdown_ms(
    output_dir: Union[pathlib.Path, RunResult],
    by_kind: bool = False,
    count_load: bool = False,
    discount_static_compact: bool = True,
) -> dict[str, float] | None:
    """Split net_time_ms (total - alloc - free - load) into scatter/gather/
    copy/launch time plus whatever's left over as "unaccounted" -- host-side
    work that happens outside any instrumented runtime call (e.g. computation
    in the generated host loop nest). None if no total.csv-type file is
    present.

    If by_kind is True, the "scatter" and "gather" buckets are instead split
    into one "scatter:<kind>"/"gather:<kind>" entry per transfer op kind
    recorded in that CSV's `kind` column (e.g. "scatter:on_array",
    "scatter:blocks", "gather:array") -- note these dynamic keys aren't
    covered by NET_BREAKDOWN_CATEGORIES. Each side independently falls back
    to its single flat bucket if the recorded CSV predates the `kind` column.
    These are the same bucket names PREDICTED_TO_MEASURED maps the cost
    model's predicted categories onto, so predicted_breakdown_ms' output can
    be compared against this one bucket by bucket.

    If count_load is True, program load is NOT discounted from net and
    appears as its own "load" bucket -- the whole-program (RQ4) view, where
    a load recurring per inference is exactly the cost under study.
    """
    net = net_time_ms(
        output_dir,
        discount_load=not count_load,
        discount_static_compact=discount_static_compact,
    )
    if net is None:
        return None
    scatter = scatter_time_ms(output_dir) or 0.0
    if discount_static_compact:
        # Already out of net, so it has to be out of the bucket too.
        scatter -= amortizable_time_ms(output_dir, "scatter")
    gather = gather_time_ms(output_dir) or 0.0
    copy = copy_time_ms(output_dir) or 0.0
    launch = launch_time_ms(output_dir) or 0.0
    load = (load_time_ms(output_dir) or 0.0) if count_load else 0.0
    extra = {"load": load} if count_load else {}

    # Only the repacks net_time_ms left in are shown, so the buckets keep
    # summing to net: with the default discount that is the per-inference ones,
    # and the amortizable ones are already out of the total.
    compacts = _sum_time_ms_by_kind(output_dir, "compact")
    if not compacts:
        flat = compact_time_ms(output_dir)
        compacts = {"": flat} if flat else {}
    if discount_static_compact:
        compacts.pop("static", None)
    compact_buckets = {
        (f"compact:{kind}" if kind else "compact"): t for kind, t in compacts.items()
    }

    transfers = {"scatter": scatter, "gather": gather}
    if by_kind:
        # Each direction falls back to its own flat bucket independently: a
        # run may well record kinds for scatter but not gather (gather.csv
        # grew the column at the same time, but old output/ dirs get mixed
        # with new ones when only part of a sweep is re-run).
        for direction in ("scatter", "gather"):
            by_k = _sum_time_ms_by_kind(
                output_dir,
                direction,
                drop_amortizable=discount_static_compact,
            )
            if by_k:
                del transfers[direction]
                transfers.update({f"{direction}:{k}": t for k, t in by_k.items()})

    unaccounted = (
        net
        - sum(transfers.values())
        - sum(compact_buckets.values())
        - copy
        - launch
        - load
    )
    return {
        **transfers,
        "copy": copy,
        **compact_buckets,
        "launch": launch,
        **extra,
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


def predicted_breakdown_ms(cost_csv: pathlib.Path) -> dict[str, float] | None:
    """The cost model's per-op prediction (ir/cost.csv, written during
    compile by --upmem-annotate-costs) folded onto net_breakdown_ms's
    buckets via predicted_bucket, so the two can be compared bucket by
    bucket. None if the file doesn't exist (e.g. a compile predating
    cost.csv, or a lowering that emits no per-op breakdown)."""
    cost_csv = pathlib.Path(cost_csv)
    if not cost_csv.exists():
        return None
    df = pd.read_csv(cost_csv)
    result: dict[str, float] = {}
    # Summed over blocks: a cost.csv has one row per (block, category, label)
    # and the measured side has no notion of blocks to split them by.
    for (category, cost_label), group in df.groupby(["category", "label"]):
        bucket = predicted_bucket(str(category), str(cost_label))
        result[bucket] = result.get(bucket, 0.0) + float(group["cost_ms"].sum())
    return result


def breakdown_comparison(
    output_dir: Union[pathlib.Path, RunResult], cost_csv: pathlib.Path
) -> pd.DataFrame | None:
    """One row per breakdown bucket with predicted_ms (from cost_csv, folded
    by predicted_breakdown_ms), measured_ms (from net_breakdown_ms(
    by_kind=True)) and rel_error -- the signed relative error
    (predicted - measured) / measured, so over-prediction is positive.

    Buckets present on only one side are kept with 0.0 on the other, since a
    prediction with no measured counterpart (or vice versa) is exactly the
    kind of mismatch this comparison is meant to surface -- but their
    rel_error is NaN where measured_ms is 0, there being nothing to be
    relative to. Rows are in net_breakdown_sort_ix order. None if either side
    is missing entirely.

    The measured side is the undiscounted one. The cost model predicts what
    the program executes, so that is what its accuracy has to be judged
    against; amortizing a weight transfer is a statement about how often a
    deployment pays it, not about whether it happened. Comparing against the
    amortized view would score the model on transfers that were deliberately
    subtracted from it."""
    measured = net_breakdown_ms(output_dir, by_kind=True, discount_static_compact=False)
    predicted = predicted_breakdown_ms(cost_csv)
    if measured is None or predicted is None:
        return None
    rows = []
    buckets = sorted(
        set(measured) | set(predicted), key=lambda b: (net_breakdown_sort_ix(b), b)
    )
    for bucket in buckets:
        m = measured.get(bucket, 0.0)
        p = predicted.get(bucket, 0.0)
        rows.append(
            {
                "bucket": bucket,
                "predicted_ms": p,
                "measured_ms": m,
                "rel_error": (p - m) / m if m else float("nan"),
            }
        )
    return pd.DataFrame(
        rows, columns=["bucket", "predicted_ms", "measured_ms", "rel_error"]
    )


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

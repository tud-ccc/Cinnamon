"""RQ3's predicted-vs-measured join: one row per (config, term).

The predicted side is the cost model's per-op breakdown (ir/cost.csv,
written during compile by --upmem-annotate-costs); the measured side is the
bench binary's output/*.csv as measurements.py reads them. Both sides are
folded to the paper's three terms -- transfer (scatter + gather), kernel
(launch), combined (net) -- because that is the granularity fig:fidelity
argues at; the finer bucket alignment lives in
measurements.PREDICTED_TO_MEASURED and stays available for debugging.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pandas as pd

from . import measurements
from .aggregate import iter_config_dirs

# cost.csv categories folded into each term. "cpu"/"other" is host-side work
# outside any instrumented call; it belongs to combined only.
_PREDICTED_TERM_CATEGORIES = {
    "transfer": ("transfer", "transfer_back"),
    "kernel": ("kernel",),
}


def _predicted_terms(cost_csv: pathlib.Path) -> dict[str, float] | None:
    if not cost_csv.exists():
        return None
    df = pd.read_csv(cost_csv)
    terms = {
        term: float(df.loc[df["category"].isin(cats), "cost_ms"].sum())
        for term, cats in _PREDICTED_TERM_CATEGORIES.items()
    }
    # combined drops the cost model's `excluded` rows (transfers of data
    # pinned on the device across inferences, see SimCost) to match the
    # measured combined, which is net_time_ms -- it subtracts the runtime rows
    # of those same transfers. The transfer term keeps them, because the
    # measured transfer term is the raw scatter+gather time; the asymmetry is
    # the measured side's, and predicting it is the point. Older cost.csvs
    # have no such column and nothing to drop.
    charged = (
        df if "excluded" not in df.columns else df.loc[~df["excluded"].astype(bool)]
    )
    terms["combined"] = float(charged["cost_ms"].sum())
    return terms


def _measured_terms(
    output_dir: pathlib.Path,
) -> (
    tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float, float]]]
    | None
):
    """(raw, charged, noise) per term. The raw transfer term is the whole
    scatter+gather time, which is what the transfer panel compares against
    (its predicted side keeps the `excluded` rows for the same reason). The
    charged one drops the amortizable static scatters, because net_time_ms
    subtracts those: only the charged time is a part of combined, so only it
    can be expressed as a share of it. The noise is measurements.noise of
    each raw term's per-iteration series.

    Each process's first iteration is a warmup and is left out: the first
    launch, the first touch of a host buffer and the program load all land
    in it."""
    net = measurements.net_series_ms(output_dir, drop_first=True)
    if net is None or net.empty:
        return None
    scatter = measurements.series_ms(output_dir, "scatter", drop_first=True)
    gather = measurements.series_ms(output_dir, "gather", drop_first=True)
    kernel = measurements.series_ms(output_dir, "launch", drop_first=True)

    def median(series) -> float:
        return float(series["ms"].median()) if series is not None else 0.0

    # A transfer iteration is its scatters and its gathers together, so the
    # noise is taken of their sum, iteration by iteration.
    parts = [x for x in (scatter, gather) if x is not None]
    transfer = (
        pd.concat(parts)
        .groupby("iteration", as_index=False)
        .agg(process=("process", "first"), ms=("ms", "sum"))
        if parts
        else None
    )
    amortized = measurements.amortizable_time_ms(output_dir, "scatter", drop_first=True)
    raw = {
        "transfer": median(scatter) + median(gather),
        "kernel": median(kernel),
        "combined": float(net["ms"].median()),
    }
    charged = dict(raw, transfer=median(scatter) - amortized + median(gather))
    noise = {
        "transfer": measurements.noise(transfer),
        "kernel": measurements.noise(kernel),
        "combined": measurements.noise(net),
    }
    return raw, charged, noise


def _config_knobs(config_csv: pathlib.Path) -> dict[str, float]:
    """The two knobs every benchmark's space has, so that the frame stays
    rectangular across benchmarks -- the rest of config.csv is per-benchmark
    (tile sizes, loop order) and has no common column. They are what the
    reporting layer groups the residuals by: dpus is the fan-out the transfer
    model extrapolates along, tasklets the kernel model's."""
    if not config_csv.exists():
        return {"dpus": float("nan"), "tasklets": float("nan")}
    meta = pd.read_csv(config_csv).iloc[0].to_dict()
    return {k: meta.get(k, float("nan")) for k in ("dpus", "tasklets")}


def config_rows(job: tuple[str, pathlib.Path, pathlib.Path]) -> list[dict]:
    """fidelity_frame's three rows for one (fn_name, compile_dir, config_dir)
    job, none when either side is missing. A top-level function of one
    argument, so that a caller can map it over a process pool."""
    fn_name, compile_dir, config_dir = job
    predicted = _predicted_terms(compile_dir / "ir" / "cost.csv")
    measured = _measured_terms(config_dir / "output")
    if predicted is None or measured is None:
        return []
    raw, charged, noise = measured
    knobs = _config_knobs(compile_dir / "config.csv")
    return [
        {
            "fn_name": fn_name,
            "label": config_dir.name,
            "term": term,
            "predicted_ms": predicted[term],
            "measured_ms": raw[term],
            "charged_ms": charged[term],
            "net_ms": raw["combined"],
            "noise_within": noise[term][0],
            "noise_between": noise[term][1],
            "noise_se": noise[term][2],
            **knobs,
        }
        for term in ("transfer", "kernel", "combined")
    ]


def fidelity_frame(
    compile_root: pathlib.Path,
    run_root: pathlib.Path,
    map_jobs: Callable[[Callable, list], list] = lambda fn, jobs: list(map(fn, jobs)),
) -> pd.DataFrame:
    """One row per (fn_name, label, term) with predicted_ms, measured_ms and
    the two measured times the paper's share-of-time weighting is a ratio of:
    charged_ms (what this term contributes to measured combined) over net_ms
    (measured combined itself, repeated on every term's row). The share is
    left to the reporting layer to divide, so that it can total the two sums
    over a set of configs rather than average per-config ratios. Each row also
    carries the config's dpus and tasklets, the two knobs fig:fidelity reads
    the residuals against. Configs missing either side are skipped: a compile
    without a bench has no measured truth, a bench whose compile predates
    cost.csv has no prediction. Empty frame when nothing joins -- the assemble
    layer turns that into a MISSING note, not an error.

    `map_jobs(config_rows, jobs)` measures the config dirs; it must return the
    results in job order, as the serial default does. The assemble layer
    passes one that spreads them over processes."""
    jobs = [
        (fn_name, pathlib.Path(compile_root) / fn_name / config_dir.name, config_dir)
        for fn_name, config_dir in iter_config_dirs(pathlib.Path(run_root))
    ]
    rows = [row for rows in map_jobs(config_rows, jobs) for row in rows]
    columns = [
        "fn_name",
        "label",
        "term",
        "predicted_ms",
        "measured_ms",
        "charged_ms",
        "net_ms",
        "noise_within",
        "noise_between",
        "noise_se",
        "dpus",
        "tasklets",
    ]
    return pd.DataFrame(rows, columns=columns)

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
) -> tuple[dict[str, float], dict[str, float]] | None:
    """(raw, charged) per term. The raw transfer term is the whole
    scatter+gather time, which is what the transfer panel compares against
    (its predicted side keeps the `excluded` rows for the same reason). The
    charged one drops the amortizable static scatters, because net_time_ms
    subtracts those: only the charged time is a part of combined, so only it
    can be expressed as a share of it."""
    net = measurements.net_time_ms(output_dir)
    if net is None:
        return None
    scatter = measurements.scatter_time_ms(output_dir) or 0.0
    gather = measurements.gather_time_ms(output_dir) or 0.0
    kernel = measurements.launch_time_ms(output_dir) or 0.0
    amortized = measurements.amortizable_time_ms(output_dir, "scatter")
    raw = {"transfer": scatter + gather, "kernel": kernel, "combined": net}
    charged = dict(raw, transfer=scatter - amortized + gather)
    return raw, charged


def fidelity_frame(compile_root: pathlib.Path, run_root: pathlib.Path) -> pd.DataFrame:
    """One row per (fn_name, label, term) with predicted_ms, measured_ms and
    the two measured times the paper's share-of-time weighting is a ratio of:
    charged_ms (what this term contributes to measured combined) over net_ms
    (measured combined itself, repeated on every term's row). The share is
    left to the reporting layer to divide, so that it can total the two sums
    over a set of configs rather than average per-config ratios. Configs
    missing either side are skipped: a compile without a bench has no measured
    truth, a bench whose compile predates cost.csv has no prediction. Empty
    frame when nothing joins -- the assemble layer turns that into a MISSING
    note, not an error."""
    rows = []
    for fn_name, config_dir in iter_config_dirs(pathlib.Path(run_root)):
        compile_dir = pathlib.Path(compile_root) / fn_name / config_dir.name
        predicted = _predicted_terms(compile_dir / "ir" / "cost.csv")
        measured = _measured_terms(config_dir / "output")
        if predicted is None or measured is None:
            continue
        raw, charged = measured
        for term in ("transfer", "kernel", "combined"):
            rows.append(
                {
                    "fn_name": fn_name,
                    "label": config_dir.name,
                    "term": term,
                    "predicted_ms": predicted[term],
                    "measured_ms": raw[term],
                    "charged_ms": charged[term],
                    "net_ms": raw["combined"],
                }
            )
    columns = [
        "fn_name",
        "label",
        "term",
        "predicted_ms",
        "measured_ms",
        "charged_ms",
        "net_ms",
    ]
    return pd.DataFrame(rows, columns=columns)

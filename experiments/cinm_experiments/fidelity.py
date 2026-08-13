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
    terms["combined"] = float(df["cost_ms"].sum())
    return terms


def _measured_terms(output_dir: pathlib.Path) -> dict[str, float] | None:
    net = measurements.net_time_ms(output_dir)
    if net is None:
        return None
    scatter = measurements.scatter_time_ms(output_dir) or 0.0
    gather = measurements.gather_time_ms(output_dir) or 0.0
    kernel = measurements.launch_time_ms(output_dir) or 0.0
    return {"transfer": scatter + gather, "kernel": kernel, "combined": net}


def fidelity_frame(compile_root: pathlib.Path, run_root: pathlib.Path) -> pd.DataFrame:
    """One row per (fn_name, label, term) with predicted_ms, measured_ms and
    share_of_total (the measured term's share of measured combined -- the
    paper's share-of-time weighting). Configs missing either side are
    skipped: a compile without a bench has no measured truth, a bench whose
    compile predates cost.csv has no prediction. Empty frame when nothing
    joins -- the assemble layer turns that into a MISSING note, not an
    error."""
    rows = []
    for fn_name, config_dir in iter_config_dirs(pathlib.Path(run_root)):
        compile_dir = pathlib.Path(compile_root) / fn_name / config_dir.name
        predicted = _predicted_terms(compile_dir / "ir" / "cost.csv")
        measured = _measured_terms(config_dir / "output")
        if predicted is None or measured is None:
            continue
        total = measured["combined"]
        for term in ("transfer", "kernel", "combined"):
            rows.append(
                {
                    "fn_name": fn_name,
                    "label": config_dir.name,
                    "term": term,
                    "predicted_ms": predicted[term],
                    "measured_ms": measured[term],
                    "share_of_total": measured[term] / total if total else float("nan"),
                }
            )
    columns = [
        "fn_name",
        "label",
        "term",
        "predicted_ms",
        "measured_ms",
        "share_of_total",
    ]
    return pd.DataFrame(rows, columns=columns)

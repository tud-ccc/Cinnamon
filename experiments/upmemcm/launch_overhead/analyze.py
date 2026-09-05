"""Turn the launch sweep into the shape of a cost-model term.

The kernel here is empty, so a cost model that prices instructions and DMAs
predicts zero for every one of these launches and the whole measurement is
the overhead it does not charge. The question is only what that overhead is a
function of, so this fits the candidate forms against the sweep and reports
which of them the data actually supports.

CANDIDATES are the forms worth arguing about, including the three that
UpmemPythonSimulator.cpp carries commented out beside its launchOverhead
term (linear in DPUs, linear in ranks, log2 of DPUs).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DPUS_PER_RANK = 64

# name -> the regressors of a linear model in those terms; the constant is
# added to all of them, so "constant" is the model with no regressor at all.
CANDIDATES: dict[str, list] = {
    "constant": [],
    "log2(dpus)": [lambda n: np.log2(n)],
    "dpus": [lambda n: n],
    "ranks": [lambda n: np.ceil(n / DPUS_PER_RANK)],
    "log2(dpus) + dpus": [lambda n: np.log2(n), lambda n: n],
}


def load(results_csv: pathlib.Path) -> pd.DataFrame:
    """Per-DPU-count launch statistics, in ms.

    The median, not the mean: a sweep this cheap picks up the occasional
    cold outlier (one 4x launch in fifty), and one of those moves a mean by
    more than the effect being measured."""
    df = pd.read_csv(results_csv)
    launches = df[df["phase"] == "launch"]
    stats = (
        launches.groupby("allocated_dpus")["ns"]
        .agg(
            median_ms=lambda s: s.median() / 1e6,
            mean_ms=lambda s: s.mean() / 1e6,
            p05_ms=lambda s: s.quantile(0.05) / 1e6,
            p95_ms=lambda s: s.quantile(0.95) / 1e6,
            n="count",
        )
        .reset_index()
    )
    for phase in ("alloc", "load"):
        once = df[df["phase"] == phase].set_index("allocated_dpus")["ns"] / 1e6
        stats[f"{phase}_ms"] = stats["allocated_dpus"].map(once)
    return stats


def fit(stats: pd.DataFrame) -> pd.DataFrame:
    """Least squares for each candidate, on the medians. Returns one row per
    candidate with its coefficients and its RMSE, worst point first ranked by
    fit quality."""
    n = stats["allocated_dpus"].to_numpy(dtype=float)
    y = stats["median_ms"].to_numpy()
    rows = []
    for name, terms in CANDIDATES.items():
        design = np.column_stack([np.ones_like(n)] + [t(n) for t in terms])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coeffs
        rows.append(
            {
                "form": name,
                "rmse_ms": float(np.sqrt(np.mean((y - pred) ** 2))),
                "max_err_ms": float(np.max(np.abs(y - pred))),
                "coeffs": ", ".join(f"{c:.6g}" for c in coeffs),
                "_pred": pred,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_ms").reset_index(drop=True)


def plot(stats: pd.DataFrame, fits: pd.DataFrame, out_path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = stats["allocated_dpus"]
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    top.fill_between(n, stats["p05_ms"], stats["p95_ms"], color="#4D4D4D", alpha=0.2)
    top.plot(n, stats["median_ms"], "o-", color="#4D4D4D", label="measured (median)")
    # Only the best two: the point is which form the data picks, and five
    # curves through eight points is a thicket, not evidence.
    for (_, row), style in zip(fits.iterrows(), ("-", "--")):
        top.plot(
            n,
            row["_pred"],
            style,
            color="#DD8452" if style == "-" else "#8172B3",
            label=f"{row['form']}  (rmse {row['rmse_ms']:.3f} ms)",
        )
    top.set_xscale("log", base=2)
    top.set_ylabel("dpu_launch (ms)")
    top.set_title("Cost of launching an empty DPU program")
    top.grid(linestyle="--", alpha=0.4)
    top.set_axisbelow(True)
    top.legend()

    best = fits.iloc[0]
    bottom.axhline(0.0, color="black", linewidth=0.8)
    bottom.plot(n, stats["median_ms"] - best["_pred"], "o-", color="#DD8452")
    bottom.set_xscale("log", base=2)
    bottom.set_xticks(list(n))
    bottom.set_xticklabels([str(int(v)) for v in n], rotation=45)
    bottom.set_xlabel("DPUs in the set")
    bottom.set_ylabel(f"residual of\n{best['form']} (ms)")
    bottom.grid(linestyle="--", alpha=0.4)
    bottom.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def report(results_csv: pathlib.Path, out_path: pathlib.Path) -> None:
    stats = load(results_csv)
    fits = fit(stats)
    print(
        stats[
            ["allocated_dpus", "median_ms", "p05_ms", "p95_ms", "alloc_ms", "load_ms"]
        ].to_string(index=False, float_format=lambda v: f"{v:9.4f}")
    )
    print()
    print(
        fits[["form", "rmse_ms", "max_err_ms", "coeffs"]].to_string(
            index=False, float_format=lambda v: f"{v:9.4f}"
        )
    )
    plot(stats, fits, out_path)
    print(f"\nwrote {out_path}")

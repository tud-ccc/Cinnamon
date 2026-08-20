"""fig:fidelity -- predicted vs measured, one panel per term (rq3.csv).

Log-log scatter with a y=x reference; per-panel Spearman rho, MAPE, and the
share of measured time the term accounts for -- summed over the suite, so
transfer and kernel add up to at most the whole of it (the paper's argument
that transfer-term fidelity is what matters is made with that share). Top-k
overlap against the measured ranking is printed for the combined term.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from _reporting import load_or_skip, parse_dirs, save_fig

TERMS = ["transfer", "kernel", "combined"]
TOP_K = 20


def _topk_overlap(predicted, measured, k: int) -> float:
    pred_top = set(np.argsort(predicted)[:k])
    meas_top = set(np.argsort(measured)[:k])
    return len(pred_top & meas_top) / k


def main() -> None:
    results_dir, out_dir = parse_dirs("plots")
    df = load_or_skip(results_dir, "rq3.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, len(TERMS), figsize=(4.2 * len(TERMS), 4))
    for ax, term in zip(axes, TERMS):
        sub = df[(df["term"] == term) & (df["measured_ms"] > 0)]
        if sub.empty:
            ax.set_title(f"{term} (no data)")
            continue
        p, m = sub["predicted_ms"].to_numpy(), sub["measured_ms"].to_numpy()
        for bench, b in sub.groupby("benchmark"):
            ax.scatter(b["measured_ms"], b["predicted_ms"], s=8, alpha=0.5, label=bench)
        lo = min(p[p > 0].min(), m.min())
        hi = max(p.max(), m.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("measured (ms)")
        ax.set_ylabel("predicted (ms)")
        rho = spearmanr(p, m).statistic if len(sub) > 2 else float("nan")
        mape = float(np.mean(np.abs((p - m) / m)) * 100)
        # Summed over configs, not averaged per config: the share is what
        # fraction of the suite's measured time the term is, so a 0.1 ms
        # config must not weigh as much as a 100 ms one.
        share = float(sub["charged_ms"].sum() / sub["net_ms"].sum() * 100)
        stats = f"$\\rho$={rho:.2f}  MAPE={mape:.0f}%"
        if term != "combined":
            stats += f"  share={share:.0f}%"
        else:
            stats += f"  top-{TOP_K} overlap={_topk_overlap(p, m, TOP_K):.2f}"
        ax.set_title(f"{term}\n{stats}", fontsize=10)
    axes[0].legend(fontsize=7)
    save_fig(fig, out_dir, "fidelity.pdf")


if __name__ == "__main__":
    sys.exit(main())

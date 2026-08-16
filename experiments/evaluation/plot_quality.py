"""fig:quality -- end-to-end quality bars, normalized to PrIM (rq1.csv).

One group per benchmark: PrIM / ATiM / ours / CPU, each the best (minimum
total_ms over configs and fns) that system measured for the benchmark,
plotted as speedup relative to PrIM so the hand-optimized kernels are the
1.0 line. Systems with no rows yet are skipped per benchmark; CINM 1.0 is
deliberately absent here -- it goes to the companion table
(table_quality.py), per the paper's figure note.

ATiM gets two bars. The schedules published with their artifact and the
ones we reproduced by tuning on this machine are different searches on
different hardware, and the minimum over the pair would be a
configuration neither run produced -- so they are separate systems here,
never a single "ATiM".
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _reporting import load_or_skip, parse_dirs, save_fig

# Bar order; prim is the baseline.
SYSTEMS = ["prim", "atim_published", "atim_reproduced", "ours", "cpu"]


def main() -> None:
    results_dir, out_dir = parse_dirs("plots")
    df = load_or_skip(results_dir, "rq1.csv")
    if df is None:
        return

    best = (
        df.dropna(subset=["total_ms"])
        .groupby(["benchmark", "system"])["total_ms"]
        .min()
        .unstack()
    )
    if "prim" not in best.columns:
        print("rq1.csv has no prim rows yet -- nothing to normalize against")
        return
    benches = sorted(best.index)
    width = 0.8 / len(SYSTEMS)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(benches)), 4))
    x = np.arange(len(benches))
    for i, system in enumerate(SYSTEMS):
        if system not in best.columns:
            continue
        speedup = best.loc[benches, "prim"] / best.loc[benches, system]
        ax.bar(x + (i - len(SYSTEMS) / 2 + 0.5) * width, speedup, width, label=system)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x, benches, rotation=30, ha="right")
    ax.set_ylabel("speedup over PrIM")
    ax.legend()
    save_fig(fig, out_dir, "quality.pdf")


if __name__ == "__main__":
    sys.exit(main())

"""fig:wholeprogram -- per-operator vs whole-program arms (rq4.csv).

One bar pair per (program, variant): the per-operator-tuned and the
whole-program-tuned arm, each stacked into kernel (launch) / scatter /
gather / load / rest. The stack uses the undiscounted breakdown -- RQ4's
whole point is the load and rescatter cost the per-operator arm pays per
inference. Colors come from measurements.net_breakdown_color_ix so every
breakdown figure in the repo shades a bucket the same way.

Execution is synchronous (upmem.wait_for blocks the host, so blocks on
disjoint groups still run in program order): parallel variants
show staticity preservation, not overlap, and this figure must not imply
otherwise.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _reporting import load_or_skip, parse_dirs, save_fig

# After _reporting: importing it puts experiments/ on sys.path, which is
# what makes the shared harness importable from a plain `python` run.
from cinm_experiments.measurements import net_breakdown_color_ix  # noqa: E402

ARMS = ["peroper", "wholeprog"]
ARM_LABEL = {"peroper": "per-operator", "wholeprog": "whole-program"}
# Stack order: the terms the paper names first, the remainder on top.
BUCKETS = ["launch", "scatter", "gather", "load", "copy", "unaccounted"]
BUCKET_LABEL = {"launch": "kernel"}
_CMAP = plt.get_cmap("tab10")


def main() -> None:
    results_dir, out_dir = parse_dirs("plots")
    df = load_or_skip(results_dir, "rq4.csv")
    if df is None:
        return

    # Best (minimum total) config per (program, arm) -- each arm shows the
    # configuration its own tuning strategy picked.
    best = df.loc[df.groupby(["program", "arm"])["total_ms"].idxmin()]
    programs = sorted(best["program"].unique())
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(programs)), 4.5))
    x = np.arange(len(programs))
    for a, arm in enumerate(ARMS):
        bottoms = np.zeros(len(programs))
        for bucket in BUCKETS:
            col = f"{bucket}_ms"
            heights = np.array(
                [
                    float(
                        best.loc[
                            (best["program"] == p) & (best["arm"] == arm), col
                        ].sum()
                    )
                    if col in best.columns
                    else 0.0
                    for p in programs
                ]
            )
            if not heights.any():
                continue
            ax.bar(
                x + (a - 0.5) * width,
                heights,
                width * 0.9,
                bottom=bottoms,
                color=_CMAP(net_breakdown_color_ix(bucket)),
                # The left bar of each pair is the per-operator arm; the
                # whole-program arm is hatched, so the pair reads without
                # a second legend.
                hatch="//" if arm == "wholeprog" else None,
                label=BUCKET_LABEL.get(bucket, bucket) if a == 0 else None,
            )
            bottoms += heights
    ax.set_xticks(x, programs, rotation=30, ha="right")
    ax.set_ylabel("time per inference (ms), undiscounted")
    ax.legend(
        fontsize=8,
        title=f"stacks; plain={ARM_LABEL['peroper']}, hatched={ARM_LABEL['wholeprog']}",
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    save_fig(fig, out_dir, "wholeprogram.pdf")


if __name__ == "__main__":
    sys.exit(main())

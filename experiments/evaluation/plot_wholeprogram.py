"""fig:wholeprogram -- per-operator vs whole-program arms (rq4.csv).

One panel per size class (1MB/16MB/64MB/256MB of weights per gemm), one
bar pair per program inside it: the per-operator-tuned and the
whole-program-tuned arm, each stacked into kernel (launch) / scatter /
gather / load / repack / rest. Panels get their own y scale -- classes are
two orders of magnitude apart, and the comparison lives inside a pair, not
across panels. Read left to right, the panels sweep across the device's
contention threshold: in the smallest class every per-operator set fits
resident and the arms tie, while the larger classes force the per-operator
sets to evict each other and the recurring load + rescatter cost appears
in the plain bars only. The stack uses the undiscounted breakdown -- RQ4's
whole point is that under UPMEM_RT_CACHE amortization is physical, so
whatever still recurs in the steady state is real and charged.

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
BUCKETS = [
    "launch",
    "scatter",
    "gather",
    "load",
    "compact:static",
    "compact:dyn",
    "copy",
    "unaccounted",
]
BUCKET_LABEL = {
    "launch": "kernel",
    "compact:static": "repack (static)",
    "compact:dyn": "repack (dyn)",
}
CLASSES = ["1MB", "16MB", "64MB", "256MB"]
_CMAP = plt.get_cmap("tab10")


def main() -> None:
    results_dir, out_dir = parse_dirs("plots")
    df = load_or_skip(results_dir, "rq4.csv")
    if df is None:
        return

    df = df.copy()
    df["cls"] = df["fn_name"].str.rsplit("_", n=1).str[-1]
    classes = [c for c in CLASSES if c in set(df["cls"])]
    programs = sorted(df["program"].unique())
    width = 0.35

    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(max(6, 1.9 * len(programs)) * len(classes) / 2.2, 4.0),
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(programs))
    for ax, cls in zip(axes, classes):
        sub = df[df["cls"] == cls]
        for a, arm in enumerate(ARMS):
            bottoms = np.zeros(len(programs))
            for bucket in BUCKETS:
                col = f"{bucket}_ms"
                heights = np.array(
                    [
                        float(
                            sub.loc[
                                (sub["program"] == p) & (sub["arm"] == arm), col
                            ].sum()
                        )
                        if col in sub.columns
                        else 0.0
                        for p in programs
                    ]
                )
                # Clamp the residual bucket: a slightly negative unaccounted
                # is timing jitter and must not corrupt the stack.
                heights = np.maximum(heights, 0.0)
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
                    label=(
                        BUCKET_LABEL.get(bucket, bucket)
                        if a == 0 and ax is axes[0]
                        else None
                    ),
                )
                bottoms += heights
        ax.set_title(cls, fontsize=10)
        ax.set_xticks(x, programs, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("time per inference (ms), undiscounted")
    axes[0].legend(
        fontsize=8,
        title=f"stacks; plain={ARM_LABEL['peroper']}, hatched={ARM_LABEL['wholeprog']}",
        title_fontsize=8,
        loc="upper left",
    )
    fig.tight_layout()
    save_fig(fig, out_dir, "wholeprogram.pdf")


if __name__ == "__main__":
    sys.exit(main())

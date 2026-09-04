"""fig:wholeprogram -- per-operator vs whole-program arms (rq4.csv).

One panel per size class (1MB/16MB/64MB/256MB of weights per gemm), one
bar pair per program inside it: the per-operator-tuned and the
whole-program-tuned arm, each stacked into kernel (launch) / scatter /
gather / load / repack / rest.

Two files come out. wholeprogram.pdf is the device-only figure, the one
RQ4 argues from. wholeprogram_cpu.pdf, drawn whenever cpu.csv exists, adds
a third unstacked bar per program: the same program on the host CPU via
stock TVM (cpu_baseline.py) -- context for the two device bars, not a
competitor, since RQ4's question is what per-operator allocation costs on
the device. It is grey and flat, since a CPU run has no scatter or load
segment to break out, which is itself the comparison. Panels get their own y scale -- classes are
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


def draw(df, cpu):
    """The figure, with the CPU context bar iff `cpu` is a frame. Both
    variants are drawn from the same code so the device arms are laid out
    identically in each: the pair a reader compares must not shift because
    a third bar joined it."""
    classes = [c for c in CLASSES if c in set(df["cls"])]
    programs = sorted(df["program"].unique())
    # Three slots per program when the CPU context bar is drawn, two
    # otherwise, so the pair keeps the width it had before cpu.csv existed.
    slots = 3 if cpu is not None else 2
    width = 0.8 / slots

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
                    x + (a - (slots - 1) / 2) * width,
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
        if cpu is not None:
            sub_cpu = cpu[cpu["cls"] == cls]
            heights = np.array(
                [
                    float(sub_cpu.loc[sub_cpu["program"] == p, "total_ms"].sum())
                    for p in programs
                ]
            )
            if heights.any():
                ax.bar(
                    x + (len(ARMS) - (slots - 1) / 2) * width,
                    heights,
                    width * 0.9,
                    color="0.55",
                    label="TVM CPU (context)" if ax is axes[0] else None,
                )
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
    return fig


def main() -> None:
    results_dir, out_dir = parse_dirs("plots")
    df = load_or_skip(results_dir, "rq4.csv")
    if df is None:
        return

    df = df.copy()
    df["cls"] = df["fn_name"].str.rsplit("_", n=1).str[-1]

    # The device-only figure is always drawn: it is the one RQ4 argues from,
    # and it must not depend on whether the CPU run has happened. The
    # context variant is drawn beside it when cpu.csv is there, under its
    # own name, so the paper can include either without a rerun.
    save_fig(draw(df, None), out_dir, "wholeprogram.pdf")
    cpu = load_or_skip(results_dir, "cpu.csv")
    if cpu is not None:
        save_fig(draw(df, cpu), out_dir, "wholeprogram_cpu.pdf")


if __name__ == "__main__":
    sys.exit(main())

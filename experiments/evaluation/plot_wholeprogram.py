"""fig:wholeprogram -- per-operator vs whole-program arms (rq4.csv).

One panel per size class (1MB/64MB/256MB/512MB of weights per gemm), one
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
CLASSES = ["1MB", "64MB", "256MB", "512MB"]
_TAB10 = plt.get_cmap("tab10").colors
_TAB20 = plt.get_cmap("tab20").colors


def _color(bucket):
    """The bucket's colour. tab10 holds ten, and net_breakdown_color_ix
    numbers more buckets than that -- load and the two repack buckets sit
    past the end, where indexing tab10 silently clamps and painted all three
    the same cyan as each other. Those take tab20's light variants instead,
    which are distinct from every tab10 hue; the buckets tab10 does cover
    keep the colour they have in every other breakdown figure."""
    i = net_breakdown_color_ix(bucket)
    if i < len(_TAB10):
        return _TAB10[i]
    return _TAB20[(2 * (i - len(_TAB10)) + 1) % len(_TAB20)]


def _draw_spread(ax, sub, arm, programs, offsets, width, legend):
    """The seed-to-seed spread of one arm's totals, over its stacked bar.

    A range whisker rather than a standard deviation, and every seed's total
    as its own dot. The per-operator arm's repeats are not a cloud around a
    mean: an allocation either co-resides or evicts, so its totals can land
    in two groups with nothing between them, and a bar plus a symmetric
    interval would draw a distribution that never occurred. The dots show
    the shape directly and cost almost nothing; the whisker only says where
    the extremes are, which needs no assumption about what lies between.

    The count under a per-operator bar is how many seeds kept their weights
    resident, read off a recurring program load in the steady state -- under
    UPMEM_RT_CACHE a resident set is never reloaded, so load_ms > 0 means
    that seed's sets evicted each other. It is the frequency with which
    per-operator search preserves residency by accident, which is a result
    rather than an error term.
    """
    for i, p in enumerate(programs):
        rows = sub[(sub["program"] == p) & (sub["arm"] == arm)]
        totals = rows["total_ms"].to_numpy(dtype=float)
        if len(totals) < 2:
            continue
        lo, hi = float(totals.min()), float(totals.max())
        ax.vlines(offsets[i], lo, hi, color="0.15", linewidth=0.9, zorder=4)
        # Deterministic jitter: seeds spread across the bar in a fixed order,
        # so redrawing the same data gives the same picture.
        jitter = np.linspace(-0.28, 0.28, len(totals)) * width
        dots = ax.scatter(
            offsets[i] + jitter,
            totals,
            s=5,
            color="0.1",
            zorder=5,
            linewidths=0,
        )
        legend.setdefault(f"per-seed total (n={len(totals)})", dots)
        if arm == "peroper" and "load_ms" in rows.columns:
            resident = int((rows["load_ms"].to_numpy(dtype=float) <= 0).sum())
            ax.annotate(
                f"{resident}/{len(totals)}",
                (offsets[i], hi),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=5.5,
                color="0.35",
            )


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
    # Legend entries are collected across every panel, not taken from the
    # first: a bucket can be absent there and present later -- program load
    # is exactly that, always zero in the smallest class and the whole point
    # of the figure in the larger ones -- and labelling only the first panel
    # left it drawn but unnamed.
    legend: dict[str, object] = {}
    for ax, cls in zip(axes, classes):
        sub = df[df["cls"] == cls]
        for a, arm in enumerate(ARMS):
            bottoms = np.zeros(len(programs))
            offsets = x + (a - (slots - 1) / 2) * width
            for bucket in BUCKETS:
                col = f"{bucket}_ms"
                # Mean over seeds, never the sum: a cell holds one row per
                # repeat. Segment means are linear, so they stack to the mean
                # total, which is what lets the whisker below carry the whole
                # bar's spread on its own.
                heights = np.array(
                    [
                        float(
                            sub.loc[
                                (sub["program"] == p) & (sub["arm"] == arm), col
                            ].mean()
                        )
                        if col in sub.columns
                        else 0.0
                        for p in programs
                    ]
                )
                heights = np.nan_to_num(heights)
                # Clamp the residual bucket: a slightly negative unaccounted
                # is timing jitter and must not corrupt the stack.
                heights = np.maximum(heights, 0.0)
                if not heights.any():
                    continue
                bars = ax.bar(
                    offsets,
                    heights,
                    width * 0.9,
                    bottom=bottoms,
                    color=_color(bucket),
                    # The left bar of each pair is the per-operator arm; the
                    # whole-program arm is hatched, so the pair reads without
                    # a second legend.
                    hatch="//" if arm == "wholeprog" else None,
                )
                legend.setdefault(BUCKET_LABEL.get(bucket, bucket), bars[0])
                bottoms += heights
            _draw_spread(ax, sub, arm, programs, offsets, width, legend)
        if cpu is not None:
            sub_cpu = cpu[cpu["cls"] == cls]
            heights = np.array(
                [
                    float(sub_cpu.loc[sub_cpu["program"] == p, "total_ms"].sum())
                    for p in programs
                ]
            )
            if heights.any():
                bars = ax.bar(
                    x + (len(ARMS) - (slots - 1) / 2) * width,
                    heights,
                    width * 0.9,
                    color="0.55",
                    zorder=2,
                )
                legend.setdefault("TVM CPU (context)", bars[0])
        ax.set_title(cls, fontsize=10)
        ax.set_xticks(x, programs, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("time per inference (ms), undiscounted")
    axes[0].legend(
        legend.values(),
        legend.keys(),
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

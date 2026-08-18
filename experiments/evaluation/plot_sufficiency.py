"""fig:sufficiency (e1.csv + sample_census.csv): the distributional half of
E1 -- where every arm's points land inside the measured space.

One panel per (benchmark, size): the shared uniform draw as a horizontal
half-violin -- 300 measured points is a real distribution, where a violin
over 32 seeds would invent shape, which is why the seeds are a strip of
dots instead. Only the upper half is drawn, resting on the x axis: it is
the panel's backdrop, a wash saying where the space is, and the point
arms float over it rather than beside it.

x is slowdown over the panel's space best (best measured point of any arm,
the witness tab:sufficiency prints), on a decade scale so the tails stay
comparable across panels instead of the big sizes flattening everything.
Everything in the panel is plotted in log10 of that ratio on a linear
axis wearing decade ticks, rather than raw ratios on a log axis: the
violin has to be a density *of* log x for its width to mean probability
per unit of the axis it is drawn on. (In linear space the bandwidth is set
by a spread that runs to hundreds, so the body reads as mass out in the
tail when the mass is near 1.)

Overlaid: every search seed's pick, top-k's best, and the transcribed ATiM
point per variant. The one picture carries C1 (the violin body: most of
the space is bad), E1 (each arm lands in the good region), and A3 (the
seed scatter) without being "about" the search.

The full grid is every benchmark x every size -- deliberately oversized
for the paper; the paper version picks one size per benchmark once the
large campaigns land. A (benchmark, size) the campaign has not measured
yet stays in the grid as a labelled empty panel, and a size a benchmark
does not have is marked distinctly: absence should look like absence, not
like a smaller figure. Panels whose sample is still filling say n=k/300
in red, the same provisional marker tab:sufficiency uses (and for the
same reason: the pool is benched in dpus order, so a partial sample is
the low-DPU corner, not a random subsample).

Colors: the three overlay arms are the first three categorical slots of
the reference palette (all-pairs-validated triple); the two ATiM variants
share one hue and differ by marker shape, so identity never rests on a
fourth hue. The distribution body is unstroked neutral gray -- it is the
panel's context, not a series, so it stays the quietest thing in the
panel and the overlay arms carry the ink.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from _reporting import load_or_skip, parse_dirs, save_fig

# Categorical slots 1-3 (validated as an all-pairs triple) + neutrals.
C_SEARCH = "#2a78d6"  # blue: every seed's pick
C_ATIM = "#eb6834"  # orange: transcribed ATiM, both variants (shape splits them)
C_TOPK = "#1baf7a"  # aqua: best measured top-k point
C_BODY = "#c3c2b7"  # neutral: the sample distribution body
C_FAINT = "#898781"  # neutral: axis furniture and the absence markers
C_MUTED = "#52514e"

SIZE_ORDER = ["4MB", "64MB", "256MB", "512MB"]

# (system, size-lane y): the half-violin rests on the x axis, the point arms
# ride above it.
Y_VIOLIN, Y_SEEDS, Y_POINTS = 0.0, 0.60, 0.24
# Full (two-sided) violin width, so the drawn half reaches half of this.
VIOLIN_W = 1.7


def _size_label(fn: str) -> str:
    return fn.rsplit("_", 1)[1]


def main() -> None:
    results_dir, out_dir = parse_dirs("plots")
    df = load_or_skip(results_dir, "e1.csv")
    if df is None:
        return
    census_path = results_dir / "sample_census.csv"
    census = pd.read_csv(census_path) if census_path.exists() else None
    if census is None:
        print("sample_census.csv not assembled -- existence/completeness unknown")

    benches = sorted(df["benchmark"].unique())
    # Which (bench, size) exist at all: the census enumerates every function
    # of the campaign, measured or not; without it, fall back to e1's rows.
    exists = set()
    pool_n = {}
    source = census if census is not None else df
    for row in source.itertuples():
        exists.add((row.benchmark, _size_label(row.fn_name)))
        if census is not None:
            pool_n[(row.benchmark, _size_label(row.fn_name))] = int(row.n_rows)

    fig, axes = plt.subplots(
        len(benches),
        len(SIZE_ORDER),
        figsize=(2.4 * len(SIZE_ORDER), 1.15 * len(benches)),
        sharex=True,
        squeeze=False,
    )
    for ax_row, bench in zip(axes, benches):
        for ax, size in zip(ax_row, SIZE_ORDER):
            ax.set_yticks([])
            ax.set_ylim(0.0, 1.15)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            if (bench, size) not in exists:
                # This benchmark has no function of this size -- structural
                # absence, distinct from a stack that has not run.
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=C_FAINT,
                    fontsize=8,
                )
                ax.spines["bottom"].set_visible(False)
                # "both": the decade scale's minor ticks would otherwise
                # survive the hidden spine as a floating strip of dashes.
                ax.tick_params(bottom=False, labelbottom=False, which="both")
                continue

            sub = df[
                (df["benchmark"] == bench) & (df["fn_name"].str.endswith("_" + size))
            ]
            by = {s: g["total_ms"].to_numpy() for s, g in sub.groupby("system")}
            arms = [by.get(s) for s in ("sample", "topk", "search")]
            arms = [a for a in arms if a is not None and len(a)]
            if not arms:
                ax.text(
                    0.5,
                    0.5,
                    "not run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=C_MUTED,
                    fontsize=8,
                )
                continue
            # The witness: best measured point of any arm, = tab:sufficiency's
            # space best. Everything in the panel is a slowdown over it.
            best = min(a.min() for a in arms)

            ax.axvline(0.0, color=C_FAINT, lw=0.8, ls=":", zorder=2)
            sample = by.get("sample")
            if sample is not None and len(sample) > 1:
                parts = ax.violinplot(
                    np.log10(sample / best),
                    positions=[Y_VIOLIN],
                    vert=False,
                    widths=VIOLIN_W,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    # Clip the mirrored half away: what is left is a ridge
                    # standing on the x axis, under everything else.
                    verts = body.get_paths()[0].vertices
                    verts[:, 1] = np.clip(verts[:, 1], Y_VIOLIN, np.inf)
                    body.set_facecolor(C_BODY)
                    body.set_edgecolor("none")
                    body.set_alpha(0.5)
                    body.set_zorder(0)
                n = len(sample)
                want = pool_n.get((bench, size))
                partial = want is not None and n < want
                ax.text(
                    0.98,
                    0.93,
                    f"n={n}/{want}" if partial else f"n={n}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                    color="red" if partial else C_MUTED,
                )
            search = by.get("search")
            if search is not None and len(search):
                rng = np.random.default_rng(0)  # fixed jitter, stable output
                jitter = rng.uniform(-0.13, 0.13, len(search))
                ax.scatter(
                    np.log10(search / best),
                    Y_SEEDS + jitter,
                    s=9,
                    color=C_SEARCH,
                    alpha=0.7,
                    linewidths=0,
                    zorder=3,
                )
            topk = by.get("topk")
            if topk is not None and len(topk):
                ax.scatter(
                    [np.log10(topk.min() / best)],
                    [Y_POINTS],
                    s=42,
                    marker="D",
                    color=C_TOPK,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=4,
                )
            for system, marker in (
                ("atim_published_transcribed", "^"),
                ("atim_reproduced_transcribed", "v"),
            ):
                pts = by.get(system)
                if pts is not None and len(pts):
                    ax.scatter(
                        [np.log10(pts.min() / best)],
                        [Y_POINTS],
                        s=42,
                        marker=marker,
                        color=C_ATIM,
                        edgecolors="white",
                        linewidths=0.8,
                        zorder=4,
                    )

    for ax, size in zip(axes[0], SIZE_ORDER):
        ax.set_title(size.removesuffix("MB") + " MB", fontsize=9)
    for ax_row, bench in zip(axes, benches):
        ax_row[0].set_ylabel(
            bench.removeprefix("prim_"),
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
        )
    # The axis is linear in log10(slowdown) but wears decade ticks, so it
    # reads as a log axis while the violin's width stays a density per unit
    # of the axis. sharex ties the tick machinery across the whole grid.
    minor = [np.log10(m) + d for d in range(-3, 4) for m in range(2, 10)]
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"$10^{{{v:.0f}}}$")
        )
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor))
    axes[-1][0].set_xlabel("slowdown over space best ($\\times$, log)", fontsize=8)

    handles = [
        plt.matplotlib.patches.Patch(
            facecolor=C_BODY,
            edgecolor="none",
            alpha=0.5,
            label="uniform sample (measured)",
        ),
        plt.matplotlib.lines.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=C_SEARCH,
            markersize=4,
            label="search picks (all seeds)",
        ),
        plt.matplotlib.lines.Line2D(
            [],
            [],
            marker="D",
            ls="",
            color=C_TOPK,
            markersize=6,
            markeredgecolor="white",
            label="top-k best",
        ),
        plt.matplotlib.lines.Line2D(
            [],
            [],
            marker="^",
            ls="",
            color=C_ATIM,
            markersize=6,
            markeredgecolor="white",
            label="ATiM transcribed (published)",
        ),
        plt.matplotlib.lines.Line2D(
            [],
            [],
            marker="v",
            ls="",
            color=C_ATIM,
            markersize=6,
            markeredgecolor="white",
            label="ATiM transcribed (reproduced)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, out_dir, "sufficiency.pdf")
    save_fig(fig, out_dir, "sufficiency.png")


if __name__ == "__main__":
    sys.exit(main())

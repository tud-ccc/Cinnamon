"""Per-class cost profiles: latency against DPU count, and what was allocated.

One panel per graph, one line per program-identity class: the measured cost
of the class's best configuration at each point of the shared-resource menu,
which is exactly the input the graph allocator solves over. The faint dashed
line from each class's smallest device size is perfect strong scaling (slope
-1 on log-log); the gap to it is the diminishing return the allocator trades
between classes, so a class that hugs its reference wants every DPU it can
get and one that flattens early is cheap to shrink.

Markers show the decision: a star at the point each pinned device set
actually runs, a cross where a set was left timeshared. The panel subtitle
carries the graph's allocation summary -- classes, sets, the host/device
split, the achieved objective.

Under each profile is its marginal return: ms saved per DPU added, over the
same device-size axis. That is the convexity test (convex exactly when this
never rises), and it is what a marginal-gain greedy like the latency solve
sees when it decides where to spend the next unit. Convexity cannot be read
off the profile itself -- log-log does not preserve it -- so this panel is
the only honest place to look.

Both panels carry a band when the profile was measured with
`profile-seeds` > 1: the spread of independent searches of the same pinned
space. It matters most on the marginal panel, since differencing amplifies
noise -- a rise there is only evidence of accelerating returns if it clears
the band, which is what the filled versus hollow rings distinguish.

Unlike its sibling plot_*.py scripts this one reads compiler dumps rather
than results/*.csv: a profile is a solver artifact, not an assembled
measurement. Point it at a dump-dir root (the `dump-dir=` given to
upmem-infer-accelerator) and it draws every profiles.csv underneath, with
the allocation.csv, groups.csv and profile_seeds.csv the same graph dumped
beside it.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from _reporting import save_fig  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
MAX_COLS = 3
MAX_TICKS = 6


@dataclasses.dataclass
class Graph:
    """One solved graph: the profiles the allocator was given, and -- when
    the solve got that far -- the summary and the device sets it produced."""

    name: str
    profiles: pd.DataFrame
    alloc: pd.Series | None
    groups: pd.DataFrame | None
    seeds: pd.DataFrame | None

    def spread(self, class_ix: int) -> pd.DataFrame | None:
        """min/max cost per menu point over the repeated searches of one
        class, or None when the profile was measured once (nothing to band)."""
        if self.seeds is None:
            return None
        cls = self.seeds[self.seeds["class"] == class_ix]
        if cls.empty or cls["seed"].nunique() < 2:
            return None
        return cls.groupby("resource")["cost_ms"].agg(["min", "max"])


def _label(row) -> str:
    """A class's legend entry: its debug tag when the IR carries one, its
    index otherwise, with the member count when it stands for several
    blocks."""
    tag = row["debug_tag"]
    name = tag if isinstance(tag, str) and tag else f"class {row['class']}"
    return f"{name} (x{row['multiplicity']})" if row["multiplicity"] > 1 else name


def _summary(alloc: pd.Series) -> str:
    """The allocation.csv row as a subtitle. The host count is a fallback
    count: these are blocks that target the platform but fit no
    configuration, not the host-pinned parts of the program, which are never
    part of this graph."""
    pinned = f"{alloc['n_groups_pinned']} pinned"
    if alloc["n_groups_timeshared"]:
        pinned += f", {alloc['n_groups_timeshared']} timeshared"
    return (
        f"{alloc['n_classes']} classes -> {alloc['n_groups']} sets ({pinned}); "
        f"{alloc['n_blocks_device']}/{alloc['n_blocks']} blocks on "
        f"{alloc['platform']}\n"
        f"{alloc['objective']} {alloc['objective_ms']:.4g} ms, "
        f"{alloc['resource_used']}/{alloc['resource_budget']} units pinned"
    )


def _read(path: pathlib.Path) -> pd.DataFrame | None:
    """A sibling dump, or None when this graph did not get that far (an
    allocation that found nothing feasible leaves profiles.csv and no more)."""
    return pd.read_csv(path) if path.exists() else None


def _collect(paths: list[pathlib.Path]) -> list[Graph]:
    """One Graph per profiles.csv found under `paths`. A path may be the CSV
    itself or any directory above it; the graph name is the directory the
    dump wrote it into, which is what the pass named the graph."""
    files: list[pathlib.Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        else:
            files.extend(sorted(path.rglob("profiles.csv")))
    graphs = []
    for path in files:
        profiles = pd.read_csv(path)
        if profiles.empty:
            continue
        alloc = _read(path.parent / "allocation.csv")
        graphs.append(
            Graph(
                name=path.parent.name,
                profiles=profiles,
                alloc=None if alloc is None or alloc.empty else alloc.iloc[0],
                groups=_read(path.parent / "groups.csv"),
                seeds=_read(path.parent / "profile_seeds.csv"),
            )
        )
    return graphs


def _draw(ax, graph: Graph) -> None:
    # Classes with no feasible configuration dump a measurement-less row
    # (they stay on the host); they have nothing to plot but are worth
    # naming, since a graph that is mostly host tells a story the drawn
    # lines do not.
    drawn = graph.profiles.dropna(subset=["resource", "cost_ms"])
    host = graph.profiles["class"].nunique() - drawn["class"].nunique()
    ticks: set[float] = set()
    color_of_class: dict[int, str] = {}
    # A whole program has more classes than the default cycle has colors, and
    # a repeated color reads as one class measured twice.
    cmap = plt.get_cmap("tab20" if drawn["class"].nunique() > 10 else "tab10")
    for color_ix, (class_ix, cls) in enumerate(drawn.groupby("class")):
        cls = cls.sort_values("resource")
        (line,) = ax.plot(
            cls["resource"],
            cls["cost_ms"],
            marker="o",
            ms=4,
            color=cmap(color_ix % cmap.N),
            label=_label(cls.iloc[0]),
        )
        color_of_class[class_ix] = line.get_color()
        # Where lower-envelope repair replaced a point, the raw measurement
        # is drawn dashed behind the profile: the gap between the two is the
        # cliff a stalled search seed would have cut into the allocation.
        if "raw_cost_ms" in cls and (cls["raw_cost_ms"] > cls["cost_ms"]).any():
            ax.plot(
                cls["resource"],
                cls["raw_cost_ms"],
                linestyle="--",
                lw=0.8,
                alpha=0.5,
                color=line.get_color(),
            )
        # How far independent searches of the same pinned space landed apart.
        # A wiggle in the line that the band covers is search luck, not shape.
        band = graph.spread(class_ix)
        if band is not None:
            ax.fill_between(
                band.index,
                band["min"],
                band["max"],
                color=line.get_color(),
                alpha=0.25,
                lw=0,
            )
        # Perfect strong scaling anchored at the class's smallest device
        # size: cost * resource constant, which both axes being log renders
        # as a straight line of slope -1 through the menu's own x values.
        base = cls.iloc[0]
        ideal = base["cost_ms"] * base["resource"] / cls["resource"]
        ax.plot(
            cls["resource"],
            ideal,
            ls="--",
            lw=0.8,
            alpha=0.4,
            color=line.get_color(),
        )
        ticks.update(cls["resource"])

    # The allocator's choice, on top of the menu it chose from. A set is
    # drawn at the point it runs, which for a timeshared set is its best
    # point rather than anything reserved for it.
    pinned = timeshared = False
    if graph.groups is not None:
        for _, group in graph.groups.iterrows():
            timeshared = timeshared or bool(group["timeshared"])
            pinned = pinned or not group["timeshared"]
            ax.plot(
                group["point_resource"],
                group["cost_ms"],
                marker="X" if group["timeshared"] else "*",
                ms=9 if group["timeshared"] else 13,
                mec="black",
                mew=0.5,
                ls="none",
                color=color_of_class.get(group["class"], "black"),
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # Label a subset of the menu when it is a fine one: a granularity of 64
    # over a 2560-DPU device is 40 values, and every one of them labelled is
    # an unreadable smear. The subset is spaced geometrically, because that
    # is what comes out evenly spread on a log axis.
    menu = sorted(ticks)
    shown = sorted(
        {
            min(menu, key=lambda v: abs(np.log(v / target)))
            for target in np.geomspace(menu[0], menu[-1], MAX_TICKS)
        }
    )
    ax.set_xticks(shown, [f"{int(t)}" for t in shown])
    # Only the menu values are meaningful on x. On y the minor ticks stay: a
    # profile rarely spans a whole decade, and without them the axis carries
    # a single labelled power of ten.
    ax.xaxis.set_minor_locator(NullLocator())
    # The reference lines run below the measurements by construction; bound
    # the axis to the data so they do not squash it.
    ax.set_ylim(drawn["cost_ms"].min() / 1.3, drawn["cost_ms"].max() * 1.3)
    ax.set_xlabel("DPUs pinned")
    ax.set_ylabel("cost of best config (ms)")

    title = graph.name
    if host:
        title += f" ({host} class{'' if host == 1 else 'es'} on the host)"
    if graph.alloc is not None:
        title += "\n" + _summary(graph.alloc)
    ax.set_title(title, fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    marks = [("*", "pinned set")] if pinned else []
    if timeshared:
        marks.append(("X", "timeshared set"))
    for marker, label in marks:
        handles.append(
            Line2D([], [], marker=marker, ls="none", color="grey", mec="black")
        )
        labels.append(label)
    # Outside the axes: a whole program has a class per distinct kernel, and
    # a dozen entries inside the frame would sit on top of the curves.
    ax.legend(handles, labels, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _secants(cls: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(right endpoint, marginal improvement) per segment of one class's
    menu, sorted by device size.

    The marginal improvement is -(dCost/dResource) across the segment: how
    many ms the step off the previous menu entry bought, per DPU it spent.
    Dividing by the actual spacing is what makes an uneven menu comparable
    -- 512->1024 buys more than 512->576 for no interesting reason, and only
    the per-DPU rate says which was the better deal.

    It is also the convexity test. A profile is convex exactly when this is
    non-increasing: every further DPU buys no more than the one before it.
    Convexity is not readable off the profile plot -- log-log does not
    preserve it -- so this is the only honest place to look for it."""
    cls = cls.sort_values("resource")
    r, y = cls["resource"].to_numpy(float), cls["cost_ms"].to_numpy(float)
    return r[1:], -np.diff(y) / np.diff(r)


def _seed_gains(
    graph: Graph, class_ix: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(right endpoints, min gain, max gain) over the repeated searches: the
    same secant computed independently per seed, so the band is what search
    noise does to the marginal returns. Differencing amplifies noise, which
    is exactly why the band belongs on this panel more than on the profile."""
    if graph.seeds is None:
        return None
    cls = graph.seeds[graph.seeds["class"] == class_ix]
    if cls.empty or cls["seed"].nunique() < 2:
        return None
    piv = cls.pivot(index="resource", columns="seed", values="cost_ms")
    piv = piv.sort_index().dropna()
    if len(piv) < 2:
        return None
    r = piv.index.to_numpy(float)
    gains = np.column_stack(
        [-np.diff(piv[s].to_numpy(float)) / np.diff(r) for s in piv.columns]
    )
    return r[1:], gains.min(axis=1), gains.max(axis=1)


def _draw_marginal(ax, graph: Graph) -> None:
    drawn = graph.profiles.dropna(subset=["resource", "cost_ms"])
    cmap = plt.get_cmap("tab20" if drawn["class"].nunique() > 10 else "tab10")
    ticks: set[float] = set()
    convex = total = rises = solid = banded = 0
    for color_ix, (class_ix, cls) in enumerate(drawn.groupby("class")):
        if len(cls) < 2:
            continue  # a one-point menu has no step to price
        at, gain = _secants(cls)
        color = cmap(color_ix % cmap.N)
        ax.plot(at, gain, marker="o", ms=4, color=color)

        band = _seed_gains(graph, class_ix)
        if band is not None:
            ax.fill_between(band[0], band[1], band[2], color=color, alpha=0.25, lw=0)

        # Where the curve rises, a further DPU bought MORE than the one
        # before it: returns accelerating, convexity broken, and the point a
        # marginal-gain greedy can stop short of.
        rising = np.flatnonzero(np.diff(gain) > 0) + 1
        # A rise only means something if it clears the search noise: the
        # segment's worst repeat must still beat its predecessor's best.
        # Without repeats no rise can be qualified, and all of them are drawn
        # as unconfirmed.
        real = np.zeros(len(at), dtype=bool)
        if band is not None and len(band[0]) == len(at):
            real[1:] = band[1][1:] > band[2][:-1]
        for mask, face in (
            (rising[real[rising]], "red"),
            (rising[~real[rising]], "none"),
        ):
            ax.plot(
                at[mask],
                gain[mask],
                ls="none",
                marker="o",
                ms=9,
                mfc=face,
                mec="red",
                mew=1.2,
                alpha=0.9,
            )
        total += 1
        convex += len(rising) == 0
        rises += len(rising)
        solid += int(real[rising].sum())
        banded += band is not None
        ticks.update(at)

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    # Improvements span decades and go negative where a class gets slower
    # with more DPUs, which no log axis takes.
    ax.set_yscale("symlog", linthresh=1e-4)
    menu = sorted(ticks)
    if menu:
        shown = sorted(
            {
                min(menu, key=lambda v: abs(np.log(v / target)))
                for target in np.geomspace(menu[0], menu[-1], MAX_TICKS)
            }
        )
        ax.set_xticks(shown, [f"{int(t)}" for t in shown])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("DPUs pinned (right end of the step)")
    ax.set_ylabel("marginal gain (ms saved per DPU)")
    if banded:
        subtitle = f"{solid} of {rises} accelerations survive the seed band (filled)"
    else:
        subtitle = f"{rises} accelerations, unconfirmed without repeats"
    ax.set_title(
        f"marginal returns -- convex in {convex}/{total} classes\n"
        f"(convex = curve never rises; {subtitle})",
        fontsize=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dumps",
        nargs="*",
        type=pathlib.Path,
        default=[HERE / "data"],
        help="dump-dir roots (or profiles.csv files) to draw",
    )
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "plots")
    args = parser.parse_args()

    roots = args.dumps or [HERE / "data"]
    graphs = _collect(roots)
    if not graphs:
        where = ", ".join(str(p) for p in roots)
        print(f"no profiles.csv dumped yet -- nothing to draw (looked in {where})")
        return

    # Two rows per band of graphs: the profile, and the marginal returns of
    # that same profile directly under it, sharing the device-size axis.
    ncols = min(MAX_COLS, len(graphs))
    bands = -(-len(graphs) // ncols)
    fig, axes = plt.subplots(
        2 * bands,
        ncols,
        figsize=(4.6 * ncols, 7.6 * bands),
        squeeze=False,
        layout="constrained",
    )
    for ix in range(bands * ncols):
        top, bottom = (
            axes[2 * (ix // ncols)][ix % ncols],
            axes[2 * (ix // ncols) + 1][ix % ncols],
        )
        if ix >= len(graphs):
            top.axis("off")
            bottom.axis("off")
            continue
        _draw(top, graphs[ix])
        _draw_marginal(bottom, graphs[ix])

    save_fig(fig, args.out, "profiles.pdf")


if __name__ == "__main__":
    sys.exit(main())

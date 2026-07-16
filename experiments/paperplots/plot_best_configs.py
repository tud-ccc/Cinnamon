#!/usr/bin/env python3
"""
Plot net execution time distributions from aggregated benchmark results across sources.

For each (source, fn_name, seed, iteration):
  net_time = total.elapsed_ns - sum(alloc.elapsed_ns) - sum(free.elapsed_ns)

Then averaged over iterations to get one value per (source, fn_name, seed).

Produces:
  - plots/net_times.csv    combined data for further analysis
  - plots/violin.{pdf,png} violin per problem, one violin per source
  - plots/bars.{pdf,png}   grouped bar chart, mean ± std over seeds

Usage:
  python3 plot_best_configs.py \\
      --exp-root .experiments/tags \\
      --sources prim_red_cinm2_CA prim_red_cinm2_hybrid400 \\
      --out experiments/paperplots/plots/
"""

import argparse
import math
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


SOURCE_LABEL_SUFFIXES = {
    "cinm2_CA": "Cycle accurate",
    "cinm2_CA_200ms": "Cycle accurate (TO 200ms)",
    "cinm2_ca400": "Cycle accurate (TO 400ms)",
    "cinm2_fast": "Fast",
    "cinm2_hybrid200": "Hybrid (TO 200ms)",
    "cinm2_hybrid400": "Hybrid (TO 400ms)",
    "cinm1": "CINM 1.0",
}

SOURCE_LABELS = {
    f"prim_{f}_{suffix}": label
    for f in ["red", "gemv"]
    for suffix, label in SOURCE_LABEL_SUFFIXES.items()
}

FUNC_ORDER = ["4MB", "64MB", "256MB", "512MB"]

NET_SPEEDUP_COLORS = [plt.get_cmap("Dark2")(i) for i in range(8)]


def _iter_col(df: pd.DataFrame) -> str:
    return "iter" if "iter" in df.columns else "iteration"


def _id_col(df: pd.DataFrame) -> str:
    return "seed" if "seed" in df.columns else "config_id"


def load_net_time(agg_dir: pathlib.Path) -> pd.DataFrame:
    """Return DataFrame[fn_name, seed, net_time_ms] (mean over iterations)."""
    total = pd.read_csv(agg_dir / "total.csv").rename(columns={_iter_col: "iteration"})
    alloc = pd.read_csv(agg_dir / "alloc.csv")
    free = pd.read_csv(agg_dir / "free.csv")

    # total uses "iter", alloc/free use "iteration" — normalise
    total = total.rename(columns={_iter_col(total): "iteration"})

    run_id = _id_col(total)
    group3 = ["fn_name", run_id, "iteration"]

    alloc_sum = (
        alloc.groupby(group3, as_index=False)["elapsed_ns"]
        .sum()
        .rename(columns={"elapsed_ns": "alloc_ns"})
    )
    free_sum = (
        free.groupby(group3, as_index=False)["elapsed_ns"]
        .sum()
        .rename(columns={"elapsed_ns": "free_ns"})
    )

    merged = (
        total[["fn_name", run_id, "iteration", "elapsed_ns"]]
        .merge(alloc_sum, on=group3, how="left")
        .merge(free_sum, on=group3, how="left")
    )
    merged[["alloc_ns", "free_ns"]] = merged[["alloc_ns", "free_ns"]].fillna(0)
    merged["net_ns"] = merged["elapsed_ns"] - merged["alloc_ns"] - merged["free_ns"]

    result = (
        merged.groupby(["fn_name", run_id], as_index=False)["net_ns"]
        .mean()
        .rename(columns={"net_ns": "net_time_ms", run_id: "seed"})
    )
    result["net_time_ms"] /= 1e6
    return result


def plot_violins(data: pd.DataFrame, out_dir: pathlib.Path, sources: list[str]):
    funcs = list(data["fn_name"].unique())
    funcs.sort(key=lambda f: (_problem_of(f), *_size_sort_key(f)))
    ncols = 4
    nrows = math.ceil(len(funcs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, fn in zip(axes, funcs):
        sub = data[data["fn_name"] == fn]
        pairs = [
            (i, s, sub[sub["source"] == s]["net_time_ms"].dropna().values)
            for i, s in enumerate(sources)
        ]
        present = [(i, s, v) for i, s, v in pairs if len(v) > 0]
        if not present:
            ax.set_visible(False)
            continue
        positions, _, vals = zip(*present)
        ax.violinplot(vals, positions=positions, showmedians=True)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(
            [SOURCE_LABELS.get(s, s) for _, s, _ in present],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_title(fn)
        if ax in axes[::ncols]:
            ax.set_ylabel("Net time (ms)")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    for ax in axes[len(funcs) :]:
        ax.set_visible(False)

    fig.suptitle("Net execution time per source and problem size")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"violin.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  violin  → {out_dir}/violin.{{pdf,png}}")


def load_bo_timings(bo_timings_dir: pathlib.Path) -> pd.DataFrame:
    """Return DataFrame[fn_name, seed, mean_elapsed_ms] (mean over iters per seed)."""
    csv = bo_timings_dir / "timings.csv"
    df = pd.read_csv(csv)
    result = (
        df.groupby(["fn_name", "seed"], as_index=False)["elapsed_ms"]
        .mean()
        .rename(columns={"elapsed_ms": "mean_elapsed_ms"})
    )
    return result


def _problem_of(fn_name: str) -> str:
    return fn_name.split("_", 1)[0]


def _size_of(fn_name: str) -> str:
    return fn_name.split("_", 1)[1] if "_" in fn_name else fn_name


def _size_sort_key(fn_name: str):
    size = _size_of(fn_name)
    try:
        return (0, FUNC_ORDER.index(size))
    except ValueError:
        return (1, size)


def plot_speedup_bars(
    data: pd.DataFrame,
    out_dir: pathlib.Path,
    sources: list[str],
    value_col: str,
    out_name: str,
    subtitle: str,
    colors: list[str] | None = None,
    ymax: float = 2.0,
):
    """Bar chart of speedup over CINM 1.0, one row per problem (red / gemv)."""
    all_funcs = list(data["fn_name"].unique())
    problems = sorted({_problem_of(f) for f in all_funcs})

    def order_funcs(fns):
        return sorted(fns, key=_size_sort_key)

    funcs_by_problem = {
        p: order_funcs([f for f in all_funcs if _problem_of(f) == p]) for p in problems
    }

    # mean over seeds per (source, fn_name)
    stats = (
        data.groupby(["source", "fn_name"])[value_col]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats["problem"] = stats["fn_name"].map(_problem_of)

    # baseline = CINM 1.0, within each problem — times aren't comparable across problems
    baseline_means = {}
    for p in problems:
        baseline_src = f"prim_{p}_cinm1"
        sub = stats[(stats["problem"] == p) & (stats["source"] == baseline_src)]
        if sub.empty:
            print(f"  WARNING: no {baseline_src} data, skipping problem {p}", file=sys.stderr)
            continue
        baseline_means[p] = sub.set_index("fn_name")["mean"]

    problems = [p for p in problems if p in baseline_means]

    # sources with data for a given problem, in the order given by `sources`
    sources_by_problem = {
        p: [s for s in sources if s in set(stats.loc[stats["problem"] == p, "source"])]
        for p in problems
    }

    fig, axes = plt.subplots(
        len(problems),
        1,
        figsize=(max(6, 2 * max(len(funcs_by_problem[p]) for p in problems)), 4.5 * len(problems)),
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, p in zip(axes, problems):
        funcs = funcs_by_problem[p]
        x = np.arange(len(funcs))
        p_sources = sources_by_problem[p]
        width = 0.8 / max(len(p_sources), 1)
        base = baseline_means[p]
        for i, src in enumerate(p_sources):
            lbl = SOURCE_LABELS.get(src, src)
            sub = stats[(stats["source"] == src) & (stats["problem"] == p)].set_index("fn_name")
            xs, heights, errs, actuals, clipped = [], [], [], [], []
            for j, f in enumerate(funcs):
                if f not in sub.index or f not in base.index:
                    continue
                m = sub.loc[f, "mean"]
                if m <= 0:
                    continue
                s = sub.loc[f, "std"] if not pd.isna(sub.loc[f, "std"]) else 0
                b = base[f]
                speedup = b / m
                # error propagation: d(b/m)/dm = -b/m^2 → relative std passes through
                err = (b / m) * (s / m)
                is_clipped = speedup > ymax
                xs.append(x[j] + (i - len(p_sources) / 2 + 0.5) * width)
                heights.append(ymax if is_clipped else speedup)
                errs.append(0 if is_clipped else err)
                actuals.append(speedup)
                clipped.append(is_clipped)
            color = colors[i % len(colors)] if colors else f"C{i}"
            bars = ax.bar(xs, heights, width * 0.9, yerr=errs, capsize=3, color=color, label=lbl)
            for bar, is_clipped, val in zip(bars, clipped, actuals):
                if not is_clipped:
                    continue
                bar.set_hatch("//")
                ax.annotate(
                    f"{val:.2g}×",
                    xy=(bar.get_x() + bar.get_width() / 2, ymax),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=6,
                    rotation=90,
                )

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(funcs, rotation=30, ha="right")
        ax.set_ylabel("Speedup over CINM 1.0")
        ax.set_title(p)
        ax.legend(fontsize=8)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle(f"Speedup over CINM 1.0 ({subtitle}, mean ± std over seeds)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{out_name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {out_name} → {out_dir}/{out_name}.{{pdf,png}}")


def plot_bars(data: pd.DataFrame, out_dir: pathlib.Path, sources: list[str]):
    """Bar chart of mean net time ± std over seeds, one row per problem (red / gemv)."""
    all_funcs = list(data["fn_name"].unique())
    problems = sorted({_problem_of(f) for f in all_funcs})
    funcs_by_problem = {
        p: sorted([f for f in all_funcs if _problem_of(f) == p], key=_size_sort_key)
        for p in problems
    }

    stats = (
        data.groupby(["fn_name", "source"])["net_time_ms"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats["problem"] = stats["fn_name"].map(_problem_of)

    sources_by_problem = {
        p: [s for s in sources if s in set(stats.loc[stats["problem"] == p, "source"])]
        for p in problems
    }

    fig, axes = plt.subplots(
        len(problems),
        1,
        figsize=(max(6, 2 * max(len(funcs_by_problem[p]) for p in problems)), 4.5 * len(problems)),
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, p in zip(axes, problems):
        funcs = funcs_by_problem[p]
        x = np.arange(len(funcs))
        p_sources = sources_by_problem[p]
        width = 0.8 / max(len(p_sources), 1)
        for i, src in enumerate(p_sources):
            lbl = SOURCE_LABELS.get(src, src)
            sub = stats[(stats["source"] == src) & (stats["problem"] == p)].set_index("fn_name")
            means = [sub.loc[f, "mean"] if f in sub.index else 0 for f in funcs]
            stds = [sub.loc[f, "std"] if f in sub.index else 0 for f in funcs]
            ax.bar(
                x + (i - len(p_sources) / 2 + 0.5) * width,
                means,
                width * 0.9,
                yerr=stds,
                capsize=3,
                color=f"C{i}",
                label=lbl,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(funcs, rotation=30, ha="right")
        ax.set_ylabel("Mean net time (ms)")
        ax.set_title(p)
        ax.legend(fontsize=8)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle("Mean net execution time ± std over seeds")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"bars.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  bars    → {out_dir}/bars.{{pdf,png}}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--exp-root", required=True, help="Path to .experiments/tags/ directory"
    )
    parser.add_argument(
        "--sources", nargs="+", required=True, help="Experiment tag names to include"
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for plots and combined CSV"
    )
    args = parser.parse_args()

    exp_root = pathlib.Path(args.exp_root)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for src in args.sources:
        agg_dir = exp_root / src / "aggregated"
        if not agg_dir.exists():
            print(f"  WARNING: {agg_dir} does not exist, skipping", file=sys.stderr)
            continue
        print(f"  Loading {src}")
        df = load_net_time(agg_dir)
        df["source"] = src
        frames.append(df)

    if not frames:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)
    combined_csv = out_dir / "net_times.csv"
    data.to_csv(combined_csv, index=False)
    print(f"  combined → {combined_csv}")

    sources_present = [s for s in args.sources if s in data["source"].unique()]
    plot_violins(data, out_dir, sources_present)
    plot_bars(data, out_dir, sources_present)
    plot_speedup_bars(
        data,
        out_dir,
        sources_present,
        value_col="net_time_ms",
        out_name="speedup_net",
        subtitle="net execution time",
        colors=NET_SPEEDUP_COLORS,
    )

    bo_frames = []
    for src in args.sources:
        bo_dir = exp_root / src / "bo_timings"
        if not bo_dir.exists():
            continue
        df = load_bo_timings(bo_dir)
        df["source"] = src
        bo_frames.append(df)

    if bo_frames:
        bo_data = pd.concat(bo_frames, ignore_index=True)
        bo_sources = [s for s in args.sources if s in bo_data["source"].unique()]
        plot_speedup_bars(
            bo_data,
            out_dir,
            bo_sources,
            value_col="mean_elapsed_ms",
            out_name="speedup",
            subtitle="BO-search timings",
        )

    print("Done.")


if __name__ == "__main__":
    main()

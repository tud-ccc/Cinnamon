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
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


SOURCE_LABELS = {
    "prim_red_cinm2_CA":        "Cycle accurate",
    "prim_red_cinm2_CA_200ms":  "Cycle accurate (TO 200ms)",
    "prim_red_cinm2_fast":      "Fast",
    "prim_red_cinm2_hybrid200": "Hybrid (TO 200ms)",
    "prim_red_cinm2_hybrid400": "Hybrid (TO 400ms)",
}

FUNC_ORDER = ["red_4MB", "red_64MB", "red_256MB", "red_512MB"]


def _iter_col(df: pd.DataFrame) -> str:
    return "iter" if "iter" in df.columns else "iteration"


def _id_col(df: pd.DataFrame) -> str:
    return "seed" if "seed" in df.columns else "config_id"


def load_net_time(agg_dir: pathlib.Path) -> pd.DataFrame:
    """Return DataFrame[fn_name, seed, net_time_ms] (mean over iterations)."""
    total = pd.read_csv(agg_dir / "total.csv").rename(columns={_iter_col: "iteration"})
    alloc = pd.read_csv(agg_dir / "alloc.csv")
    free  = pd.read_csv(agg_dir / "free.csv")

    # total uses "iter", alloc/free use "iteration" — normalise
    total = total.rename(columns={_iter_col(total): "iteration"})

    run_id = _id_col(total)
    group3 = ["fn_name", run_id, "iteration"]

    alloc_sum = (
        alloc.groupby(group3, as_index=False)["elapsed_ns"]
        .sum().rename(columns={"elapsed_ns": "alloc_ns"})
    )
    free_sum = (
        free.groupby(group3, as_index=False)["elapsed_ns"]
        .sum().rename(columns={"elapsed_ns": "free_ns"})
    )

    merged = (
        total[["fn_name", run_id, "iteration", "elapsed_ns"]]
        .merge(alloc_sum, on=group3, how="left")
        .merge(free_sum,  on=group3, how="left")
    )
    merged[["alloc_ns", "free_ns"]] = merged[["alloc_ns", "free_ns"]].fillna(0)
    merged["net_ns"] = merged["elapsed_ns"] - merged["alloc_ns"] - merged["free_ns"]

    result = (
        merged.groupby(["fn_name", run_id], as_index=False)["net_ns"].mean()
        .rename(columns={"net_ns": "net_time_ms", run_id: "seed"})
    )
    result["net_time_ms"] /= 1e6
    return result


def plot_violins(data: pd.DataFrame, out_dir: pathlib.Path, sources: list[str]):
    funcs = [f for f in FUNC_ORDER if f in data["fn_name"].unique()]
    fig, axes = plt.subplots(1, len(funcs), figsize=(4 * len(funcs), 5), sharey=False)
    if len(funcs) == 1:
        axes = [axes]

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
        ax.set_xticklabels([SOURCE_LABELS.get(s, s) for _, s, _ in present],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(fn)
        if ax is axes[0]:
            ax.set_ylabel("Net time (ms)")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

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
        df.groupby(["fn_name", "seed"], as_index=False)["elapsed_ms"].mean()
        .rename(columns={"elapsed_ms": "mean_elapsed_ms"})
    )
    return result


def plot_speedup_bars(data: pd.DataFrame, out_dir: pathlib.Path, sources: list[str]):
    """Bar chart of speedup over the slowest target, per problem (fn_name)."""
    funcs = [f for f in FUNC_ORDER if f in data["fn_name"].unique()]
    if not funcs:
        funcs = sorted(data["fn_name"].unique())

    # mean over seeds per (source, fn_name)
    stats = (
        data.groupby(["source", "fn_name"])["mean_elapsed_ms"]
        .agg(mean="mean", std="std").reset_index()
    )

    # baseline = source with highest mean time (averaged across all problems)
    overall = stats.groupby("source")["mean"].mean()
    baseline_src = overall.idxmax()
    print(f"  speedup baseline: {baseline_src} (slowest overall)")

    baseline_means = (
        stats[stats["source"] == baseline_src]
        .set_index("fn_name")["mean"]
    )

    x = np.arange(len(funcs))
    width = 0.8 / len(sources)
    labels = [SOURCE_LABELS.get(s, s) for s in sources]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(funcs)), 5))
    for i, (src, lbl) in enumerate(zip(sources, labels)):
        sub = stats[stats["source"] == src].set_index("fn_name")
        speedups, errs = [], []
        for fn in funcs:
            if fn not in sub.index or fn not in baseline_means.index:
                speedups.append(0)
                errs.append(0)
                continue
            base = baseline_means[fn]
            m = sub.loc[fn, "mean"]
            s = sub.loc[fn, "std"] if not pd.isna(sub.loc[fn, "std"]) else 0
            speedups.append(base / m if m > 0 else 0)
            # error propagation: d(base/m)/dm = -base/m^2 → relative std passes through
            errs.append((base / m) * (s / m) if m > 0 else 0)
        ax.bar(
            x + (i - len(sources) / 2 + 0.5) * width,
            speedups, width * 0.9,
            yerr=errs, capsize=3,
            label=lbl,
        )

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(funcs)
    ax.set_ylabel(f"Speedup over {SOURCE_LABELS.get(baseline_src, baseline_src)}")
    ax.set_title("Speedup over slowest target (BO-search timings, mean ± std over seeds)")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"speedup.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  speedup → {out_dir}/speedup.{{pdf,png}}")


def plot_bars(data: pd.DataFrame, out_dir: pathlib.Path, sources: list[str]):
    funcs  = [f for f in FUNC_ORDER if f in data["fn_name"].unique()]
    labels = [SOURCE_LABELS.get(s, s) for s in sources]

    stats = (
        data.groupby(["fn_name", "source"])["net_time_ms"]
        .agg(mean="mean", std="std").reset_index()
    )

    x     = np.arange(len(funcs))
    width = 0.8 / len(sources)

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(funcs)), 5))
    for i, (src, lbl) in enumerate(zip(sources, labels)):
        sub   = stats[stats["source"] == src].set_index("fn_name")
        means = [sub.loc[f, "mean"] if f in sub.index else 0 for f in funcs]
        stds  = [sub.loc[f, "std"]  if f in sub.index else 0 for f in funcs]
        ax.bar(
            x + (i - len(sources) / 2 + 0.5) * width,
            means, width * 0.9,
            yerr=stds, capsize=3,
            label=lbl,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(funcs)
    ax.set_ylabel("Mean net time (ms)")
    ax.set_title("Mean net execution time ± std over seeds")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"bars.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  bars    → {out_dir}/bars.{{pdf,png}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-root", required=True,
                        help="Path to .experiments/tags/ directory")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="Experiment tag names to include")
    parser.add_argument("--out", required=True,
                        help="Output directory for plots and combined CSV")
    args = parser.parse_args()

    exp_root = pathlib.Path(args.exp_root)
    out_dir  = pathlib.Path(args.out)
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
        plot_speedup_bars(bo_data, out_dir, bo_sources)

    print("Done.")


if __name__ == "__main__":
    main()

"""Experiment-specific analysis/plotting for the CINM 1.0 vs CINM 2.0
comparison. Kept separate from experiment.py so the pipeline definition and
the figure-making code don't tangle."""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def geomean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x))))


def print_summary(comparison: pd.DataFrame) -> None:
    print(f"\n{len(comparison)} matched (benchmark, dpus, tasklets) configs\n")
    print(f"{'benchmark':20s} {'n_pairs':>8s} {'geomean speedup':>18s}")
    for fn_name, sub in comparison.groupby("fn_name"):
        print(f"{fn_name:20s} {len(sub):8d} {geomean(sub['speedup']):18.3f}")
    print(f"\n{'OVERALL':20s} {len(comparison):8d} {geomean(comparison['speedup']):18.3f}")


def plot_speedup(comparison: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path:
    """Bar chart of geomean speedup per benchmark, with whiskers showing the
    CINM 2.0 25th-75th percentile spread across its BO search seeds."""
    funcs = sorted(comparison["fn_name"].unique())
    means, lo, hi = [], [], []
    for f in funcs:
        sub = comparison[comparison.fn_name == f]
        means.append(geomean(sub["speedup"]))
        lo.append(geomean(sub["cinm1_ms"] / sub["cinm2_p75"]))
        hi.append(geomean(sub["cinm1_ms"] / sub["cinm2_p25"]))

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(funcs)), 5))
    x = np.arange(len(funcs))
    colors = [plt.get_cmap("Dark2")(i % 8) for i in range(len(funcs))]
    err_lo = np.array(means) - np.array(lo)
    err_hi = np.array(hi) - np.array(means)
    ax.bar(x, means, color=colors, yerr=[err_lo, err_hi], capsize=4)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(funcs, rotation=45, ha="right")
    ax.set_ylabel("Geomean speedup, CINM 2.0 vs CINM 1.0\n(net time, same dpus/tasklets)")
    ax.set_title(
        "CINM 2.0 vs CINM 1.0 codegen at matched hardware configs\n"
        "(bars: CINM2 seed-median; whiskers: CINM2 25th-75th pct across seeds)"
    )
    fig.tight_layout()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cinm1_vs_cinm2_speedup.{ext}", dpi=200)
    print(f"\nWrote {out_dir / 'cinm1_vs_cinm2_speedup.pdf'}")
    return out_dir / "cinm1_vs_cinm2_speedup.pdf"


def _draw_violin(ax, data: list[np.ndarray], positions: np.ndarray, colors: list) -> None:
    """Violin body per position, plus the raw points as a jittered-free
    scatter on top (so a violin backed by few configs still shows its actual
    sample instead of just a KDE blob). Positions with under 2 points are
    skipped by violinplot itself (gaussian_kde needs at least 2 distinct
    values) but still get their point(s) scattered."""
    multi = [i for i, d in enumerate(data) if len(d) >= 2]
    if multi:
        parts = ax.violinplot(
            [data[i] for i in multi],
            positions=positions[multi],
            showmedians=True,
            showextrema=True,
        )
        for i, body in zip(multi, parts["bodies"]):
            body.set_facecolor(colors[i])
            body.set_edgecolor("black")
            body.set_alpha(0.7)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[key].set_color("black")
            parts[key].set_linewidth(1)
    for x, d in zip(positions, data):
        ax.scatter([x] * len(d), d, color="black", s=8, alpha=0.5, zorder=3)


def plot_speedup_violin(comparison: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path:
    """Violin plot of matched-config speedup (same (dpus, tasklets) on both
    sides), one violin per benchmark. Each point is one (dpus, tasklets)
    working group's speedup with CINM 2.0's seed noise collapsed by geomean
    (speedup_seed_geomean, see compare() in dodo.py) -- unlike plot_speedup,
    which also geomeans across configs down to a single bar, the spread shown
    here is the sensitivity of the win to which hardware config was tested,
    not seed noise."""
    funcs = sorted(comparison["fn_name"].unique())
    data = [
        comparison.loc[comparison.fn_name == f, "speedup_seed_geomean"].to_numpy()
        for f in funcs
    ]
    colors = [plt.get_cmap("Dark2")(i % 8) for i in range(len(funcs))]

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(funcs)), 5))
    positions = np.arange(1, len(funcs) + 1)
    _draw_violin(ax, data, positions, colors)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(funcs, rotation=45, ha="right")
    ax.set_ylabel(
        "Speedup, CINM 2.0 vs CINM 1.0\n(net time, same dpus/tasklets, geomean over seeds)"
    )
    ax.set_title(
        "CINM 2.0 vs CINM 1.0 codegen at matched hardware configs\n"
        "(one point per (dpus,tasklets) config tested)"
    )
    fig.tight_layout()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cinm1_vs_cinm2_speedup_violin.{ext}", dpi=200)
    print(f"\nWrote {out_dir / 'cinm1_vs_cinm2_speedup_violin.pdf'}")
    return out_dir / "cinm1_vs_cinm2_speedup_violin.pdf"


def plot_best_speedup(
    comparison_best: pd.DataFrame, out_dir: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    """CINM 2.0 (dpus/tasklets left free to its own search) vs CINM 1.0's
    best time anywhere in its matched-config sweep, per fn_name -- the
    steelmanned-CINM-1.0 comparison, as opposed to plot_speedup{,_violin}'s
    matched-config one. comparison_best has one row per (fn_name, seed),
    where seed indexes CINM 2.0's independent unconstrained search runs (see
    task_cinm2_search_unconstrained in dodo.py), so its spread is run-to-run
    autotuner noise, not config-sweep sensitivity.

    Two views of that same population: a violin showing the full spread, and
    a bar chart collapsing each fn_name down to one geomean -- for when the
    violin is too crowded to read at a glance, at the cost of the
    seed-to-seed detail."""
    funcs = sorted(comparison_best["fn_name"].unique())
    colors = [plt.get_cmap("Dark2")(i % 8) for i in range(len(funcs))]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = [
        comparison_best.loc[comparison_best.fn_name == f, "speedup_vs_cinm1_best"].to_numpy()
        for f in funcs
    ]

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(funcs)), 5))
    positions = np.arange(1, len(funcs) + 1)
    _draw_violin(ax, data, positions, colors)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(funcs, rotation=45, ha="right")
    ax.set_ylabel(
        "Speedup vs CINM 1.0's best config\n(CINM 2.0, dpus/tasklets unconstrained, per search seed)"
    )
    ax.set_title(
        "CINM 2.0 (free search) vs CINM 1.0's best-ever config, per benchmark\n"
        "(spread: independent search seeds)"
    )
    fig.tight_layout()
    violin_path = out_dir / "cinm1_best_vs_cinm2_speedup_violin.pdf"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cinm1_best_vs_cinm2_speedup_violin.{ext}", dpi=200)

    means = [geomean(d) for d in data]
    lo = [np.quantile(d, 0.25) for d in data]
    hi = [np.quantile(d, 0.75) for d in data]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(funcs)), 5))
    x = np.arange(len(funcs))
    err_lo = np.array(means) - np.array(lo)
    err_hi = np.array(hi) - np.array(means)
    ax.bar(x, means, color=colors, yerr=[err_lo, err_hi], capsize=4)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(funcs, rotation=45, ha="right")
    ax.set_ylabel("Geomean speedup vs CINM 1.0's best config\n(CINM 2.0, dpus/tasklets unconstrained)")
    ax.set_title(
        "CINM 2.0 (free search) vs CINM 1.0's best-ever config, per benchmark\n"
        "(bars: geomean across search seeds; whiskers: 25th-75th pct)"
    )
    fig.tight_layout()
    bar_path = out_dir / "cinm1_best_vs_cinm2_speedup_bar.pdf"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cinm1_best_vs_cinm2_speedup_bar.{ext}", dpi=200)

    print(f"\nWrote {violin_path}")
    print(f"Wrote {bar_path}")
    return violin_path, bar_path

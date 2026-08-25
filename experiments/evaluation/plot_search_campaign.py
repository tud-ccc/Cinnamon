"""fig:campaign -- the search-strategy comparison (campaign.csv).

Four figures, all regret against campaign_ref.csv's pooled reference
optimum (regret = best_so_far / ref - 1, floored at 0.1% so log axes
survive exact hits):

- campaign_anytime.pdf: median regret vs evaluations spent, IQR band
  across seeds, one panel per function. The headline figure.
- campaign_anytime_cpu.pdf: the same curves against cumulative CPU
  seconds -- the axis on which a surrogate's fit time has to pay for
  itself.
- campaign_final.pdf: final-regret distribution per arm and function;
  box height is the seed-dependence the campaign exists to reduce.
- campaign_reliability.pdf: fraction of (function, seed) runs within
  epsilon of the reference, pooled and per space-size class. Answers
  "P(a single search is within 5%)", the wholeprogram requirement.
- campaign_bestofk.pdf: expected best-of-k-restarts regret per class --
  whether restarts substitute for a better algorithm.

Arm colors are fixed (identity, never rank) and CVD-validated; `random`
is additionally dashed so the baseline reads without color.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _reporting import load_or_skip, parse_dirs, save_fig

# Fixed arm order and colors (Okabe-Ito subset; adjacent pairs pass the
# CVD floor). Color follows the arm across every figure.
ARMS = ["bananas", "bananas_rand", "random", "descent", "ga"]
ARM_COLOR = {
    "bananas": "#0072B2",
    "bananas_rand": "#E69F00",
    "random": "#009E73",
    "descent": "#D55E00",
    "ga": "#CC79A7",
}
ARM_STYLE = {arm: "--" if arm == "random" else "-" for arm in ARMS}

# Space-size classes (feasible-set size, see docs/SearchStrategyPlan.md).
CLASS_OF = {
    "va": "small (~500)",
    "geva": "small (~500)",
    "red": "small (~500)",
    "gemv": "medium (~40k)",
    "mtv": "medium (~40k)",
    "mmtv": "large (~1M)",
    "ttv": "large (~1M)",
}
CLASS_ORDER = ["small (~500)", "medium (~40k)", "large (~1M)"]

REGRET_FLOOR = 0.1  # percent; exact hits plot here instead of -inf


def fn_class(fn: str) -> str:
    return CLASS_OF[fn.rsplit("_", 1)[0]]


def panel_grid(n: int, ncols: int = 5):
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), squeeze=False
    )
    return fig, [ax for row in axes for ax in row]


def style_axes(ax) -> None:
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def arm_legend(fig, arms) -> None:
    handles = [
        plt.Line2D([], [], color=ARM_COLOR[a], linestyle=ARM_STYLE[a], label=a)
        for a in arms
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(arms), frameon=False)


def regret_pct(df: pd.DataFrame) -> pd.Series:
    return np.maximum(100 * (df["best_so_far"] / df["ref_cost"] - 1), REGRET_FLOOR)


def plot_anytime(df: pd.DataFrame, out_dir, name: str, xcol: str, xlabel: str) -> None:
    fns = sorted(df["fn"].unique(), key=lambda f: (fn_class(f), f))
    arms = [a for a in ARMS if a in set(df["arm"])]
    fig, axes = panel_grid(len(fns))
    for ax, fn in zip(axes, fns):
        sub = df[df["fn"] == fn]
        for arm in arms:
            cur = sub[sub["arm"] == arm]
            if cur.empty:
                continue
            if xcol == "eval_idx":
                grouped = cur.groupby("eval_idx")["regret"]
                x = grouped.median().index.to_numpy()
                med, lo, hi = (
                    grouped.median().to_numpy(),
                    grouped.quantile(0.25).to_numpy(),
                    grouped.quantile(0.75).to_numpy(),
                )
            else:
                # Clocks differ per seed: interpolate each seed's step curve
                # onto a shared log-spaced grid, then take quartiles.
                cur = cur.dropna(subset=[xcol])
                if cur.empty:
                    continue
                x = np.geomspace(max(cur[xcol].min(), 1.0), cur[xcol].max(), 64)
                per_seed = []
                for _, run in cur.groupby("seed"):
                    run = run.sort_values(xcol)
                    per_seed.append(
                        np.interp(
                            x,
                            run[xcol],
                            run["regret"],
                            left=np.nan,
                            right=run["regret"].iloc[-1],
                        )
                    )
                stack = np.vstack(per_seed)
                with np.errstate(all="ignore"):
                    med = np.nanmedian(stack, axis=0)
                    lo = np.nanquantile(stack, 0.25, axis=0)
                    hi = np.nanquantile(stack, 0.75, axis=0)
            ax.plot(
                x, med, color=ARM_COLOR[arm], linestyle=ARM_STYLE[arm], linewidth=1.2
            )
            ax.fill_between(x, lo, hi, color=ARM_COLOR[arm], alpha=0.15, linewidth=0)
        ax.set_yscale("log")
        if xcol != "eval_idx":
            ax.set_xscale("log")
        ax.set_title(fn, fontsize=8)
        ax.tick_params(labelsize=7)
        style_axes(ax)
    for ax in axes[len(fns) :]:
        ax.axis("off")
    fig.supxlabel(xlabel, fontsize=9)
    fig.supylabel("regret vs reference [%] (median, IQR)", fontsize=9)
    arm_legend(fig, arms)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, out_dir, name)
    plt.close(fig)


def plot_final(final: pd.DataFrame, out_dir) -> None:
    fns = sorted(final["fn"].unique(), key=lambda f: (fn_class(f), f))
    arms = [a for a in ARMS if a in set(final["arm"])]
    fig, axes = panel_grid(len(fns))
    for ax, fn in zip(axes, fns):
        sub = final[final["fn"] == fn]
        data, colors = [], []
        for arm in arms:
            vals = sub[sub["arm"] == arm]["regret"].to_numpy()
            data.append(vals if len(vals) else [np.nan])
            colors.append(ARM_COLOR[arm])
        box = ax.boxplot(
            data, patch_artist=True, widths=0.6, medianprops={"color": "black"}
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_linewidth(0.5)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(arms) + 1))
        ax.set_xticklabels(
            [a.replace("bananas_", "b+") for a in arms], fontsize=6, rotation=45
        )
        ax.set_title(fn, fontsize=8)
        ax.tick_params(labelsize=7)
        style_axes(ax)
    for ax in axes[len(fns) :]:
        ax.axis("off")
    fig.supylabel("final regret vs reference [%]", fontsize=9)
    arm_legend(fig, arms)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, out_dir, "campaign_final.pdf")
    plt.close(fig)


def plot_reliability(final: pd.DataFrame, out_dir) -> None:
    arms = [a for a in ARMS if a in set(final["arm"])]
    classes = ["all"] + [c for c in CLASS_ORDER if c in set(final["class"])]
    eps = np.geomspace(REGRET_FLOOR, 300, 128)
    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 2.8))
    axes = np.atleast_1d(axes)
    for ax, cls in zip(axes, classes):
        sub = final if cls == "all" else final[final["class"] == cls]
        for arm in arms:
            vals = sub[sub["arm"] == arm]["regret"].to_numpy()
            if not len(vals):
                continue
            frac = [(vals <= e).mean() for e in eps]
            ax.plot(
                eps,
                frac,
                color=ARM_COLOR[arm],
                linestyle=ARM_STYLE[arm],
                linewidth=1.4,
            )
        ax.axvline(5, color="gray", linewidth=0.6, linestyle=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("regret threshold ε [%]", fontsize=8)
        ax.tick_params(labelsize=7)
        style_axes(ax)
    axes[0].set_ylabel("P(run within ε)", fontsize=9)
    arm_legend(fig, arms)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, out_dir, "campaign_reliability.pdf")
    plt.close(fig)


def plot_bestofk(
    final: pd.DataFrame, out_dir, kmax: int = 8, nboot: int = 1000
) -> None:
    rng = np.random.default_rng(0)
    arms = [a for a in ARMS if a in set(final["arm"])]
    classes = [c for c in CLASS_ORDER if c in set(final["class"])]
    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 2.8))
    axes = np.atleast_1d(axes)
    ks = np.arange(1, kmax + 1)
    for ax, cls in zip(axes, classes):
        sub = final[final["class"] == cls]
        for arm in arms:
            per_fn = []
            for _, grp in sub[sub["arm"] == arm].groupby("fn"):
                vals = grp["regret"].to_numpy()
                if not len(vals):
                    continue
                draws = rng.choice(vals, size=(nboot, kmax))
                per_fn.append(np.minimum.accumulate(draws, axis=1).mean(axis=0))
            if not per_fn:
                continue
            ax.plot(
                ks,
                np.median(np.vstack(per_fn), axis=0),
                color=ARM_COLOR[arm],
                linestyle=ARM_STYLE[arm],
                linewidth=1.4,
                marker="o",
                markersize=3,
            )
        ax.set_yscale("log")
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("independent restarts k", fontsize=8)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=7)
        style_axes(ax)
    axes[0].set_ylabel("E[best-of-k regret] [%] (median over fns)", fontsize=9)
    arm_legend(fig, arms)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, out_dir, "campaign_bestofk.pdf")
    plt.close(fig)


def main() -> None:
    results_dir, out_dir = parse_dirs("plots")
    df = load_or_skip(results_dir, "campaign.csv")
    ref = load_or_skip(results_dir, "campaign_ref.csv")
    if df is None or ref is None:
        return
    df = df.merge(ref[["fn", "ref_cost"]], on="fn")
    df = df.dropna(subset=["best_so_far"])
    df["regret"] = regret_pct(df)
    df["class"] = df["fn"].map(fn_class)

    plot_anytime(df, out_dir, "campaign_anytime.pdf", "eval_idx", "evaluations spent")
    cpu = df.dropna(subset=["cpu_ms"]).assign(cpu_ms=lambda d: d["cpu_ms"] / 1000)
    plot_anytime(
        cpu, out_dir, "campaign_anytime_cpu.pdf", "cpu_ms", "CPU seconds spent"
    )

    final = df.sort_values("eval_idx").groupby(["arm", "fn", "seed"]).tail(1).copy()
    plot_final(final, out_dir)
    plot_reliability(final, out_dir)
    plot_bestofk(final, out_dir)


if __name__ == "__main__":
    sys.exit(main())

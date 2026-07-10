#!/usr/bin/env python3
"""
Landscape analysis for pool.csv from exhaustive search (oracle).

Produces PNG figures in the same directory as the CSV:

  landscape_cost_dist.png     — cost histogram with percentile markers
  landscape_marginals.png     — per-dim mean/IQR of log10(cost) vs dim value
  landscape_roughness.png     — |Δlog10 cost| distribution per axis-aligned step
  landscape_variogram.png     — variogram γ(h) per dimension (correlation length)
  landscape_2d_marginals.png  — pairwise 2-D mean-cost heatmaps (all dim pairs)
  landscape_compensation.png  — iso-product vs axis-aligned Dirichlet energy
  landscape_topk.png          — marginal distribution shift: all → top-k configs

Also prints a text summary with key metrics.

Usage:
    python analyze_landscape.py pool.csv [--top-frac 0.1] [--max-lag 6]
"""
import argparse
import string
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

META_COLS = {"visited", "valid", "cost", "mu", "sigma", "acq", "eval_iter", "eval_time_ms"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pool(csv_path: Path, stats: dict):
    """
    Load pool.csv, return (full_df, observed_good_df, dim_col_names).
    'good' rows are valid=1 and have a finite cost.
    Adds a logcost column (log10 of cost).
    Populates stats with basic dataset metrics.
    """
    df = pd.read_csv(csv_path)
    dim_cols = [c for c in df.columns if c not in META_COLS]

    valid_mask = df["valid"].eq(1) if "valid" in df.columns else pd.Series(True, index=df.index)
    obs_mask   = df["cost"].notna() & np.isfinite(df["cost"])
    good = df[valid_mask & obs_mask].copy()
    good["logcost"] = np.log10(good["cost"])

    n_total = len(df)
    n_valid = int(valid_mask.sum())
    n_obs   = int((valid_mask & obs_mask).sum())
    print(f"Loaded {n_total} configs total: {n_valid} valid, {n_obs} with observed cost")
    print(f"Dimensions ({len(dim_cols)}): {', '.join(dim_cols)}")

    stats.update(
        csv_name     = csv_path.name,
        n_total      = n_total,
        n_valid      = n_valid,
        n_obs        = n_obs,
        n_dims       = len(dim_cols),
        dim_list     = ", ".join(dim_cols),
        cost_min     = f"{good['cost'].min():.3g}",
        cost_max     = f"{good['cost'].max():.3g}",
        logcost_min  = f"{good['logcost'].min():.3f}",
        logcost_max  = f"{good['logcost'].max():.3f}",
    )
    return df, good, dim_cols


def subindex_maps(good, dim_cols):
    """For each dim, map value → 0-based sub-index in sorted unique values."""
    return {d: {v: i for i, v in enumerate(sorted(good[d].unique()))} for d in dim_cols}


# ── Axis-aligned and iso-product pair finders ─────────────────────────────────

def axial_pairs(good, dim_cols, sub_maps, dim, lag=1):
    """
    Return arrays (logcost_a, logcost_b) for all pairs of observed configs that
    differ in `dim` by exactly `lag` sub-index steps and are equal on every other
    dimension.

    Approach: add a sub-index column for `dim`, then self-join on
    (other_dims, sub_index) after shifting the right side by `lag` so that
    the join key for the right row equals the original sub-index minus lag.
    This makes left._sub == shifted_right._sub exactly when the right row's
    original sub-index is left's sub-index + lag.
    """
    other = [c for c in dim_cols if c != dim]
    g = good.copy()
    g["_sub"] = g[dim].map(sub_maps[dim])

    right = g.copy()
    right["_sub"] -= lag  # shift so: right._sub_shifted == left._sub iff orig = left+lag

    merged = g.merge(
        right[other + ["_sub", "logcost"]],
        on=other + ["_sub"],
        suffixes=("_a", "_b"),
    )
    return merged["logcost_a"].values, merged["logcost_b"].values


def isoproduct_pairs(good, dim_cols, dim_i, dim_j):
    """
    Return arrays (logcost_a, logcost_b) for all pairs of observed configs that
    lie on the same iso-product curve dim_i * dim_j = c with all other dims equal,
    but differ in at least one of (dim_i, dim_j).

    These are the "compensating" neighbours: (R, D) ~ (R/n, D*n).
    """
    other = [c for c in dim_cols if c not in (dim_i, dim_j)]
    g = good.copy().reset_index(drop=True)
    g["_id"]   = g.index
    g["_prod"] = g[dim_i] * g[dim_j]

    join_cols = other + ["_prod"]
    merged = g.merge(g[join_cols + ["_id", "logcost"]], on=join_cols, suffixes=("_a", "_b"))
    merged = merged[merged["_id_a"] < merged["_id_b"]]  # deduplicate; drop self-pairs
    return merged["logcost_a"].values, merged["logcost_b"].values


# ── Dirichlet energy (mean squared difference over edges) ─────────────────────

def dirichlet_energy(costs_a, costs_b):
    if len(costs_a) == 0:
        return np.nan
    return float(np.mean((costs_a - costs_b) ** 2))


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _dim_grid(n, max_cols=3):
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _hide_extra(axes_flat, n):
    for ax in axes_flat[n:]:
        ax.set_visible(False)


# ── Figure 1: cost distribution ───────────────────────────────────────────────

def plot_cost_dist(good, out_dir):
    logc = good["logcost"].values
    pcts = [10, 25, 50, 75, 90]
    vals = np.percentile(logc, pcts)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(logc, bins=50, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.4)
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(pcts)))
    for p, v, c in zip(pcts, vals, colors):
        ax.axvline(v, color=c, linewidth=1.4, linestyle="--",
                   label=f"p{p} = {10**v:.2g}")
    ax.set_xlabel("log₁₀(cost)")
    ax.set_ylabel("count")
    ax.set_title(f"Cost distribution  —  {len(good)} valid observed configs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "landscape_cost_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    p10, p90 = np.percentile(logc, 10), np.percentile(logc, 90)
    return {
        "_name": path.name,
        "_log": [
            f"    cost range: [{good['cost'].min():.3g}, {good['cost'].max():.3g}]",
            f"    log10 range: [{logc.min():.3f}, {logc.max():.3f}]",
        ],
        "logcost_p10_p90_gap": f"{p90 - p10:.2f}",
        "cost_p10_p90_ratio":  f"{10 ** (p90 - p10):.1f}",
    }


# ── Figure 2: per-dimension marginals ─────────────────────────────────────────

def plot_marginals(good, dim_cols, out_dir):
    n = len(dim_cols)
    nrows, ncols = _dim_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    _hide_extra(axes.flatten(), n)

    for ax, dim in zip(axes.flatten(), dim_cols):
        vals  = sorted(good[dim].unique())
        grp   = good.groupby(dim)["logcost"]
        means = [grp.get_group(v).mean()         for v in vals]
        q25   = [grp.get_group(v).quantile(0.25) for v in vals]
        q75   = [grp.get_group(v).quantile(0.75) for v in vals]

        ax.plot(vals, means, marker="o", markersize=4, linewidth=1.5, color="steelblue")
        ax.fill_between(vals, q25, q75, alpha=0.2, color="steelblue")
        ax.set_xlabel(dim)
        ax.set_ylabel("log₁₀(cost)")
        ax.set_title(dim)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-dimension marginals  (mean ± IQR, other dims averaged out)", fontsize=12)
    fig.tight_layout()
    path = out_dir / "landscape_marginals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"_name": path.name}


# ── Figure 3: roughness ───────────────────────────────────────────────────────

def plot_roughness(good, dim_cols, sub_maps, out_dir):
    data, means = {}, {}
    for dim in dim_cols:
        a, b = axial_pairs(good, dim_cols, sub_maps, dim, lag=1)
        if len(a):
            data[dim]  = np.abs(a - b)
            means[dim] = data[dim].mean()

    if not data:
        return {"_name": None, "_log": ["  (no axis-aligned pairs found — skipping roughness)"]}

    # sort dims by mean roughness descending
    dims_sorted = sorted(data, key=lambda d: means[d], reverse=True)

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.4), 4))
    positions = list(range(len(dims_sorted)))
    vp = ax.violinplot([data[d] for d in dims_sorted], positions=positions,
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(dims_sorted, rotation=30, ha="right")
    ax.set_ylabel("|Δlog₁₀(cost)| per axis step  (lag=1)")
    ax.set_title("Per-dimension roughness  —  sorted by mean (higher = more rugged)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "landscape_roughness.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    log = ["    mean |Δlog₁₀f| per axis step (sorted):"]
    table_rows = ["| Dimension | Mean \\|Δlog₁₀f\\| | Std | Pairs |",
                  "|-----------|-----------------|-----|-------|"]
    for dim in dims_sorted:
        n_pairs = len(data[dim])
        log.append(f"      {dim:20s}  {means[dim]:.4f} ± {data[dim].std():.4f}  (n={n_pairs})")
        table_rows.append(
            f"| `{dim}` | {means[dim]:.4f} | {data[dim].std():.4f} | {n_pairs} |"
        )
    return {
        "_name": path.name,
        "_log": log,
        "roughness_table": "\n".join(table_rows),
    }


# ── Figure 4: variogram ───────────────────────────────────────────────────────

def plot_variogram(good, dim_cols, sub_maps, out_dir, max_lag=6):
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 4))

    for ci, dim in enumerate(dim_cols):
        max_sub = max(sub_maps[dim].values()) if sub_maps[dim] else 0
        lags, gammas = [], []
        for h in range(1, min(max_lag + 1, max_sub + 1)):
            a, b = axial_pairs(good, dim_cols, sub_maps, dim, lag=h)
            if len(a) >= 5:
                lags.append(h)
                gammas.append(np.mean((a - b) ** 2) / 2.0)
        if lags:
            ax.plot(lags, gammas, marker="o", markersize=4, linewidth=1.5,
                    color=cmap(ci / max(len(dim_cols), 1)), label=dim)

    ax.set_xlabel("Lag  h  (discrete sub-index steps)")
    ax.set_ylabel("γ(h)  =  ½ · E[(Δlog₁₀f)²]")
    ax.set_title("Variogram per dimension\n"
                 "Flat curve → no correlation beyond lag 1.  "
                 "Rising curve → long correlation length.")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "landscape_variogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"_name": path.name}


# ── Figure 5: 2-D marginal heatmaps ──────────────────────────────────────────

def plot_2d_marginals(good, dim_cols, out_dir):
    pairs = list(combinations(dim_cols, 2))
    if not pairs:
        return
    nrows, ncols = _dim_grid(len(pairs), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    _hide_extra(axes.flatten(), len(pairs))

    vmin, vmax = good["logcost"].min(), good["logcost"].max()

    for ax, (di, dj) in zip(axes.flatten(), pairs):
        piv = (good.pivot_table(index=di, columns=dj, values="logcost", aggfunc="mean")
               .sort_index(ascending=True))
        piv = piv[sorted(piv.columns)]

        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis_r",
                       norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                       interpolation="nearest")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_yticks(range(len(piv.index)))
        ax.set_xticklabels(list(piv.columns), rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(list(piv.index), fontsize=6)
        ax.set_xlabel(dj, fontsize=9)
        ax.set_ylabel(di, fontsize=9)
        ax.set_title(f"{di} × {dj}", fontsize=10)
        plt.colorbar(im, ax=ax, label="mean log₁₀ cost", fraction=0.046)

    fig.suptitle("2-D marginals  (mean log₁₀ cost, other dims averaged out)\n"
                 "Shared colour scale — panels are directly comparable.", fontsize=11)
    fig.tight_layout()
    path = out_dir / "landscape_2d_marginals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"_name": path.name}


# ── Figure 6: compensating dimension analysis ─────────────────────────────────

def plot_compensation(good, dim_cols, sub_maps, out_dir):
    """
    For each dim pair (di, dj), compare:
      - Axis-aligned Dirichlet energy: mean (Δlog10f)² over edges where di or dj
        changes by one step (all other dims fixed).
      - Iso-product Dirichlet energy: mean (Δlog10f)² over pairs where di*dj is
        constant (all other dims fixed) but (di, dj) differ — i.e. moves like
        (R, D) → (R/n, D*n).

    If E_prod / E_axis < 1, the cost is smoother along iso-product curves than
    along grid edges → the product di*dj is a more natural coordinate than the
    individual dimensions.

    Also reports: explained variance = 1 − mean_Var[f|di*dj=c] / Var[f].
    High explained variance means knowing the product predicts cost well.
    """
    pairs = list(combinations(dim_cols, 2))
    if not pairs:
        return

    rows = []
    global_var = good["logcost"].var()

    for di, dj in pairs:
        # Axis-aligned energy for the two dims combined
        ai, bi = axial_pairs(good, dim_cols, sub_maps, di, lag=1)
        aj, bj = axial_pairs(good, dim_cols, sub_maps, dj, lag=1)
        if len(ai) + len(aj) == 0:
            continue
        e_axis = dirichlet_energy(np.concatenate([ai, aj]), np.concatenate([bi, bj]))

        # Iso-product energy
        ap, bp = isoproduct_pairs(good, dim_cols, di, dj)
        e_prod = dirichlet_energy(ap, bp)

        # Conditional variance: Var[logcost | di*dj=c], averaged over c
        g = good.copy()
        g["_prod"] = g[di] * g[dj]
        cond_var = g.groupby("_prod")["logcost"].var().mean()
        explained = 1.0 - cond_var / global_var if global_var > 0 else np.nan

        rows.append(dict(
            pair=f"{di}×{dj}",
            di=di, dj=dj,
            e_axis=e_axis,
            e_prod=e_prod,
            ratio=e_prod / e_axis if (e_axis and not np.isnan(e_axis)) else np.nan,
            n_axis=len(ai) + len(aj),
            n_prod=len(ap),
            explained=explained,
        ))

    if not rows:
        return {"_name": None, "_log": ["  (no pairs had enough data — skipping compensation)"]}

    results = pd.DataFrame(rows).sort_values("ratio")

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(14, max(4, len(rows) * 0.55 + 2)))

    # ── Left: energy ratio bar chart ──────────────────────────────────────────
    colors = ["steelblue" if r < 1 else "salmon" for r in results["ratio"]]
    ax_bar.barh(results["pair"], results["ratio"], color=colors, alpha=0.8)
    ax_bar.axvline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax_bar.set_xlabel("E_prod / E_axis\n< 1 → smoother along iso-product curve")
    ax_bar.set_title("Dirichlet energy ratio\n(sorted; blue = compensation detected)")
    ax_bar.grid(True, axis="x", alpha=0.3)

    # Annotate with explained-variance
    for i, (_, row) in enumerate(results.iterrows()):
        ax_bar.text(row["ratio"] + 0.01, i,
                    f"R²={row['explained']:.2f}", va="center", fontsize=7)

    # ── Right: scatter E_axis vs E_prod, y=x diagonal ────────────────────────
    ax_scatter.scatter(results["e_axis"], results["e_prod"], s=60, color="steelblue",
                       zorder=3, alpha=0.8)
    for _, row in results.iterrows():
        ax_scatter.annotate(
            row["pair"],
            (row["e_axis"], row["e_prod"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
        )
    lim = max(results[["e_axis", "e_prod"]].max()) * 1.08
    ax_scatter.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="ratio = 1")
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim(bottom=0)
    ax_scatter.set_xlabel("Axis-aligned Dirichlet energy  E_axis")
    ax_scatter.set_ylabel("Iso-product Dirichlet energy  E_prod")
    ax_scatter.set_title("Points below diagonal → smoother on iso-product graph")
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(True, alpha=0.3)

    fig.suptitle("Compensating dimension analysis", fontsize=12)
    fig.tight_layout()
    path = out_dir / "landscape_compensation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    log = [f"    {'Pair':25s}  {'Ratio':>7s}  {'R²(prod)':>9s}  "
           f"{'n_axis':>8s}  {'n_prod':>8s}"]
    table_rows = ["| Pair | E_prod/E_axis | R² (product) | n_axis | n_prod |",
                  "|------|-------------|--------------|--------|--------|"]
    for _, row in results.iterrows():
        flag = "  ← strong" if row["ratio"] < 0.5 else ""
        log.append(f"    {row['pair']:25s}  {row['ratio']:7.3f}  {row['explained']:9.3f}"
                   f"  {row['n_axis']:8.0f}  {row['n_prod']:8.0f}{flag}")
        marker = " ★" if row["ratio"] < 0.5 and row["explained"] > 0.7 else ""
        table_rows.append(
            f"| `{row['pair']}` | {row['ratio']:.3f} | {row['explained']:.3f}"
            f" | {row['n_axis']:.0f} | {row['n_prod']:.0f} |{marker}"
        )
    return {
        "_name": path.name,
        "_log": log,
        "compensation_table": "\n".join(table_rows),
    }


# ── Figure 7: top-k concentration ────────────────────────────────────────────

def plot_topk(good, dim_cols, out_dir, top_frac):
    """
    For each dimension, compare the marginal distribution of dim values between:
      - all valid observed configs
      - the top top_frac fraction by cost (lowest cost = best)

    A large shift in a dimension's distribution signals that good configs
    cluster at specific values of that dimension.
    """
    top_n = max(1, int(len(good) * top_frac))
    top   = good.nsmallest(top_n, "cost")

    n = len(dim_cols)
    nrows, ncols = _dim_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    _hide_extra(axes.flatten(), n)

    for ax, dim in zip(axes.flatten(), dim_cols):
        vals = sorted(good[dim].unique())
        x    = np.arange(len(vals))
        w    = 0.35
        all_f = np.array([good[dim].eq(v).sum() / len(good) for v in vals])
        top_f = np.array([top[dim].eq(v).sum()  / len(top)  for v in vals])

        ax.bar(x - w / 2, all_f, w, label="all",            color="steelblue", alpha=0.75)
        ax.bar(x + w / 2, top_f, w, label=f"top {top_frac:.0%}", color="tomato",    alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("fraction")
        ax.set_title(dim)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Top-{top_frac:.0%} concentration  "
                 f"(n_top = {top_n} / n_all = {len(good)})\n"
                 "Large shift → good configs concentrate at specific dim values.", fontsize=11)
    fig.tight_layout()
    path = out_dir / "landscape_topk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"_name": path.name, "top_frac_pct": f"{top_frac:.0%}", "n_top": top_n}


# ── README generation ─────────────────────────────────────────────────────────

def generate_readme(stats: dict, out_dir: Path):
    template_path = Path(__file__).parent / "README_landscape_template.md"
    if not template_path.exists():
        print(f"  (template not found at {template_path} — skipping README)")
        return
    tmpl = string.Template(template_path.read_text())
    # Provide safe defaults so missing stats don't abort the substitution.
    filled = tmpl.safe_substitute(stats)
    out_path = out_dir / "README_landscape.md"
    out_path.write_text(filled)
    print(f"  {out_path.name}")


# ── Per-problem analysis (module-level so ProcessPoolExecutor can pickle it) ───

def _analyze_csv(csv_path: Path, top_frac: float, max_lag: int, out_dir: Path) -> str:
    """Full landscape analysis for one pool.csv. Plots run sequentially.
    Returns the problem name. Safe to call from a subprocess."""
    import os as _os
    print(f"\n[{_os.getpid()}] === {csv_path.parent.name}")

    stats: dict = {}
    _, good, dim_cols = load_pool(csv_path, stats)
    if good.empty:
        print(f"  WARNING: no valid observed configs — skipping {csv_path}")
        return csv_path.parent.name

    sub_maps = subindex_maps(good, dim_cols)

    stats.setdefault("roughness_table",    "_No axis-aligned pairs found._")
    stats.setdefault("compensation_table", "_Not enough data for compensation analysis._")
    stats.setdefault("top_frac_pct",       f"{top_frac:.0%}")
    stats.setdefault("n_top",              max(1, int(len(good) * top_frac)))

    tasks = [
        ("cost_dist",    plot_cost_dist,    (good, out_dir)),
        ("marginals",    plot_marginals,    (good, dim_cols, out_dir)),
        ("roughness",    plot_roughness,    (good, dim_cols, sub_maps, out_dir)),
        ("variogram",    plot_variogram,    (good, dim_cols, sub_maps, out_dir, max_lag)),
        ("2d_marginals", plot_2d_marginals, (good, dim_cols, out_dir)),
        ("compensation", plot_compensation, (good, dim_cols, sub_maps, out_dir)),
        ("topk",         plot_topk,         (good, dim_cols, out_dir, top_frac)),
    ]
    for name, fn, fn_args in tasks:
        try:
            result = fn(*fn_args) or {}
            for line in result.pop("_log", []):
                print(line)
            png = result.pop("_name", None)
            if png:
                print(f"  Saved: {png}")
            stats.update(result)
        except Exception as ex:
            print(f"  ERROR in {name}:")
            traceback.print_exception(ex)

    generate_readme(stats, out_dir)
    return csv_path.parent.name


# ── Main ──────────────────────────────────────────────────────────────────────

def _collect_csvs(path: Path) -> list[Path]:
    """Resolve a path argument to a list of pool.csv files to analyse."""
    if path.suffix == ".csv":
        return [path]
    direct = path / "pool.csv"
    if direct.exists():
        # Directory that IS a problem dir (has pool.csv directly)
        return [direct]
    # Directory of problem subdirs — walk one level down
    found = sorted(path.glob("*/pool.csv"))
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Landscape analysis for exhaustive-search pool.csv.\n\n"
                    "PATH may be:\n"
                    "  • a pool.csv file       → analyse that one problem\n"
                    "  • a problem directory   → analyse pool.csv inside it\n"
                    "  • a parent directory    → analyse all */pool.csv in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in-dir", dest="in_dir", required=True, metavar="DIR",
                        help="pool.csv, problem dir, or parent dir of problems")
    parser.add_argument("--out-dir", dest="out_dir", required=True, metavar="DIR",
                        help="Output directory for plots and README")
    parser.add_argument("--top-frac", type=float, default=0.05,
                        help="Fraction of best configs to highlight (default: 0.05)")
    parser.add_argument("--max-lag", type=int, default=6,
                        help="Maximum variogram lag in sub-index steps (default: 6)")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="Worker processes for multi-problem mode (default: n_problems)")
    args = parser.parse_args()

    csv_paths = _collect_csvs(Path(args.in_dir))
    if not csv_paths:
        sys.exit(f"No pool.csv found under {args.in_dir}")

    if len(csv_paths) == 1:
        # Single problem: run plots in parallel (existing behaviour)
        csv_path = csv_paths[0]
        out_dir  = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Landscape analysis: {csv_path}")
        print(f"=== Output directory:   {out_dir}\n")

        stats: dict = {}
        _, good, dim_cols = load_pool(csv_path, stats)
        if good.empty:
            sys.exit("No valid observed configs in CSV — run exhaustive search first.")

        sub_maps = subindex_maps(good, dim_cols)
        stats.setdefault("roughness_table",    "_No axis-aligned pairs found._")
        stats.setdefault("compensation_table", "_Not enough data for compensation analysis._")
        stats.setdefault("top_frac_pct",       f"{args.top_frac:.0%}")
        stats.setdefault("n_top",              max(1, int(len(good) * args.top_frac)))

        plot_tasks = {
            "cost_dist":    (plot_cost_dist,    (good, out_dir)),
            "marginals":    (plot_marginals,    (good, dim_cols, out_dir)),
            "roughness":    (plot_roughness,    (good, dim_cols, sub_maps, out_dir)),
            "variogram":    (plot_variogram,    (good, dim_cols, sub_maps, out_dir, args.max_lag)),
            "2d_marginals": (plot_2d_marginals, (good, dim_cols, out_dir)),
            "compensation": (plot_compensation, (good, dim_cols, sub_maps, out_dir)),
            "topk":         (plot_topk,         (good, dim_cols, out_dir, args.top_frac)),
        }
        futures = {}
        with ProcessPoolExecutor() as executor:
            for name, (fn, fn_args) in plot_tasks.items():
                futures[executor.submit(fn, *fn_args)] = name
            for future in tqdm(as_completed(futures), total=len(futures), desc="plots"):
                name = futures[future]
                ex = future.exception()
                if ex:
                    tqdm.write(f"  ERROR in {name}:")
                    tqdm.write("".join(traceback.format_exception(ex)))
                else:
                    result = future.result() or {}
                    for line in result.pop("_log", []):
                        tqdm.write(line)
                    png = result.pop("_name", None)
                    if png:
                        tqdm.write(f"  Saved: {png}")
                    stats.update(result)

        print("\n[README]")
        generate_readme(stats, out_dir)

    else:
        # Multiple problems: one process per problem, plots sequential inside each
        workers = args.workers or len(csv_paths)
        print(f"=== Landscape analysis: {len(csv_paths)} problems, {workers} workers")
        for p in csv_paths:
            print(f"    {p}")
        print()

        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for p in csv_paths:
                p_out = Path(args.out_dir) / p.parent.name
                p_out.mkdir(parents=True, exist_ok=True)
                futures[executor.submit(_analyze_csv, p, args.top_frac, args.max_lag, p_out)] = p
            for future in tqdm(as_completed(futures), total=len(futures), desc="problems"):
                p = futures[future]
                ex = future.exception()
                if ex:
                    tqdm.write(f"  ERROR in {p.parent.name}:")
                    tqdm.write("".join(traceback.format_exception(ex)))
                else:
                    tqdm.write(f"  Done: {future.result()}")

    print("\nDone.")


if __name__ == "__main__":
    main()

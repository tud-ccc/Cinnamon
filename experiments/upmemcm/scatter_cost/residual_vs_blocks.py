#!/usr/bin/env python3
"""
Diagnostic for the scatter:sg cost model: does the *current* per-region
regression's residual have structure in `blocks_per_dpu` specifically?

Reconstructs the exact model baked into upmem_cost_model's scatterSgCostMs
LUT (same splits as dodo.py's sg bench_splits, same per-region "lasso"
feature set fit via plain OLS -- see analyze.py's main() --split path) via
fit_regime_hybrid, then plots relative residual (pred-measured)/measured
against blocks_per_dpu, faceted per production region, colored by
block_size. blocks_per_dpu is swept exhaustively (1-24, no gaps) in every
region, so any trend visible here is a real model-shape problem, not a
sampling-density artifact.

Usage:
  python3 residual_vs_blocks.py plots/sg/results_agg.csv
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze import (
    PROBLEM,
    EXTRA_TEMPLATES,
    build_templates,
    combined_regime_masks,
    fit_regime_hybrid,
    _cmap_for,
    _norm_for,
)

SPLITS = {"block_size": [1023], "num_dpus": [16, 24, 64, 128, 256, 384]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--out", default="plots/sg/residual_vs_blocks_per_dpu.png")
    args = parser.parse_args()

    agg = pd.read_csv(args.csv)
    PROBLEM.add_derived_columns(agg)
    templates = build_templates(PROBLEM, extra=EXTRA_TEMPLATES)

    masks, labels = combined_regime_masks(agg, SPLITS)
    y = agg["ms"].to_numpy(dtype=float)
    # Production LUT only ever tries "lasso" per region (see main()'s
    # mask_templates), so it always wins trivially -- reproduce that instead
    # of re-running the region-by-region comparison.
    regime_keys = ["lasso"] * len(masks)
    hybrid = fit_regime_hybrid(templates, agg, y, SPLITS, regime_keys)
    pred = hybrid["pred"]
    rel_resid = (pred - y) / y  # negative = underestimate, positive = overestimate

    print(
        f"Reconstructed hybrid: relRMSE={hybrid['rel_rmse'] * 100:.2f}%  "
        f"(matches log.log's {16.00}% for the full sg LUT)"
    )

    n = len(masks)
    ncols = min(7, n)
    nrows = (n + ncols - 1) // ncols

    color_col = "block_size"
    color_dim = next(d for d in PROBLEM.all_dims if d.col == color_col)
    color_norm, _ = _norm_for(color_dim, agg[color_col])
    cmap = _cmap_for(color_dim, default="plasma")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.8 * ncols, 2.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    sc = None
    for i, (mask, label) in enumerate(zip(masks, labels)):
        ax = axes[divmod(i, ncols)[0]][divmod(i, ncols)[1]]
        if not mask.any():
            ax.axis("off")
            continue
        xv = agg.loc[mask, "blocks_per_dpu"]
        rv = rel_resid[mask]
        cv = agg.loc[mask, color_col]

        # Per-blocks_per_dpu median trend, to see the shape through the
        # per-(num_dpus,block_size) scatter within the region.
        trend = pd.DataFrame({"x": xv.to_numpy(), "r": rv}).groupby("x")["r"].median()

        sc = ax.scatter(xv, rv, s=4, alpha=0.25, c=cv, cmap=cmap, norm=color_norm)
        ax.plot(
            trend.index, trend.to_numpy(), color="black", linewidth=1.5, label="median"
        )
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
        corr = np.corrcoef(xv, rv)[0, 1]
        ax.set_title(f"{label}\ncorr(blocks_per_dpu, relresid)={corr:.2f}", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[divmod(j, ncols)[0]][divmod(j, ncols)[1]].axis("off")

    fig.supxlabel("blocks_per_dpu")
    fig.supylabel("(pred - measured) / measured")
    fig.suptitle(
        "scatter:sg current-model relative residual vs. blocks_per_dpu, per region"
    )
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label="block_size")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

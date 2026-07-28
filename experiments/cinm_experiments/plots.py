"""Shared measured-vs-predicted calibration scatter, factored out of the
cost-model validation experiments (cost_bench, scatter_cost) that each used
to draw their own near-identical version of it -- log-log scatter, y=x
reference line, optional colorbar, optional stats text box. Only the
rendering is shared; fitting/metric computation (NNLS, Spearman rho, RMSE,
...) stays with each caller, since what's being fit and how differs per
experiment."""
from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np


def plot_measured_vs_predicted(
    predicted,
    measured,
    *,
    out_path: pathlib.Path | None = None,
    ax=None,
    color=None,
    color_label: str | None = None,
    cmap: str = "viridis",
    log_color: bool = True,
    norm=None,
    cbar_ticks: Sequence[float] | None = None,
    xlabel: str = "predicted",
    ylabel: str = "measured",
    title: str | None = None,
    annotate_lines: Sequence[str] | None = None,
    lim: tuple[float, float] | None = None,
    legend: bool = True,
):
    """Log-log scatter of measured vs predicted values with a y=x reference
    line, optionally colored by `color`, plus an optional stats text box
    (`annotate_lines`, rendered verbatim -- computing those stats is the
    caller's job).

    Draws onto `ax` if given, for embedding into a multi-panel figure (e.g.
    scatter_cost's plot_all_regression_fits, which shares one colorbar per
    row across several calls) -- the caller then owns the colorbar/figure
    saving. Otherwise creates its own fig/ax, draws a colorbar if `color` is
    given, and saves to `out_path` (required in that case).

    Returns the scatter PathCollection (or None if `color` was None), so a
    caller drawing onto a shared `ax` can build its own colorbar from it.
    """
    predicted = np.asarray(predicted, dtype=float)
    measured = np.asarray(measured, dtype=float)

    own_fig = ax is None
    fig = None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    if lim is not None:
        lo, hi = lim
    else:
        positive = np.concatenate([predicted[predicted > 0], measured[measured > 0]])
        pad = 1.15
        lo, hi = positive.min() / pad, positive.max() * pad

    sc = None
    if color is not None:
        color = np.asarray(color, dtype=float)
        if norm is None:
            norm = LogNorm(vmin=color.min(), vmax=color.max()) if log_color else Normalize()
        sc = ax.scatter(predicted, measured, s=10, alpha=0.7, c=color, cmap=cmap, norm=norm)
    else:
        ax.scatter(predicted, measured, s=10, alpha=0.7, color="steelblue")

    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    if legend:
        ax.legend(fontsize=8)

    if annotate_lines:
        ax.text(0.03, 0.97, "\n".join(annotate_lines),
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    if own_fig:
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, label=color_label, ticks=cbar_ticks)
            if log_color:
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        fig.tight_layout()
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return sc


def log2_ticks(values) -> list[float]:
    """Power-of-2 tick values spanning `values`' range -- the recurring
    colorbar tick scheme for DPU-count-colored calibration plots."""
    values = np.asarray(values, dtype=float)
    k_min = int(np.floor(np.log2(values.min())))
    k_max = int(np.ceil(np.log2(values.max())))
    return [2 ** k for k in range(k_min, k_max + 1)]

#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FREQ_HZ = 350e6  # 350 MHz

# Figure layout — reduce WIDTH_PER_TASKLET to pull x values closer together
WIDTH_PER_TASKLET = 0.3  # inches per tasklet on the x axis
FIG_HEIGHT = 5.0  # inches


def ns_to_cycles(ns):
    return ns * FREQ_HZ * 1e-9


csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
sim_csv_file = sys.argv[2] if len(sys.argv) > 2 else "sim_results.csv"

df = pd.read_csv(csv_file)
df_sim = pd.read_csv(sim_csv_file)

tasklet_counts = sorted(df["tasklets"].unique())
multi_tasklets = [t for t in tasklet_counts if t > 1]

# Real data: per-trial values and medians
box_data = [
    ns_to_cycles(df[df["tasklets"] == t]["overhead_ns_per_barrier"].values)
    for t in tasklet_counts
]
medians = pd.Series(
    {
        t: np.median(ns_to_cycles(df[df["tasklets"] == t]["overhead_ns_per_barrier"]))
        for t in tasklet_counts
    }
)

# Sim data: single value per tasklet
sim_values = pd.Series(
    {
        t: ns_to_cycles(
            df_sim[df_sim["tasklets"] == t]["overhead_ns_per_barrier"].values[0]
        )
        for t in tasklet_counts
    }
)

# ── helpers ──────────────────────────────────────────────────────────────────


def fit_segment(tasklets, values):
    positions = [tasklet_counts.index(t) + 1 for t in tasklets]
    slope, intercept, r, *_ = stats.linregress(positions, values[tasklets].values)
    return slope, intercept, r**2


def find_best_cutoff(candidates, values, label=""):
    print(f"\n── cutoff search{' (' + label + ')' if label else ''} ──")
    print(f"{'cutoff':>8}  {'R²(lo)':>8}  {'R²(hi)':>8}  {'R²(weighted)':>13}")
    best_cutoff, best_r2 = None, -1
    for cutoff in candidates[1:-1]:
        lo = [t for t in candidates if t <= cutoff]
        hi = [t for t in candidates if t > cutoff]
        if len(lo) < 2 or len(hi) < 2:
            continue
        _, _, r2_lo = fit_segment(lo, values)
        _, _, r2_hi = fit_segment(hi, values)
        r2_w = (len(lo) * r2_lo + len(hi) * r2_hi) / len(candidates)
        marker = ""
        if r2_w > best_r2:
            best_r2, best_cutoff = r2_w, cutoff
            marker = " <"
        print(f"{cutoff:>8}  {r2_lo:>8.4f}  {r2_hi:>8.4f}  {r2_w:>13.4f}{marker}")
    print(f"Best cutoff: {best_cutoff}  (weighted R²={best_r2:.4f})")
    return best_cutoff


def draw_regression(ax, tasklets, values, color, label):
    positions = [tasklet_counts.index(t) + 1 for t in tasklets]
    slope, intercept, r2 = fit_segment(tasklets, values)
    xs = np.linspace(positions[0], positions[-1], 200)
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.5, label=label)
    return slope, intercept, r2


def add_annotations(ax, annotations):
    for i, (color, label, val_or_slope, intercept, r2) in enumerate(annotations):
        if intercept is None:
            text = f"{label}:  {val_or_slope:.2f} cycles"
        else:
            text = f"{label}:  slope={val_or_slope:.3f} cyc/tasklet,  intercept={intercept:.2f} cyc,  R²={r2:.4f}"
        ax.text(
            0.01,
            0.97 - i * 0.08,
            text,
            transform=ax.transAxes,
            color=color,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )


def align_zeros(ax_left, ax_right):
    l_bot, l_top = ax_left.get_ylim()
    r_bot, r_top = ax_right.get_ylim()
    f = -l_bot / (l_top - l_bot)  # fraction from bottom where 0 sits on left axis
    r_range = max(r_top / (1 - f) if f < 1 else 0, -r_bot / f if f > 0 else 0)
    ax_right.set_ylim(-f * r_range, (1 - f) * r_range)


def add_separators(ax, cutoff):
    ax.axvline(tasklet_counts.index(1) + 1.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(
        tasklet_counts.index(cutoff) + 1.5, color="gray", linestyle=":", linewidth=1
    )


# ── plot 1: real measurements (box plots) ────────────────────────────────────

best_cutoff = find_best_cutoff(multi_tasklets, medians, label="real")
lo = [t for t in multi_tasklets if t <= best_cutoff]
hi = [t for t in multi_tasklets if t > best_cutoff]

fig1, ax1 = plt.subplots(
    figsize=(2 + len(tasklet_counts) * WIDTH_PER_TASKLET, FIG_HEIGHT)
)
ax1.boxplot(box_data, tick_labels=tasklet_counts, showfliers=False)

annotations1 = [("tab:green", "T = 1", medians[1], None, None)]
annotations1.append(
    (
        "tab:blue",
        f"T ∈ [2, {best_cutoff}]",
        *draw_regression(ax1, lo, medians, "tab:blue", f"T ∈ [2, {best_cutoff}]"),
    )
)
annotations1.append(
    (
        "tab:orange",
        f"T > {best_cutoff}",
        *draw_regression(ax1, hi, medians, "tab:orange", f"T > {best_cutoff}"),
    )
)

add_separators(ax1, best_cutoff)
add_annotations(ax1, annotations1)
ax1.legend()
ax1.set_xlabel("Tasklet count")
ax1.set_ylabel("Overhead per barrier (cycles)")
ax1.set_title(f"Barrier overhead — real measurements (best cutoff: T={best_cutoff})")
ax1.grid(axis="y", linestyle="--", alpha=0.5)
fig1.tight_layout()
out1 = csv_file.rsplit(".", 1)[0] + ".png"
fig1.savefig(out1, dpi=150)
print(f"Saved {out1}")

# ── plot 2: simulated results (scatter) ──────────────────────────────────────

fig2, ax2 = plt.subplots(
    figsize=(2 + len(tasklet_counts) * WIDTH_PER_TASKLET, FIG_HEIGHT)
)
positions_all = [tasklet_counts.index(t) + 1 for t in tasklet_counts]
ax2.plot(
    positions_all,
    sim_values[tasklet_counts].values,
    color="crimson",
    linewidth=1.5,
    marker="o",
    markersize=4,
)
ax2.set_xticks(positions_all)
ax2.set_xticklabels(tasklet_counts)
ax2.set_xlabel("Tasklet count")
ax2.set_ylabel("Overhead per barrier (cycles)")
ax2.set_title("Barrier overhead — simulated")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
fig2.tight_layout()
out2 = sim_csv_file.rsplit(".", 1)[0] + ".png"
fig2.savefig(out2, dpi=150)
print(f"Saved {out2}")

# ── plot 4: merged ───────────────────────────────────────────────────────────

fig4, ax_main = plt.subplots(
    figsize=(2 + len(tasklet_counts) * WIDTH_PER_TASKLET, FIG_HEIGHT)
)

# Box plots and regression lines
ax_main.boxplot(box_data, tick_labels=tasklet_counts, showfliers=False)
draw_regression(ax_main, lo, medians, "tab:blue", f"T ∈ [2, {best_cutoff}]")
draw_regression(ax_main, hi, medians, "tab:orange", f"T > {best_cutoff}")

# Sim line
sim_positions = [tasklet_counts.index(t) + 1 for t in tasklet_counts]
ax_main.plot(
    sim_positions,
    sim_values[tasklet_counts].values,
    color="crimson",
    linewidth=1.5,
    marker="o",
    markersize=3,
    label="simulated",
    zorder=3,
)

# Difference bars on the same axis
diff = pd.Series({t: medians[t] - sim_values[t] for t in tasklet_counts})
diff_vals = diff[tasklet_counts].values
diff_colors = ["steelblue" if v >= 0 else "tomato" for v in diff_vals]
ax_main.bar(
    sim_positions,
    diff_vals,
    color=diff_colors,
    alpha=0.4,
    width=0.6,
    label="real − sim",
    zorder=2,
)

# diff_hi_tasklets = [t for t in tasklet_counts if t > 12]
# diff_reg = draw_regression(ax_main, diff_hi_tasklets, diff, "purple", "diff fit (T > 12)")
# add_annotations(ax_main, [("purple", "diff fit (T > 12)", *diff_reg)])

add_separators(ax_main, best_cutoff)
ax_main.set_xlabel("Tasklet count")
ax_main.set_ylabel("Overhead per barrier (cycles)")
ax_main.set_title("Barrier overhead — real vs simulated")
ax_main.grid(axis="y", linestyle="--", alpha=0.5)
ax_main.legend(loc="upper left")

fig4.tight_layout()
out4 = "merged.png"
fig4.savefig(out4, dpi=150)
print(f"Saved {out4}")

# ── plot 3: difference (real median − sim) ───────────────────────────────────

fig3, ax3 = plt.subplots(
    figsize=(2 + len(tasklet_counts) * WIDTH_PER_TASKLET, FIG_HEIGHT)
)
ax3.bar(
    positions_all,
    diff[tasklet_counts].values,
    color=[
        "tab:green" if t == 1 else ("tab:blue" if t <= best_cutoff else "tab:orange")
        for t in tasklet_counts
    ],
    alpha=0.75,
)
ax3.axhline(0, color="black", linewidth=0.8)

# diff_reg3 = draw_regression(ax3, diff_hi_tasklets, diff, "purple", "diff fit (T > 12)")
# add_annotations(ax3, [("purple", "diff fit (T > 12)", *diff_reg3)])
# ax3.legend()

ax3.set_xticks(positions_all)
ax3.set_xticklabels(tasklet_counts)
ax3.set_xlabel("Tasklet count")
ax3.set_ylabel("Δ overhead (cycles)")
ax3.set_title("Real − simulated barrier overhead (cycles)")
ax3.grid(axis="y", linestyle="--", alpha=0.5)
fig3.tight_layout()
out3 = sim_csv_file.rsplit(".", 1)[0].replace("sim_", "") + "diff.png"
fig3.savefig(out3, dpi=150)
print(f"Saved {out3}")

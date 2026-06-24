#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FREQ_HZ = 350e6  # 350 MHz

def ns_to_cycles(ns):
    return ns * FREQ_HZ * 1e-9

csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
df = pd.read_csv(csv_file)

tasklet_counts = sorted(df["tasklets"].unique())
multi_tasklets = [t for t in tasklet_counts if t > 1]  # T=1 handled separately
data = [ns_to_cycles(df[df["tasklets"] == t]["overhead_ns_per_barrier"].values) for t in tasklet_counts]

medians = pd.Series(
    {t: np.median(ns_to_cycles(df[df["tasklets"] == t]["overhead_ns_per_barrier"])) for t in tasklet_counts}
)

def fit_segment(tasklets):
    positions = [tasklet_counts.index(t) + 1 for t in tasklets]
    slope, intercept, r, *_ = stats.linregress(positions, medians[tasklets].values)
    return slope, intercept, r**2

# Search over cutoffs in the T>1 range (both sides need >= 2 points)
print(f"{'cutoff':>8}  {'R²(lo)':>8}  {'R²(hi)':>8}  {'R²(weighted)':>13}")
best_cutoff, best_r2 = None, -1
for cutoff in multi_tasklets[1:-1]:
    lo = [t for t in multi_tasklets if t <= cutoff]
    hi = [t for t in multi_tasklets if t > cutoff]
    if len(lo) < 2 or len(hi) < 2:
        continue
    _, _, r2_lo = fit_segment(lo)
    _, _, r2_hi = fit_segment(hi)
    r2_w = (len(lo) * r2_lo + len(hi) * r2_hi) / len(multi_tasklets)
    marker = ""
    if r2_w > best_r2:
        best_r2, best_cutoff = r2_w, cutoff
        marker = " <"
    print(f"{cutoff:>8}  {r2_lo:>8.4f}  {r2_hi:>8.4f}  {r2_w:>13.4f}{marker}")

print(f"\nBest cutoff: {best_cutoff}  (weighted R²={best_r2:.4f})")

lo = [t for t in multi_tasklets if t <= best_cutoff]
hi = [t for t in multi_tasklets if t > best_cutoff]

fig, ax = plt.subplots(figsize=(14, 5))
ax.boxplot(data, tick_labels=tasklet_counts, showfliers=False)

def draw_regression(ax, tasklets, color, label):
    positions = [tasklet_counts.index(t) + 1 for t in tasklets]
    slope, intercept, r2 = fit_segment(tasklets)
    xs = np.linspace(positions[0], positions[-1], 200)
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.5, label=label)
    return slope, intercept, r2

# T=1 annotation (single point, no regression)
t1_cycles = medians[1]
annotations = [("tab:green", "T = 1", t1_cycles, None, None)]

annotations.append(("tab:blue",   f"T ∈ [2, {best_cutoff}]", *draw_regression(ax, lo, "tab:blue",   f"T ∈ [2, {best_cutoff}]")))
annotations.append(("tab:orange", f"T > {best_cutoff}",       *draw_regression(ax, hi, "tab:orange", f"T > {best_cutoff}")))

# Vertical separators
ax.axvline(tasklet_counts.index(1) + 1.5,           color="gray", linestyle=":", linewidth=1)
ax.axvline(tasklet_counts.index(best_cutoff) + 1.5, color="gray", linestyle=":", linewidth=1)

for i, (color, label, val_or_slope, intercept, r2) in enumerate(annotations):
    if intercept is None:
        text = f"{label}:  median = {val_or_slope:.3f} cycles"
    else:
        text = f"{label}:  slope={val_or_slope:.3f} cycles/tasklet,  intercept={intercept:.3f} cycles,  R²={r2:.4f}"
    ax.text(0.01, 0.97 - i * 0.08, text, transform=ax.transAxes,
            color=color, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))

ax.legend()
ax.set_xlabel("Tasklet count")
ax.set_ylabel("Overhead per barrier (cycles)")
ax.set_title(f"Barrier overhead per trial (best cutoff: T={best_cutoff})")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
out = csv_file.rsplit(".", 1)[0] + ".png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")

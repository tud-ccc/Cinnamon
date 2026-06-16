# BO Problem Synopsis — ${problem_name}

**Search space:** ${n_total} configs (${n_valid} valid) — **${n_seeds} seeds**, up to ${n_iters} iterations each  
**Oracle best:** ${oracle_best} — **Mean best found:** ${best_cost_mean} (gap: ${gap_pct_mean}%),  best seed: ${best_cost_best} (gap: ${gap_pct_best}%)  
**Objective scale:** ${scale}

Seed-level details: see each `seed_*/README_bo_synopsis.md`.

---

## 3. Search Quality

### Recall

![Recall](recall_pcts.png)

Fraction of the oracle's top-k% configurations that BO has found, as a function of
evaluations. Multiple thresholds (2%, 5%, 10%, 15%) show whether BO first narrows to the
right neighbourhood and then fills it in, or misses the best region entirely.
Lines are mean across seeds; shaded band is ±1σ.

A steep early rise followed by a plateau means BO quickly found the
right region but stopped exploring it — consider increasing `max-evals` or `kappa`.
A flat portion means the search is focused on exploration rather than exploitation.
If exploration is also low quality during those phases, then the surrogate is not providing enough guidance and the search is close to random.

Keep in mind those are percentages and the evaluation budget limits the ability of the search to explore the best regions exhaustively.

---

### Best cost found

![Best cost found](best_cost_found.png)

Running minimum of the observed cost over evaluations, compared to the oracle best
(red dashed line). The gap at the end of the curve is the optimality shortfall.
Lines are individual seeds; bold line is the mean; shaded band is ±1σ.

**How to read:** A rapid drop followed by a flat plateau means BO found a good solution
early but stopped improving — either the best region is already saturated or BO is
exploring irrelevant regions. A curve that never flattens and ends far from the oracle
line means the budget was exhausted before convergence.

### First-hit curve

![First-hit curve](first_hit_curve.png)

For each quality threshold T (% above oracle best), the number of evaluations needed
for a seed to first observe a configuration within that quality level.
Blue = mean across seeds; red dashed = worst seed.
The gray dotted line marks the total evaluation budget.

**How to read:** A steep drop near 0% means BO reliably finds near-optimal solutions
quickly. A flat region at the budget line for small T means no seed reached that quality
within the budget — the threshold is too tight for the current settings.
The gap between mean and worst-seed curves shows consistency: a large gap means at
least one seed consistently underperforms.

---

## 4. Timing

### Wall-clock time per evaluation

![Aggregate timings](agg_timings.png)

Cumulative wall-clock time as a function of evaluation number, aggregated across seeds.
Bold black line is the mean; shaded band is the IQR (25th–75th percentile); thin coloured
lines are individual seeds.

**How to read:** A linear curve means each evaluation takes roughly the same time
(expected for fixed-cost simulators). A curve that steepens over time means later
evaluations are more expensive — common when the surrogate fitting cost grows with the
number of observations.

---

## 5. Surrogate Learning Curves — Aggregate

### RMSE

| Validation | Training |
|:---:|:---:|
| ![Validation RMSE](agg_validation_rmse.png) | ![Training RMSE](agg_training_rmse.png) |

Mean surrogate RMSE across all seeds, with IQR band (25th–75th percentile). Thin coloured
lines are individual seeds. See per-seed plots for the tasklet breakdown.

MAPE (relative error): [validation MAPE](agg_validation_mape.png) · [training MAPE](agg_training_mape.png)

Validation RMSE that plateaus or rises while training RMSE keeps falling is overfitting.
Both staying high means underfitting. A tight IQR band means the surrogate quality is
consistent across seeds; a wide band means some seeds converge much better than others.

---

## 5. Diagnosis — Surrogate Calibration

> These plots aggregate over all seeds. Per-seed heatmaps are in the respective seed folders.

### σ vs. distance from observations

![Sigma vs distance](pool_sigma_vs_dist.png)

Median surrogate uncertainty (σ) as a function of discrete grid-step distance to the
nearest successfully-evaluated configuration, with IQR band (single seed) or mean±1σ band
(multiple seeds). Per-seed thin lines shown when multiple seeds are present.

σ should rise monotonically with distance. A flat or non-monotone curve
means the ensemble's uncertainty is not distance-aware — the acquisition function will not
reliably direct search towards unexplored regions, and UCB will behave like greedy
exploitation. This is the earliest diagnostic that something is wrong with ensemble
diversity.

---

### σ scatter (uncertainty vs. distance)

![Sigma scatter](pool_sigma_scatter.png)

Each visited config plotted as (distance to nearest observation, σ), coloured by predicted
cost (μ). Reveals whether the ensemble assigns more uncertainty to rare or extreme-cost
points. This is the scatter version of the line plot above.

Points in the bottom right are points where the ensemble is overconfident far from data.
Points in the top left are points where the ensemble is not confident enough even when near known data.

# BO Seed Synopsis — ${problem_name} / ${seed_name}

**Search space:** ${n_total} configs (${n_valid} valid) — **${n_obs} evaluated** (${n_failed} failed) over ${n_iters} iterations  
**Best cost found:** ${best_cost} (oracle best: ${oracle_best}, gap: ${gap_pct}%)  
**Objective scale:** ${scale}

---

## 1. Final Surrogate Inspection

These four plots share the same grid layout (x = ${ax_x}, y = ${ax_y}, facets = ${ax_f}).
`pool_cost` and `pool_mu` intentionally use the same colour scale so deviations are visible
by direct comparison.

The other dimensions of the search space are aggregated with an aggregation method that depends on the metric:
- cost, mu, acq: min
- sigma: mean

**Colour legend (all heatmaps):** 
- white = unsampled
- light gray = constraint-violated (invalid configuration)
- dark gray = evaluated but failed
- colour scale = metric value.

### Observed cost

![Observed cost](pool_cost.png)

Sampled configurations over BO coloured by true cost.
Valid, but non-sampled configurations are white.
Samples that we tried to take but that failed (and were valid) are dark gray.
Those points do not count towards the evaluation budget, even though they might cost evaluation time.

A contiguous colored region means BO focused its interest on this region.
It better be a region with low cost value. If it isn't, then acquisition is probably miscalibrated. 

If the cheapest cell is isolated and surrounded by white, BO may have found it by luck rather than
guided search — check recall_pcts in the [problem synopsis](../README_bo_synopsis.md).

---

### Surrogate minimum mean prediction (μ)

![Surrogate mu](pool_mu.png)

The ensemble's posterior mean over the entire valid grid after the final BO iteration.
Displayed is the _best_ prediction for the invisible (reduced over) dimensions.

Agreement with `pool_cost` in sampled regions indicates the surrogate
fit the training data properly (although also see RMSE plots below).
The overall color gradient should roughly match the oracle's own color gradient, indicating the surrogate modeled the best region and its neighborhood properly.

> **TODO:** Add a third panel showing the oracle cost on the same colour scale for
> direct ground-truth comparison.

---

### Surrogate uncertainty (σ)

![Surrogate sigma](pool_sigma.png)

Ensemble standard deviation across the full grid — a proxy for epistemic uncertainty.
High σ in unvisited regions is expected and desirable; high σ in visited regions signals
ensemble disagreement on already-observed data, which indicates poor fit or insufficient
training epochs.

A σ that is uniformly high across the board means
the ensemble has not converged; uniformly low σ everywhere means it has collapsed to
overconfident predictions and will exploit rather than explore.

---

### Acquisition map (α)

![Acquisition](pool_acq.png)

UCB acquisition scores across valid unvisited cells at the final iteration. The
lightest-coloured cell is the one BO would select next if the run were to continue. The acquisition is a function of mu and sigma.

Keep in mind acquisition does not reflect the fitness of a point, as it incorporates sigma.
Good regions that have already been oversampled expectedly show low acquisition, especially here at the end of the optimization process.

A concentrated bright spot means BO has a clear next target (confident
exploitation or well-localised uncertainty). A diffuse bright region means the surrogate
is still highly uncertain across many candidates and would explore widely — expected early
in the run, a concern late in it.

---

## 2. Surrogate Learning Curves

### RMSE

| Validation | Training |
|:---:|:---:|
| ![Validation RMSE](pool_validation_rmse.png) | ![Training RMSE](pool_training_rmse.png) |

Error between the surrogate's μ predictions and true (scaled) costs, measured on the
held-out validation set and on the training set respectively.

Validation RMSE that plateaus or rises while training RMSE keeps falling
is a sign of overfitting — reduce epochs or increase ensemble size. Both staying high
means the surrogate is underfitting — the cost surface may be rougher than the MLP can
represent. A gap of less than ~2× between training and validation RMSE is healthy.
Per-tasklet lines highlight whether the surrogate learns certain slices faster than others.

MAPE (relative error): [validation MAPE](pool_validation_mape.png) · [training MAPE](pool_training_mape.png)

(MAPE is not directly comparable to the mu values and is therefore less interesting, while usually showing the exact same trend)

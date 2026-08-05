# Landscape Analysis — ${csv_name}

Exhaustive search over **${n_obs} observed configs**
(${n_valid} valid / ${n_total} total in Cartesian product).

**Dimensions (${n_dims}):** ${dim_list}
**Cost range:** [${cost_min}, ${cost_max}] — log₁₀: [${logcost_min}, ${logcost_max}]

---

## 1. Cost Distribution

![Cost distribution](landscape_cost_dist.png)

Histogram of log₁₀(cost) across all observed configs, with percentile markers.

**How to read:** A narrow distribution means the space is uniformly hard or easy; a wide
or bimodal one suggests some dimensions sharply separate good from bad configs. The range
from p10 to p90 spans ${logcost_p10_p90_gap} log₁₀ units — a factor of
${cost_p10_p90_ratio}× in raw cost.

---

## 2. Per-Dimension Marginals

![Per-dimension marginals](landscape_marginals.png)

Mean log₁₀(cost) ± IQR as a function of each dimension value, with all other dimensions
averaged out. The shaded band is the 25th–75th percentile across configs sharing that value.

**How to read:** A strong trend (mean moves a lot across the axis range) signals a large
*main effect* — this dimension matters a lot on average. A wide IQR band signals strong
*interaction* with other dimensions — the dimension's effect is context-dependent. Dimensions
with a strong trend and a narrow band are the easiest for a surrogate to learn from few
samples.

---

## 3. Per-Dimension Roughness

![Per-dimension roughness](landscape_roughness.png)

Distribution of |Δlog₁₀(cost)| between axis-aligned neighbours (one sub-index step),
sorted by mean roughness. Each violin covers all pairs of configs identical except in
that one dimension.

${roughness_table}

**How to read:** A violin concentrated near zero means the cost changes smoothly step-to-step —
BO can exploit gradient information over multiple hops. A heavy-tailed or wide violin means
cliff-like transitions; the surrogate needs many nearby observations to model the dimension.
Rugged dimensions benefit most from denser sampling (smaller step sizes or more LHS points).

---

## 4. Variogram

![Variogram](landscape_variogram.png)

γ(h) = ½ · E[(Δlog₁₀f)²] at lag h measures how much cost changes when moving h sub-index
steps along each dimension, holding all others fixed.

**How to read:**

- **Flat after lag 1** → no long-range correlation; each step is essentially independent.
  Keep `neighbor-depth = 1` in BO.
- **Rising through lag 3–4** → long correlation length; the surface is smoother at a
  coarser scale. Increase `neighbor-depth` so BO candidates span that range.
- **Levelling off (sill)** → the correlation range is approximately where the curve
  flattens. Beyond that lag, configs are nearly uncorrelated and neighbours of neighbours
  add little signal.

---

## 5. 2-D Marginal Heatmaps

![2-D marginal heatmaps](landscape_2d_marginals.png)

Mean log₁₀(cost) for every pair of dimensions, averaged over all other dims.
All panels share the same colour scale for direct comparison.

**How to read:** Diagonal bands or a ridge along the anti-diagonal indicate that the cost
depends on the *product* di × dj rather than on di and dj individually — the classic
compensating-dimension signature (see section 6). A flat heatmap means the pair interacts
weakly. Checkerboard patterns indicate strong but non-compensating interactions that the
surrogate needs to learn explicitly.

---

## 6. Compensating Dimension Analysis

![Compensating dimension analysis](landscape_compensation.png)

**Motivation:** If dimensions di and dj compensate — e.g., halving R and doubling D gives
a similar cost — then cost is smoother along iso-product curves {x : di · dj = c} than
along grid edges. This section tests that hypothesis with two formal metrics.

**Dirichlet energy ratio E_prod / E_axis:** mean squared log₁₀(cost) difference on the
iso-product graph divided by the same on the axis-aligned graph. Ratio < 1 means smoother
along iso-product. Strong pairs appear below the y = x diagonal in the scatter and show
short blue bars.

**R² (product):** fraction of cost variance explained by knowing di × dj alone (within
fixed other dims). R² near 1 means the product is almost a sufficient statistic for cost.

${compensation_table}

**Implication for BO:** Pairs with ratio < 0.5 *and* R² > 0.7 are strong candidates for
a *derived feature*: feed log(di × dj) to the surrogate alongside the raw dimensions.
You can also extend `neighborIndices` to include iso-product moves (R/n, D·n) so the BO
search can follow these smoother gradients directly.

---

## 7. Top-${top_frac_pct} Concentration

![Top-k concentration](landscape_topk.png)

Marginal distribution of each dimension for all ${n_obs} configs (blue) vs the best
${n_top} configs by cost (${top_frac_pct}, red).

**How to read:** A large red-vs-blue shift in a dimension means good configs cluster at
specific values — the dimension is both important and exploitable. No shift means the
dimension either doesn't matter (corroborated by a flat marginal in section 2) or good
values are spread evenly. Use shifted dimensions to bias LHS initialisation in BO:
oversample the red-shifted region for faster warm-up.

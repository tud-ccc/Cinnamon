# Search strategy overhaul: diagnosis, refactor, comparison campaign

Status: in progress (2026-08-25). Context: the BANANAS-style BO search in
`lib/Dialect/Cinm/AcceleratorInference/BananasSearch.cpp` shows strong
seed-dependence (e1 campaign), which undermines the wholeprogram profiles'
convexity and the greedy allocator's optimality argument. The search is not a
paper contribution, so we are free to replace it wholesale.

## Diagnosis (from the e1 dumps, `experiments/evaluation/data/*/search/dump`)

All numbers are simulator-space, i.e. independent of hardware noise; the
sim-cost CV across seeds tracks the hardware CV almost 1:1, so the variance is
a search-convergence failure, not cost-model infidelity. Analysis scripts can
be rerun offline against `pool.csv`/`rounds.csv` alone.

1. **Two populations of benchmarks.** va/geva/red have 432–621 feasible
   configs; 256 evals covers ~half the space and every seed converges (CV 0%).
   gemv/mtv (28k–44k) and mmtv/ttv (0.5M–1.6M) are real search problems and
   there seeds disagree wildly (sim CV 9–30%; ttv_512MB: 32 seeds → 32
   distinct best configs, median regret vs pooled best 82%).

2. **The candidate set is 100% local neighbours on every hard space.**
   `nextCandidateIndices` fills neighbours first, then `fillRandom` early
   returns because the neighbour set (1500–3700 points on 9-dim spaces)
   already exceeds `nCandidates` = 500. From `rounds.csv`: `rand=0` on every
   round of every hard benchmark. The algorithm degenerates to a 4-wide
   greedy hill climb from the 64 LHS points; the winning basin is decided by
   init RNG. This is the primary cause of seed-dependence.

3. **The surrogate memorises rather than generalises.** Train-fit Spearman ρ
   ≈ 0.99, out-of-sample ρ (selection mu vs later-observed cost) ≈ 0.27–0.47
   on hard spaces. All 7 ensemble members see all data and differ only in
   init/minibatch order, so σ collapses and Thompson sampling loses its
   exploration signal (median σ/|μ| 0.13–0.27).

4. **The economics are wrong for a surrogate this heavy.** A cycle-accurate
   sim eval costs ~0.4 CPU-s; the 256-eval budget is ~2 CPU-min of objective
   time inside a ~2 CPU-h seed. At equal budget the BO **loses to uniform
   random sampling** on most hard benchmarks (median-of-32 BO best vs
   best-of-300 uniform: mmtv_256MB 1.69×, ttv_512MB 1.51×, mtv_64MB 1.44×).
   BANANAS's design point (evals cost GPU-hours) does not apply here.

Independent of the search, wholeprogram profiles can be protected directly:
- **Warm-start along the menu axis** (seed search at size n+1 with incumbents
  from n, n−1) → near-monotone profiles by construction.
- **Profile repair**: evaluate every menu point's incumbent at every other
  menu point (projecting when constraints require), take pointwise min →
  the profile becomes a lower envelope; RNG cliffs cannot survive it.

## Step 1: pluggable `SearchStrategy` (this refactor)

Make `AcceleratorInference` generic in the search algorithm so variants can be
compared under one harness. Carve `BananasSearch.cpp` along the ask/tell line:

- `SearchStrategy` interface (new `SearchStrategy.h` in the lib dir):
  `step(rng, accept, round, nObsAtRound, batchSize, workers) → accepted`,
  plus hooks for pool.csv annotation (`hasModel`/`predict`) and
  strategy-specific diagnostics dumps (`dumpDiagnostics`).
- `BananasStrategy` (in `BananasSearch.cpp`) takes the ensemble, the
  round/batch diagnostics, the validation-snapshot plumbing, and the
  body of `nextCandidateIndices`. Bit-identical behaviour: same RNG draw
  order, same dumps.
- `CandidatePool` stays the shared substrate: space access, visited set,
  Xo/yo observation matrices, `sampleInitialSet` (LHS/uniform init is
  strategy-independent and identical across arms → paired comparisons),
  `fillNeighbors`/`fillRandom` become public shared primitives (a GA's
  mutation is a draw from `neighborIndices`).
- Driver (`runSeedBO`) constructs the strategy from
  `InferenceOptions::searchStrategy` (pass option `search-strategy`,
  default `bananas`).
- The all-rejected fallback (walk the ranking for one acceptable point) is
  policy and lives inside the strategy.

## Step 2: strategy arms (follow-ups, each its own small PR)

1. `bananas` — control, exactly today's behaviour.
2. `bananas` + fillRandom fix — candidate set gets its random half back
   (target semantics bug: `fillRandom(candSet, nCandidates)` counts the
   neighbours already present against the target). Possibly a separate
   option so both variants stay runnable.
3. `random` — uniform draws without replacement; the floor, and at 5k evals
   the "what does compute alone buy" line.
4. `descent` — random restarts + neighbourhood descent on top of
   `neighborIndices`; budget split across restarts.
5. `ga` — population GA over the discrete dims (tournament select, uniform
   crossover on dims, mutation = neighbour step). The ATiM-class defence arm.

## Step 3: offline regret harness + campaign

A doit task (or standalone script under `experiments/evaluation/`) that runs
arms × benchmarks × seeds in simulator space and reports regret against a
pooled reference optimum:

- Reference optimum per benchmark: one overnight burn of ~50k random sim
  evals (~6 CPU-h each, embarrassingly parallel), unioned with every arm's
  observations, cached forever.
- Paired seeds: same seed → same LHS init across arms; arms differ only
  after eval nInit.
- Output: one tidy CSV `campaign.csv` with columns
  `arm, benchmark, seed, eval_idx, cost, best_so_far, wall_ms, cpu_ms`
  (best_so_far materialised per eval so anytime curves need no recompute).

## Plots (deliverables of the campaign)

All from `campaign.csv`; matplotlib scripts live next to the existing
`plot_*.py` files in `experiments/evaluation/`.

1. **Anytime regret curves** (`plot_search_anytime.py`): x = evaluations
   (and a twin in CPU-seconds — the axis that indicts/acquits the surrogate),
   y = median regret vs reference optimum, one line per arm, band =
   IQR across seeds. One panel per benchmark, log-y. The headline figure.
2. **Final-regret distributions**: per benchmark, one box/strip per arm at
   budget exhaustion. Answers "which arm, at the campaign budget" and makes
   seed-variance (the actual complaint) directly visible as box height.
3. **Reliability curve**: for each arm, fraction of seeds within ε of the
   reference optimum as a function of ε (a CDF of relative regret, pooled
   and per benchmark class). This is the figure that speaks to the
   wholeprogram requirement — "P(a single search is within 5%)".
4. **Best-of-k restarts**: median regret of best-of-k independent seeds vs k,
   per arm — tells us whether restarts are a substitute for a better
   algorithm and what k the wholeprogram searches would need.
5. **Profile convexity check** (wholeprogram tie-in, later): per device
   class, cost-vs-devices profile under each arm with the per-seed spread,
   before vs after profile repair; count of convexity violations as the
   summary stat.

## Progress log

- 2026-08-25: diagnosis done (this doc). Step 1 done: `SearchStrategy`
  interface (`lib/.../AcceleratorInference/SearchStrategy.h`), BANANAS moved
  behind it as `BananasStrategy` (BananasSearch.cpp), pass option
  `search-strategy` (default `bananas`). Smoke-tested on va_4MB,
  cycle-accurate, 2 seeds × 48 evals: all dumps well-formed, best cost in
  line with the campaign. Next: arm 2 (fillRandom fix) + `random` strategy,
  then the regret harness.
- 2026-08-25 (later): arms 2 and 3 done. `n-random-candidates=K` draws K
  random candidates on top of the neighbour set (0 = legacy top-up
  behaviour); verified rand=K on every round of mtv_4MB where legacy had
  rand=0. `search-strategy=random` evaluates uniform draws with the whole
  budget; no model columns in pool.csv, no rounds.csv. Next: descent + GA
  arms, then the regret harness (`campaign.csv` + plots).
- 2026-08-25 (later): arms 4 and 5 done. `search-strategy=descent` is
  batched steepest descent over grid neighbours with random restarts (first
  climb starts from the best init observation); `search-strategy=ga` is a
  steady-state GA (binary tournament, per-parameter crossover so
  permutations never mix dimensions, neighbour-step mutation, random
  immigrants on convergence, population = best `n-init` observations).
  Smoke-tested on mtv_4MB: descent improves past init; GA's proposals
  (median 1.49 ms) concentrate far below uniform draws (median ~4.8 ms).
  All five arms now selectable; next: the regret harness
  (`campaign.csv` + plots), then the campaign itself.
- 2026-08-25 (later): regret harness done. `doit campaign` runs the four
  non-control arms (the `bananas` control is task_search's existing dump);
  `doit assemble_campaign` builds results/campaign.csv + campaign_ref.csv
  (assemble_campaign.py); `doit plot_campaign` draws the five figures
  (plot_search_campaign.py: anytime in evals and CPU-s, final-regret
  boxes, reliability CDF, best-of-k; CVD-validated fixed arm colors).
  Verified end-to-end on the existing bananas dumps: reliability shows
  P(run within 5%) ≈ 1.0 / 0.08 / 0.05 for small / medium / large spaces
  -- the quantified version of the complaint. Next: run `doit campaign`,
  then read the figures and pick the successor strategy.
- 2026-08-25 (later): campaign scoped to 5 representative functions
  (dodo.py `CAMPAIGN_FNS`: red_64MB the one small space where search does
  work, gemv_4MB the low-variance medium, mtv_256MB the high-variance
  medium, mmtv_4MB a ~500k space with cheap evals, ttv_512MB the worst
  RNG dependence). 4 arms × 5 fns × 32 seeds ≈ an afternoon, not
  overnight; set `CAMPAIGN_FNS = ()` for the full 26-function campaign
  later. The plots restrict pooled statistics to functions every arm ran,
  so scoped and full runs never mix into an unfair comparison.
- 2026-08-25 (later): multiseed engine made work-conserving. EvaluatorPool
  leases now block (they previously popped an empty freelist -- UB held off
  only by the per-seed split), and every seed may keep boBatchSize
  evaluations in flight against one shared pool, so a seed stalled on a
  slow simulation donates its capacity instead of idling its dedicated
  workers. Measured on mtv_256MB, 32 seeds x 64 workers: utilization 43%
  -> 68%, stage wall 363s -> 197s. `doit -n` remains unnecessary.
- 2026-08-25 (later): while validating that change, found that the
  2026-08-20 lowering commits (bccdf4cc and neighbours) moved the
  simulated cost landscape: on mtv_256MB seed_67, 121/163 configurations
  revisited by an identical-RNG rerun cost within 1% of the e1 dump, but
  the gemv.order=(2,1) region costs ~2x more (e1's best config: 3.26 ->
  5.29 ms). The search engine itself is unchanged (63/64 identical init
  draws; deterministic reruns). Consequence: **pre-Aug-20 dumps
  (data/*/search, data/*/sample, e1's sim-space numbers) must never be
  compared or pooled with newly generated sim costs.** The campaign's
  bananas control is therefore a re-run arm (campaign_bananas), and the
  reference optimum pools only campaign dumps. Any results/campaign.csv
  assembled before this note mixes landscapes -- re-assemble after the
  campaign runs.
- 2026-08-25 (later): scoped campaign ran (5 arms x 5 fns x 32 seeds).
  Verdict, as P(seed within 5% of the pooled reference):

  | arm          | all  | gemv_4MB | mtv_256MB | mmtv_4MB | ttv_512MB |
  |--------------|------|----------|-----------|----------|-----------|
  | bananas      | 0.21 | 0.03     | 0.00      | 0.00     | 0.00      |
  | bananas_rand | 0.64 | 0.91     | 1.00      | 0.09     | 0.19      |
  | random       | 0.26 | 0.16     | 0.06      | 0.03     | 0.03      |
  | descent      | 0.21 | 0.03     | 0.03      | 0.00     | 0.00      |
  | ga           | 0.23 | 0.09     | 0.00      | 0.03     | 0.00      |

  The candidate-set bug was the whole BANANAS story: plain bananas is
  *worse than random* on every hard space, and descent tracks it (broken
  BANANAS was a hill climb). GA at this budget sits between descent and
  random; neither control arm warrants further investment. Remaining gap:
  the 9-dim large spaces.
- 2026-08-25 (later): K-sweep on the gap. ttv_512MB (1.6M configs),
  32 seeds: K=256/1024/4096 -> median regret 45.8/32.9/0.9%, P(<=5%)
  0.19/0.44/0.66. mmtv_4MB K=256/1024 -> med 26.3/18.8. No regression on
  mtv_256MB at K=4096 (P stays 1.00); screening cost negligible
  (predict 64 ms vs fit 299 ms per round). **Decision: n-random-candidates
  defaults to 4096** (pass option + InferenceOptions), so every consumer
  -- task_search, wholeprogram profile searches -- gets the fixed search;
  K=0 restores the pre-campaign behaviour. Campaign arms pin K explicitly
  (bananas=0, bananas_rand=256, new bananas_rand4k=4096); the
  bananas_rand4k arm has no dumps yet -- run `doit campaign` to add it
  (~25 min scoped). Still open: p90 on ttv is 46% (a minority of seeds
  stall), so the wholeprogram profile still wants repair/warm-start
  (§"fix the profile convexity directly"); re-running task_search and the
  wholeprogram stack under the new default invalidates their old dumps
  (landscape rule above applies to search-version drift too).

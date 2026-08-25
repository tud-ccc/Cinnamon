# Search strategy overhaul: diagnosis, refactor, comparison campaign

Status: in progress (2026-08-25). Context: the BANANAS-style BO search in
`lib/Dialect/Cinm/AcceleratorInference/BananasSearch.cpp` shows strong
seed-dependence (e1 campaign), which cuts cliffs into the wholeprogram cost
profiles and so undermines the greedy allocator's optimality argument. The
search is not a paper contribution, so we are free to replace it wholesale.

Terminology, because conflating these two caused a wrong claim early on:
a profile is **monotone** when cost never rises with more devices, and
**convex** when the marginal gain (ms saved per added device) never rises.
Convexity implies monotonicity, not the reverse. Lower-envelope repair
buys monotonicity outright; convexity it only improves.

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
   before vs after profile repair; counts of *both* monotonicity and
   convexity violations as the summary stats -- they are different
   properties and repair only guarantees the first (plot_profiles.py's
   marginal panel already reports the convex/total tally).

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
- 2026-08-25 (later): dynamic stopping evaluated and REJECTED for now.
  Retrospective replay of "stop after X evals without >1% improvement"
  over the bananas_rand4k campaign seeds: patience 32/48 loses 52/21pp
  median regret; patience 64 saves 46% of evals but costs mmtv_4MB
  +49pp median; only patience >= 96-128 is quality-safe and saves just
  12-24% of 0.4s evaluations. Improvement is bursty (plateau, then a
  Thompson hit), so patience must approach half the budget to be safe.
  Principled BO termination (Makarova et al. 2022 regret bounds; EI
  thresholds) needs calibrated posteriors, which the ensemble's sigma is
  not. Revisit only if profiling walltime becomes a bottleneck; prefer
  menu-axis warm-starting then.
- 2026-08-25 (later): profile repair implemented (profile-repair pass
  option, default on). profileComputeBlock now applies a lower-envelope
  running argmin over the menu: a point whose pinned search measured
  worse than a smaller point's incumbent takes that incumbent (config,
  cost, residency) -- allocating R devices can always run a smaller
  point's configuration and idle the rest, so the repaired profile is
  achievable by construction and non-increasing. No projection and no
  extra evaluations needed (the plan's original cross-evaluation idea is
  subsumed). profiles.csv gains raw_cost_ms + repaired_from;
  plot_profiles.py draws the raw curve dashed behind the envelope.
  Motivation confirmed on the existing llama dump: 12/12 classes
  non-monotone (2-7 upward steps each); the small reduction classes rise
  from the first menu step, which is physics (scatter overhead), not
  noise -- repair flattens them, which is exactly what the allocator
  should see. Note the *pinned-group replay* consequence: a repaired
  point stamps the smaller resource's config, so a group allocated R may
  run dpus=R'<R with the rest idle.
- 2026-08-25 (later): repair verified on a fresh llama solve under the
  full new stack (work-conserving pool + K=4096 search + repair): 52/96
  points repaired, every class's profile non-increasing.
- 2026-08-25 (later): **correction -- repair does not deliver convexity.**
  Measured on that solve: monotone 0/12 -> 12/12 classes, but convex only
  0/12 -> 8/12, with 8 accelerating steps left in classes 3, 8, 10, 12.
  Two mechanisms, and they want different answers:
  * 2 of the 8 are *induced by repair* (class 8 @192, class 12 @1216): a
    repaired plateau followed by a real improvement is by definition an
    accelerating return. Repair trades a monotonicity violation for a
    convexity one.
  * 6 are genuine cost-model structure (threshold effects; class 10 @256
    jumps ~10x in marginal gain). Not search noise: with
    profile-seeds=2 the seed spread is 0.0% on classes 3/8/10 (only
    class 12 has real spread, up to 21.8%).

  Why this is nonetheless mostly benign: the latency greedy's grow move
  (GraphAllocation.cpp, `for (pi = group.point + 1; ...)`) enumerates
  *every* larger menu point and scores gain/spend, which is exactly the
  slope set of the profile's **lower convex hull** -- so the allocator
  already reasons over the hull and cannot stall on a plateau. The paper
  claim to make is therefore "profiles are monotone by construction and
  the allocator solves over their convex hull", not "profiles are
  convex". TODO below turns that from an argument into a measurement.
- 2026-08-25 (later): offloading criterion for transfer-bound classes.
  There is NO host cost model, so profitability cannot be a comparison
  (framed as future work); candidate a-priori rule: offload only ops
  with >= 1 static/amortizable operand. Tested against the llama
  profiles -- the correlation is strong but both directions have
  counterexamples:

  | classes | static op? | profile | verdict |
  |---|---|---|---|
  | 0, 4-7 (reductions/elementwise) | no | min at first point | bad candidates (criterion agrees) |
  | 2, 9, 10, 12 (projections/FFN) | yes | min deep in menu | good candidates (agrees) |
  | 3, 8 (attention matmuls, x96) | **no** | min at 192/256 | **good** despite no static operand |
  | 1 (rmsnorm-with-weight) | **yes** | min at first point | **bad** despite static operand |

  The attention matmuls are the expensive counterexample: all-dynamic
  operands but compute-heavy, so the static-operand rule would host a
  large share of the work. The measured criterion implemented instead
  (`host-transfer-bound-share`, default 0 = surface only): a class stays
  on the host when its best profile point is the *smallest* menu value
  AND that point's per-inference transfer share (new transfer_share
  column; amortized weight scatters excluded; e1 evidence: va/geva/red
  are 70-97% transfer, weight-stationary gemv/mtv are kernel-bound)
  exceeds the threshold. The conjunction keeps compute-bound classes
  that merely scale poorly. The static-operand rule remains the clean
  a-priori story for the paper, with the profiles as its (mostly
  supporting) evidence and attention as the honest exception.

## Open TODOs

- [ ] **Measure the greedy's optimality gap on llama** (empirical answer to
  the residual non-convexity). The allocation is small enough to solve
  exactly: 12 classes x <=16 menu points, budget 2560 DPUs at 64
  granularity = 40 units. Brute-force / DP the throughput and latency
  objectives over the dumped profiles.csv and compare with what
  allocateGraph* chose. If the gap is 0 the convexity question is closed
  empirically and the paper can say so; if not, the fix is per-class hull
  pre-processing rather than more search budget. Note the likely outcome
  (predicted 2026-08-25): the gap is probably ~0 and, more importantly,
  probably *uninformative*, because class 12 dominates the allocation --
  see below. Worth doing anyway since it is cheap and it retires an
  objection.

- [ ] **Class 12 dominates the llama allocation; part of that is an
  artifact.** It takes 1344 of 2560 DPUs (52.5% of the device) while
  contributing 6.2% of total work (23.6 ms of 383.4 ms), because it is
  the one class with multiplicity 1: every other class has 12-96 members
  that can be split across sets, so the only lever on this one is more
  devices. Its profile scales 194 ms @64 -> 17.4 ms @2048, so the
  allocator is right to feed it.
  The artifact: at 2048 DPUs its residency is 3648 B static vs 55860 B
  dynamic MRAM (94% dynamic), and ~10.6 ms of its 17.4 ms is charged as
  per-inference transfer against only 0.96 ms of amortized weight
  scatter. For the final vocabulary projection the weight matrix is the
  same on every inference and should be static. measureResidency's own
  caveat is the likely cause -- "an operand that is not directly a block
  argument (a linalg.fill accumulator, a fused intermediate) is charged
  as dynamic" -- so a reshape/cast between the weight and the op would
  hide it. Actions: (a) confirm by inspecting the class-12 block's
  operands in llama2_110M_with_compute.mlir; (b) if confirmed, teach
  staticness detection to see through the intervening ops (or pre-pad /
  pre-materialise the weight so it *is* a block argument); (c) re-profile
  and re-allocate -- correcting it should cut class 12's cost and free a
  large share of the device for the classes that actually carry the work.
  This is likely a bigger end-to-end win than anything left in the search.

  **Update (2026-08-25): confirmed and fixed.** The hider is the lowering
  of `tensor.pad` around the 32000->34048 vocab padding:
  `insert_slice(%arg15 into <constant fill produced inside a cinm.compute>)`.
  `%arg15` carries `cinm.static`, but the old isStaticValue could not
  follow it -- two gaps: insert_slice has *two* operands (the walk only
  followed single-operand views) and the destination is a compute-block
  *result* (the walk crossed blocks only in the argument direction).
  Fixed by the general rule "a pure op with static operands (and static
  region captures) yields a static value", which subsumes both plus the
  single-operand fast paths. Measured on a re-profile: at 64 DPUs the
  class's residency moves 1.6 MB/DPU from dynamic to static
  (dyn 1688544 -> 103572 B, static 3072 -> 1635648 B; x64 DPUs = the
  104 MB weight), and its cost falls 194->163 ms @64, 116->73 @128,
  68->40 @256 -- 16-41%, *understated* because that probe ran 32 evals
  against the baseline's 256.

  Two consequences to remember:
  - Operand staticness is part of the class-identity key
    (GraphInference.cpp:116), so this re-partitions the graph: llama went
    12 -> 15 classes. **Class indices are not comparable across runs
    either side of this change** -- match by source location. An earlier
    index-based comparison here produced a bogus 700x and was withdrawn.
  - The remaining cost at small device counts is kernel, not transfer, so
    this does not by itself stop class 12 dominating; it should shrink
    its share. Needs a full-budget re-profile + re-allocation to quantify.

- [ ] **Re-profile llama at full budget under the staticness fix** and
  re-run the allocation, then re-check: class 12's DPU share, whether the
  transfer-bound gate still selects the same six classes (transfer_share
  moves when weights stop being charged per-inference), and the
  monotone/convex tallies.

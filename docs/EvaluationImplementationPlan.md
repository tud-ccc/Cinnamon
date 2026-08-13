# Evaluation implementation plan (paper §8)

Status: **draft for review** (2026-08-12). Maps the evaluation section of the
CNM-search paper (`2026-cnmsearch-paper/contents/08_evaluation.tex`, header
comment = the experiment spec) onto concrete data-collection tasks in this
repository. ATiM and PrIM measurements are collected offline and enter this
pipeline only as interchange CSVs (§6.1) and manual transcriptions (§7).

Reading order: §1 what data each experiment needs; §2 the shared building
blocks; §3 compiler/runtime gaps; §4 harness gaps; §5 the dodo.py design;
§6 plots and tables; §7 the manual ATiM-transcription workflow; §8 phases;
§9 open questions.

---

## 1. Experiment → measurement inventory

Benchmarks below: the PrIM-derived set `prim_{va,red,mtv,ttv,mmtv,gemv,geva}`
(single-operator, from `experiments/prim_*.mlir`) for E1/RQ1/RQ2/RQ3/A1/A2;
the multi-operator set `2mm_{seq,par}` and `3mm_{seq,par,parseq}` — mixed
exists only for 3MM; `3mm_seq.mlir`/`3mm_par.mlir` are to be added (§4.5) —
plus one transformer block, for RQ4.

| Exp | What is measured, per benchmark | Arms run by *this* pipeline | Offline inputs |
|-----|--------------------------------|------------------------------|----------------|
| E1  | (a′) ATiM's schedule transcribed, under our codegen; (b) best-of-{uniform sample ∪ top-k-by-model}; (c) our search's pick; schedule invariants of (a′) | `eval-solution` runs of transcriptions; sample/top-k/search stacks | (a) ATiM best + its trace file; component subtotals |
| RQ1 | best measured config per system; CINM 1.0 (D,T) sweep incl. rule-decision transcription + sweep coverage | search stack (= E1's (c)); cinm1 sweep; rule-decision `eval-solution` | PrIM per benchmark; ATiM per benchmark; CPU context bar |
| RQ2 | candidates evaluated, per-candidate cost, total wall clock, offline/setup time for our search | `timings.csv` of the search stack (already dumped) + space-build time | ATiM tuning wall-clock, per-candidate cost |
| RQ3 | predicted vs measured cost, **per term** (transfer / kernel / combined), on the shared uniform sample | sample stack: predicted per-term at compile time + measured per-term breakdown | (nothing — ATiM not involved) |
| RQ4 | end-to-end whole program: per-operator arm vs whole-program arm, broken into kernel / weight scatter / **program load** | two arms of our own compiler (`graph-allocation` off/on) | (CPU context only) |
| A1  | geomean slowdown when a capability is disabled: MRAM tiling / scatter specialisation / both; infeasible cells reported as such | search stack re-run over 3 restricted spaces | — |
| A2  | rejected-but-lowers-fine fraction; accepted-but-fails (must be 0) | Cartesian sample outside feasible set + attempted lowering; the shared-sample bench doubles as the accepted-but-fails check | — |
| A3  | seed variance of reported configs | free: the search stack is multi-seed already | — |
| N1  | (conditional) quality + search time across cycle-accurate / fast / hybrid | search stack × 3 `simulator=` values | — |

Key structural fact (paper header: "THE MEASURED SAMPLE IS SHARED"): E1(b),
RQ3's fidelity ground truth, the best-vs-median number §2 quotes, and A2's
accepted-but-fails check are all served by **one** uniform sample of
n_sample=300 feasible configs per benchmark, measured on hardware once.

---

## 2. Shared building blocks

Each block is a doit task family, keyed by benchmark, connected by files.
Everything downstream consumes only the files, so partial results propagate.

### B0 `space` — dump the design space, nothing else
`cinm-opt --upmem-infer-accelerator="dump-space-only=true dump-dir=..."`
(new mode, §3.1) → `data/{bench}/space/space.json`.
Feeds: tab:sufficiency's |feasible|/|combinations| column, RQ2's setup-time
row, and the **manual transcription workflow** (§7), which is why it is P0.

### B1 `sample` — the shared uniform sample
`cinmopt.random_sample(n=OPTS.n_sample, seed=OPTS.sample_seed)`, with
**`sample-max-cost-ms` disabled** → `data/{bench}/sample/pool.csv` (params +
predicted cost). The default 2000 ms cap silently rejects slow configs and
would bias the sample — the paper's percentile (rule of three) and RQ3's
rank correlation are both void on a censored sample. The simulator eval
timeout is a different animal (corrected in review): hybrid runs
cycle-accurate under `eval-timeout-ms` and answers with the fast model when
it fires, so the timeout switches which model prices a config -- it never
drops one, hence no censoring, and with no timeout "hybrid" degenerates to
plain cycle-accurate. B1 therefore uses the search stack's exact pairing
(`simulator=hybrid, eval-timeout-ms=400`): RQ3's fidelity claim is about
the model the flow chooses with, timeout included. One seed, recorded in
OPTS, so the sample is *the* sample everywhere.

### B2 `measure(config-set)` — compile + bench a set of configurations
The core reusable block: takes rows of (fn, params, system-label), runs
`eval_solution_lowerer` (or `cinm1.lowerer` for the cinm1 arm), compiles the
bench binary, benches sequentially on hardware, leaves per-config
`output/*.csv`. This exists today inside `cinm1comparison/dodo.py`
(`_compile_one` / `_bench_one_config` / `_prev_bench_task_dep` chaining +
retry tasks); it must be **factored into `cinm_experiments/doit_blocks.py`**
(§4.1) so every stack below instantiates it instead of copying it.
Consumers: sample (E1/RQ3/A2), top-k (E1), search picks (E1/RQ1/A1/N1),
transcriptions (E1/RQ1), cinm1 sweep (RQ1), RQ4 arms.

### B3 `topk` — best-of-space by the cost model
`cinmopt.exhaustive_search` (predicted costs only, no hardware) with
`eval-timeout-ms=300` — a timeout is safe *here*, unlike B1: it only
mis-prices configs slower than 300 ms, which cannot be in the top anyway,
and it speeds the exhaustive sweep up significantly. Take top
k=OPTS.k_top rows → B2. Together with B1, defines E1's (b) = best measured
of (sample ∪ topk); the measured top-k also grounds RQ3's top-k-overlap
metric exactly instead of only within-sample.

### B4 `search` — the BO search stack
`cinmopt.bo_multiseed` (N seeds; N and spread reported as A3) →
`pools.best_per_seed` → B2 on each seed's pick. `timings.csv` per seed is
RQ2's raw data. Instantiated four ways: default space (E1(c)/RQ1),
three ablated spaces (A1), three simulators (N1, conditional), and — for
RQ4's per-operator arm — per compute block of a multi-op program.

### B5 `transcribe` — a manually-authored point, evaluated in our system
Input: a checked-in JSON file per benchmark (`points/atim/{bench}.json`,
`points/cinm1rule/{bench}.json`), each holding one *or several* candidate
param dicts (several = the paper's answer to transcription ambiguity:
measure all readings). Output: B2 on each candidate + an **invariant report**
(D, T, bytes scattered/gathered per operand, WRAM/tasklet — computed from
our compile artifacts, printed next to the values the transcriber derived
from the ATiM trace; §7). Used for E1 (a′) and RQ1's rule-decision row.

### B6 `assemble` — per-RQ CSV assembly, tolerant of holes
One task per results table: globs whatever B1–B5 produced, joins with the
offline interchange CSVs (§6.1), writes `results/{e1,rq1,...}.csv`, and
prints a MISSING report instead of failing when an input doesn't exist yet.
This is the layer that satisfies "run plotting with partial results":
plots depend only on `results/*.csv`, never on raw data.

---

## 3. Compiler / runtime changes (C++)

### 3.1 P0 — `dump-space-only` mode + inspectable permutations
*Pass*: `--upmem-infer-accelerator` gains `dump-space-only` (bool): build the
space, write `space.json`, commit nothing, run no search/sampling. Today
`space.json` is only written after a search runs
(`AcceleratorInference.cpp:515,928`).

*space.json content*, for the transcription workflow:
- For every `ParamKind::Permutation` parameter: the **item labels** (which
  iteration dim each permuted item is — `M`, `K`, …; SpaceBuilder must record
  them at `makePermutation` time), the **encoding note** (n dimensions, each
  the 1-based place of item i — `ConfigSpace.cpp` "ParmKind<Permutation>"),
  the exact **column/`eval-solution` names** of those n dimensions, and the
  full list of the n! orderings in human-readable form
  (`"K>M" ↔ [2,1]`-style) with their encodings. n ≤ 3-4 here, so listing all
  is fine; cap at some bound and emit only the mapping rule beyond it.
- For every parameter: which op it belongs to and what it means (one
  `doc` string per param, set where the space is built), so a human can map
  `op0.block0` → "MRAM tile of gemv dim M" without reading SpaceBuilder.

*Acceptance check*: transcribing the checked-in ATiM gemv trace
(`atim_gemv_1024_1_1024.py`) into `eval-solution` form using only
`space.json` + §7's procedure, without reading C++ sources.

### 3.2 Own phase (Phase 3a) — group residency: hoist alloc/load out of compute blocks (RQ4 blocker)

**STATUS 2026-08-13 — COMPLETE** (up to the hardware profiling-parity
check, which waits for Phase 1's campaign).
Done, as atomic commits with tests:
- *alloc/load split* (`upmem.alloc_dpus` allocation-only + new
  `upmem.load_program`, flow-sensitive program resolution, LLVM lowering to
  `upmemrt_dpu_alloc`/`upmemrt_dpu_load`, dedup/occupancy/verifier updates;
  the load op verifies tasklet-count against the hierarchy).
- *`cnm::CnmWorkgroupTypeInterface`* (type interface; implemented by
  `!cnm.workgroup` and `!upmem.hierarchy`) and the **forwarding contract**:
  `--convert-cnm-to-upmem` uses an in-scope ancestor block argument of
  matching workgroup shape instead of allocating, keeps the load in place,
  and does not free what it does not own (shape mismatch ⇒ local alloc;
  ambiguity ⇒ error). `BufferType` needed no change: forwarded values only
  ever stand in for the workgroup, and the substitution happens at
  conversion time, so no cnm op signature changed either.
Done (the commit-flow half):
- the **insert-alloc/free hook** (`InferencePlugin::materializeWorkgroupAlloc/
  Free`) and finalization's **shared-workgroup mode**: each pinned group's
  set is allocated once at the container-function top, freed at every exit,
  and forwarded into members as an ordinary compute_block operand; the
  member's lowering uses it and no longer allocs/frees. Timeshared groups
  and hook-less targets keep per-block allocation. Notes for posterity: the
  workgroup interface had to move into the *cinm* dialect (the
  compute_block canonicalizer must exempt forwarded workgroups from
  unused-arg deletion, and CinmIR cannot depend on CnmIR), and the
  forwarded-workgroup search must not cross IsolatedFromAbove boundaries.
- the acceptance case: lit test (two classes → two hoisted allocs) and 3mm
  end-to-end (three groups per function, one alloc each, loads on
  forwarded args, frees at exit).
Done (2026-08-13, closing the section):
- **load hoisting**: new `--upmem-hoist-load-programs` pass — a set whose
  every load references the same program gets a single load directly
  after its allocation; timeshared sets and block-argument sets stay
  untouched. Runs in the bench Makefile's host lowering after
  `--cinm-unwrap-compute-blocks --upmem-dedup-kernels` (program identity
  is symbol equality, so dedup must normalize first). On committed 3mm:
  three alloc+load pairs at the function top, the two identical gemv
  programs folded to one symbol. Wiring this up exposed a real
  **miscompile in `--upmem-dedup-kernels`**: the representative map was
  keyed by `sym_name`, and every kernel module names its program
  `@program`, so all loads were retargeted to whichever equivalence
  class the walk recorded last — fixed (keyed by op) and regression-
  tested before any campaign uses dedup.
Still open:
- re-run one class's profiling before/after on hardware once Phase 1's
  campaign starts, as the cheap regression check that hoisting did not
  leak into the profiling path (structurally it cannot: profiles are
  computed before finalization runs the hooks).
The conflation is **dialect-level, not just instrumentation**. Today (see
any graph-allocation output for 3mm): each `cinm.compute_block` body ends in
its own `upmem.alloc_dpus with program @kernels_N::@program : !upmem.
hierarchy<DxT>` … `upmem.free_dpus` — alloc and program load are bound in
one op, and the pair sits *inside* the block, so even a partitioned grid
re-allocs and re-loads on every execution. The runtime mirrors it
(`upmemrt_dpu_alloc` runs `dpu_alloc`+`dpu_load` under one timer,
`runtime/Upmem/upmem_rt.c:215-220`), and `net_time_ms` then *subtracts* the
whole thing as alloc. The whole-program arm cannot express its own claim in
this IR, so this gates all of RQ4.

**Target design** (decided in review, 2026-08-12): the core framework
inserts the alloc/free **once per group, outside the compute blocks**, and
forwards the hierarchy value into each member block. Placement, also
decided: **at the top of the container function** (free at its end). The
right long-term shape is a real init/deinit phase generated as separate
functions — out of scope for the evaluation; function-top placement is the
approximation, and the measurement convention absorbs the difference (next
paragraph). Consequences:
- the core interfaces gain an **insert-alloc/free hook** (the framework is
  target-agnostic, so it cannot name `upmem.alloc_dpus` itself);
- the forwarded value must type-check in target-agnostic verifiers, so
  `upmem.hierarchy` must implement a type interface —
  i.e. **`cnm.WorkgroupType` becomes a type interface**, and consequently
  `cnm::BufferType` is parameterized by a `WorkgroupTypeInterface` instead
  of a bare workgroup shape. This ripples through every user of
  WorkgroupType/BufferType: a **large, self-contained refactor** — its own
  phase (3a), implemented and tested independently of any experiment, not a
  side quest inside RQ4's data collection.
- the per-operator arm keeps today's per-block alloc+load, which is exactly
  the reload the experiment prices — no work needed on that side.

**Pipeline ripple** (identified in review — the def-use locality of the
workgroup is assumed all along the lowering, not just at the upmem level):
- `LinalgToCnm` creates the `cnm::WorkgroupOp` / `FreeWorkgroupOp` *inside*
  the block (`LinalgToCnm.cpp:582,646`); with hoisting these become
  insert-hook calls in the parent + a forwarded value.
- `CnmToUPMEM` **hard-casts** the workgroup's defining op
  (`cast<cnm::WorkgroupOp>(launch.getWg().getDefiningOp())`,
  `CnmToUPMEM.cpp:417`; free-user scan at `:601-614`) — a forwarded
  workgroup (block argument / cross-block value) makes this assert, not
  mis-lower. The pattern must consult the type/interface instead of the
  defining op.
- **Profiling runs are unaffected only under a proviso**: the per-class
  profiles L_i(D) must not include launch/program-load time — reload is
  priced by the allocation model itself (program-identity constraint /
  `program-reload-ms`), so counting it in the profile would double-count.
  Verify what the profiling simulator counts before relying on this.
- **The committing flow at the end of graph inference is the disturbed
  path**: finalization stamps each group member and commits it through the
  normal per-block pipeline (`GraphInference.cpp:318`), which today
  materialises a fresh workgroup per member. Commit needs a shared-
  workgroup mode: first member of a group triggers the hook's alloc in the
  parent, later members reference it, frees move to the end of the
  container function.
Phase 3a's acceptance case covers all four: the 3mm graph-allocation
output lowers end-to-end with one alloc per group, and the profiling
numbers for a class are identical before/after the refactor.

Smaller, independent pieces (can land before or with 3a):
- **Runtime**: `upmemrt_record_load(...)` around `dpu_load`, new
  `{prefix}_load.csv` — worth landing early so *every* Phase 1-2 run
  already records load time separately.
- **`measurements.py`**: `load_time_ms`, a `net_breakdown_ms` bucket, and an
  amortized/total pair of aggregations implementing the per-transfer rule.
  With function-top placement the rule operationalizes **structurally**: a
  load/static-scatter that occurs once per function invocation is init —
  the bench harness's repeated calls stand in for repeated inferences, so
  it amortizes and is discounted; one that recurs *within* a single
  invocation (the per-operator arm's per-block alloc+load) can never
  amortize and is counted. This is exactly the paper's "once per workload
  lifetime" without needing the init/deinit functions to exist yet.

### 3.3 P1 — scatter-specialisation toggle (A1 blocker)
**DONE 2026-08-13.** `enable-scatter-specialisation=false` on
`--upmem-infer-accelerator` disables both sites in the trial lowering:
the cnm site is `--cnm-scatter-optimizations` (constant scatter →
broadcast + on-device init of uniform buffers), skipped; the upmem site
is the broadcast narrowing inside `--upmem-specialize-transfers`, run
with `use-bc-xfer-codegen=false`. The block-collapsing transfer rewrites
stay on either way — they narrow form, not capability. **No space
parameter denotes the specialised forms**, so the space is identical
under both settings (the lit test replays the same eval-solution vector
through both). The standalone pipelines already control both sites
directly (omit the cnm pass; `use-bc-xfer-codegen=false`), which is what
the RQ1 parity check uses. `use-mram-tiling` already existed.

### 3.4 P1 — A2's rejected-region sampling
**Compiler half DONE 2026-08-13**: `eval-solution-force=true` skips the
membership check (completeness and unknown-name checks still apply, the
guarded error stays the default) and lets the lowering deliver its own
verdict; lit-tested on a rejected point in both modes.
Then A2 = python-side (still to write, Phase 4): sample the Cartesian
domains from `space.json`, drop rows in the feasible set (membership via
`dump-full-pool` of B3), `eval-solution-force` the rest, count
lowers-fine (and optionally runs-fine).
`accepted-but-fails` needs nothing: any compile/run failure in B1/B3's
measured sets is that counter, and the paper wants it run first — which the
phase order (§8) honours since B1 is Phase 1.

### 3.5 P2 — small RQ2 completeness item
**DONE 2026-08-13.** `buildConfigSpace` times its whole body
(declaration, solve, enumeration) into `ConfigSpace::buildWallSeconds`;
every `space.json` reports it as `space_build_seconds` (it is RQ2's
"offline/setup time" for our side; the search side is already in
`timings.csv`).

### 3.6 Optional extension (nice-to-have, gates nothing) — concurrent group execution via the `async` dialect

Recorded 2026-08-13. The whole-program arm gives independent operator
classes *disjoint* groups, so blocks with no data dependency (3MM's
parallel structure, the transformer's QKV) could in principle execute
concurrently — but execution is synchronous end to end: `upmem.wait_for`
blocks the host thread until the launch retires, so blocks serialize in
program order regardless of the allocation. Two consequences:

- **Interpretation guard for RQ4** (mirrored as a comment in the paper's
  §8.4): the parallel instances demonstrate that whole-program allocation
  *preserves staticity* — no reload, no rescatter — which is the actual
  contribution. They do **not** demonstrate concurrency speedup, and
  fig:wholeprogram's text must not read as if disjoint groups overlap in
  time. All RQ4 deltas come from the load/scatter terms.
- **The extension, if we want the parallel bars to also show overlap**: a
  pass lowering the committed `cinm.compute_block` graph onto the upstream
  `async` dialect (https://mlir.llvm.org/docs/Dialects/AsyncDialect/).
  Sketch: outline each compute_block into an `async.func`; each SSA edge
  between blocks becomes an `async.token`/`async.value<...>` dependency;
  `async.await` only where a result is consumed by the host or at function
  exit. Correctness structure falls out of what Phase 3a already built:
  pinned groups are disjoint by construction, so concurrent launches on
  different groups have no device-level hazard, while members of one
  (merged or timeshared) group are exactly the ones whose tokens must
  chain — the dependency edges plus a per-group serialization token give
  both. The async dialect is used to *represent* the concurrent program
  and its dependencies — building our own future/token type and await op
  into cinm would just re-invent `!async.token`/`!async.value` — but that
  does not commit us to the async *runtime*: the tokens can be lowered
  directly to upmem-dialect calls backed by the UPMEM SDK's own
  asynchronous primitives (`dpu_launch(set, DPU_ASYNCHRONOUS)`, per-set
  `dpu_sync` behind `upmem.wait_for`), keeping one host thread and no
  `upmemrt` thread-safety work.
- Not scheduled in any phase. If it lands, it slots after Phase 3a with no
  changes to the RQ4 harness (same drivers, same breakdown; the kernel
  segments of independent blocks simply overlap in wall clock).

---

## 4. Harness changes (`experiments/cinm_experiments/`)

### 4.1 P0 — factor the doit blocks out of `cinm1comparison/dodo.py`
New `cinm_experiments/doit_blocks.py` exposing what §2-B2 needs as reusable
task generators: `measure_config_set(...)` (compile task + chained bench
task + markers), the global bench chain (`_prev_bench_task_dep` logic — must
now chain across *stacks*, not just within one dodo), and
`retry_failed_{compiles,bench}`. `cinm1comparison/dodo.py` becomes a client;
behaviour-preserving refactor, verified by `doit list` parity there.

### 4.2 P0 — `pools.py`/`space.py` additions
- `pools.top_k(pool_csv, k)` (B3), `pools.sample_rows(pool_csv)` (B1→B2 glue).
- new `cinm_experiments/space.py`: load `space.json`, expose params/domains/
  permutation tables (also used by the A2 Cartesian sampler and §7 helper).

### 4.3 P1 — predicted-vs-measured join for RQ3
`aggregate.py` already aggregates predicted `cost.csv` per compile and
`measurements.PREDICTED_TO_MEASURED` already aligns buckets. Missing:
a `fidelity_frame(sample_dir)` that emits one row per (config, term) with
`predicted_ms, measured_ms, share_of_total` for terms
{transfer, kernel, combined} — the direct input of fig:fidelity and the
paper's share-of-time weighting. Check the sample's compile step actually
dumps `cost.csv` per config (it comes from `annotate-op-costs` /
compute_cost path in `compile_run.py` — wire it into B2 if not on by
default).

### 4.4 P1 — cinm1 arm fixes
`cinm1.py` exists; the dodo TODO already lists "Fix the CINM1-like flow".
Add: sweep-coverage accounting (every (D,T) attempted → compiled?
lowered? ran?) so RQ1's coverage sentence and the optional no-`--cinm-tiling`
coverage variant fall out of bookkeeping instead of a separate run.
Resolved (review 2026-08-12): `--cinm-infer-tiling-factors` has **no knobs
besides (D,T)** — the sweep covers its full free space, and the paper's
"VERIFY BEFORE SWEEPING" note can be marked settled.

### 4.5 P2 — RQ4 workloads and drivers
- `bench/` drivers are per-single-op; multi-op programs (2mm/3mm/transformer)
  need a whole-program driver (generic: call the generated host `main` per
  iteration, verify once, time end-to-end) — likely one templated driver
  rather than three hand-written ones.
- Transformer block: synthetic MLIR with LLaMA-2-110M shapes/dependency
  structure, i32 (generate from `samples/transformers/llama2.c` shapes or
  write directly; "one block is enough", N3 is the stretch).
- Add `3mm_seq.mlir` and `3mm_par.mlir` next to the existing
  `3mm_parseq.mlir` (= the mixed case). Grid is 2MM {seq, par} and
  3MM {seq, par, mixed} — mixed only exists for 3MM; the paper's "for both"
  wording should be adjusted to match.

---

## 5. `experiments/evaluation/dodo.py`

New directory (the old scratch `experiments/evaluation/` was removed; this
recreates it as the single paper-evaluation pipeline). Layout:

All experiment constants are OPTS at the top of the dodo — they are the
paper's numbers, so they are set once and every task derives from them.
Proposed values with their rationale (each is a knob; changing one degrades
a stated precision, not correctness):

| OPTS | value | why this number |
|---|---|---|
| `n_sample` | 300 | rule of three: 0-of-n beaten ⇒ 95% bound 3/n = **1%** — "top 1% of the feasible space" is the number the paper quotes (decided in review; 200 ⇒ 1.5%). RQ3's Spearman CI tightens along (Fisher z half-width ≈ 1.96/√(n−3) ≈ 0.11). |
| `k_top` | 200 | 50 would suffice for E1(b); 200 gives the measured ground truth for RQ3's top-k overlap up to k=200 at the same per-run cost. |
| `n_seeds` | 32 | matches the cinm1comparison campaign; A3's spread over 32 seeds is a stable percentile; halving it saves little (seed picks are cheap runs). |
| `iters` | 6 | per-run measurement repetitions, as in cinm1comparison. |
| `sample_seed` | fixed constant | B1's sample is *the* shared sample; the seed is part of the paper's methodology statement. |
| `exhaust_timeout_ms` | 300 | B3 only (never B1): mis-prices only configs >300 ms, which cannot be top-k; large exhaustive-sweep speedup. |

```
experiments/evaluation/
  dodo.py                  # tasks below; OPTS at top = the table above
  points/atim/{bench}.json      # manual transcriptions (checked in, reviewed)
  points/cinm1rule/{bench}.json # the rule's decision, transcribed
  offline/atim.csv  offline/prim.csv  offline/cpu.csv   # interchange, §6.1
  data/{bench}/space/           # B0
  data/{bench}/sample/          # B1 pool.csv + B2 compiled/ run/
  data/{bench}/topk/            # B3 + B2
  data/{bench}/search/          # B4 seed_k/ + B2 on picks
  data/{bench}/search_ablate_{mram,scatter,both}/   # A1
  data/{bench}/cinm1/           # RQ1 sweep (D,T)/
  data/{bench}/points_{atim,cinm1rule}/             # B5
  data/{prog}/rq4_{peroper,wholeprog}/              # RQ4 arms
  results/                 # B6 assembled CSVs, one per table/figure
  plots/  tables/          # emitted PDFs and .tex
```

Task graph (per single-op benchmark; ⇒ = file dependency):

```
space ⇒ (nothing; leaf used by assemble + humans)
sample ⇒ compile_sample ⇒ bench_sample
exhaust_pred ⇒ topk ⇒ compile_topk ⇒ bench_topk
search ⇒ compile_search ⇒ bench_search          (×4 spaces ×(opt.) 3 sims)
points/*.json ⇒ compile_points ⇒ bench_points ⇒ invariants_report
(cinm1) pairs ⇒ compile_cinm1 ⇒ bench_cinm1
all bench_* ⇒ assemble_{e1,rq1,rq2,rq3,a1,a2} ⇒ plot_* / table_*
```

Rules carried over from `cinm1comparison/dodo.py`, now via `doit_blocks`:
compile parallel & fallible-per-config, bench strictly sequential in ONE
global chain across all stacks (hardware timing), markers + retry tasks,
`ProgressBarReporter`. Every `assemble_*` and `plot_*` task is
missing-tolerant (§2-B6) and additionally registered under a plain
`doit plots` umbrella so a partial `data/` still yields every derivable
figure.

Hardware-run budget (sanity): per single-op benchmark ≈ 300 (sample) + 200
(topk) + N seeds×4 spaces + ~10 points + cinm1 sweep ≈ 500-600 short runs
×7 benchmarks — same order as the cinm1comparison campaign; fine
sequentially over a few nights, and doit resumes.

---

## 6. Plots and tables

### 6.1 Offline interchange format
`offline/{atim,prim,cpu}.csv`:
`benchmark, fn_name, system, config_label, total_ms, scatter_ms, kernel_ms,
gather_ms, load_ms, tuning_wallclock_s, notes`, one row per measured point,
component columns nullable. Nothing else in the pipeline knows how these
were obtained. Provenance, per source:
- **ATiM**: their reproduction script, with **their runtime module
  instrumented to measure all times** — so component subtotals and total
  under *our* timing convention come from their own harness, and the same
  instrumented runs yield `tuning_wallclock_s` for tab:walltime. ATiM rows
  for E1 additionally reference their trace file (for §7).
- **PrIM**: the hand-optimized kernels, timed under the same convention.
- **CPU**: **TVM-autotuned CPU baselines** (the context bar is itself
  autotuned, not naive — worth one methodology sentence in the paper, since
  it strengthens the "context only" framing).

### 6.2 Inventory (each = one script in `evaluation/`, input = `results/*.csv` only)

| Artifact | Script | Inputs (results/) | Notes |
|---|---|---|---|
| tab:sufficiency | `table_sufficiency.py` | e1.csv | (a) offline; (a′)(b)(c) ours; percentile + rule-of-three CI computed here; invariants to caption |
| fig:quality | `plot_quality.py` | rq1.csv | bars normalized to PrIM: PrIM/ATiM/ours/CPU; CINM 1.0 → companion table |
| tab:quality-configs | `table_quality.py` | rq1.csv | best config per benchmark + CINM 1.0 rows (published, best-of-sweep, rule-decision percentile, coverage fraction) |
| tab:walltime | `table_walltime.py` | rq2.csv | ours from timings.csv + space time; ATiM column offline |
| fig:fidelity | `plot_fidelity.py` | rq3.csv | predicted-vs-measured panels per term + combined; Spearman/Kendall/top-k overlap, MAPE on transfer term |
| fig:wholeprogram | `plot_wholeprogram.py` | rq4.csv | stacked kernel/scatter/load, two arms × {2mm,3mm}×{seq,par,mixed} + transformer |
| tab:capability | `table_capability.py` | a1.csv | geomean slowdowns; infeasible cells printed as ∅, not dropped |
| (text numbers) | `numbers.py` | a2.csv, e1.csv, rq2.csv | A2's two fractions, A3 seed spread, best-vs-median for §2 — emitted as `\newcommand` defs in `tables/numbers.tex` |

Style: reuse `cinm1comparison/plot.py` + `cinm_experiments/plots.py`
conventions (geomean helpers, breakdown colors from
`measurements.net_breakdown_color_ix`).

---

## 7. Manual workflow: ATiM → our parameter vector

Procedure (per benchmark, per ATiM trace file, e.g.
`atim/evaluation/results/tuned_modules/atim_gemv_1024_1_1024.py`):

1. Read the `apply_trace_*` function, not the lowered module: the
   `sample_perfect_tile(..., decision=[...])` lines carry every tiling
   factor; `bind(...thread_axis="blockIdx.*")` rows give D (product of bank
   axes) and `threadIdx.x` gives T; `sch.reorder(...)` + the split structure
   give the loop order; `rfactor` = host-side reduction split (K-split).
2. Open `data/{bench}/space/space.json` (B0): map each decision onto our
   named params using the per-param `doc` strings; for the order parameter,
   look the loop order up in the listed permutation table and copy its
   encoding. The `eval-solution` contract (confirmed in code): one
   `name=value` pair per *dimension*; a permutation parameter of n items
   occupies dimensions `name[0]..name[n-1]` (`SearchParam::dimName`,
   `ConfigSpace.cpp:39-44`), each holding the **1-based place of item k**;
   there is one `order` parameter per op, and ops with a single iteration
   dim have none. Missing/unknown names produce an error that lists the
   expected set (`AcceleratorInference.cpp:310-323`), so a wrong guess is
   loud, and infeasible values are rejected by feasible-set membership
   (`:326-337`).
3. Write `points/atim/{bench}.json`; where a reading is ambiguous, write all
   plausible candidates (the pipeline measures each; paper: enumerate, don't
   average the worry away).
4. Run `doit bench_points:{bench} invariants_report:{bench}` — the report
   prints, side by side, D/T/bytes-per-operand/WRAM-per-tasklet from our
   compiled artifact vs the same quantities the helper computes from the
   trace. Agreement = transcription faithful in every cost-determining
   respect (paper's structural validation); disagreement names the axis.

Tooling that makes this "easy enough" actually easy:
- **P0**: B0's `space.json` upgrades (§3.1) — the current blocker; you can't
  even see the permutation encoding without reading `ConfigSpace.cpp`.
- **P1**: `evaluation/transcribe_atim.py --trace <file> --space <space.json>`:
  parses `sample_perfect_tile/bind/reorder/rfactor` lines (they're regular),
  proposes a candidate params JSON with a per-field provenance comment, and
  computes the trace-side invariants for step 4. Human reviews and commits;
  the tool never writes `points/` directly.
- The same procedure with `points/cinm1rule/` for the rule's decision: run
  the cinm1 pipeline once per benchmark, read the inferred tiling factors
  (from IR attributes or its debug output — small dump flag if neither is
  readable today), transcribe into the MRAM-tiling-disabled subspace.

---

## 8. Phases

**Phase 0 — unblock manual + shared work (P0, do first)**
STATUS 2026-08-12: **landed except hardware validation.** §3.1
dump-space-only + permutation/doc dump (acceptance passed: the ATiM gemv
trace transcribes from space.json alone; eval-solution's membership error
now prints the nearest feasible configuration, which localised a real
finding — ATiM's 4096-worker gemv point is infeasible in prim_gemv's space
because the block couples gemv with the extent-1024 elementwise, exactly
the plan-§7 ambiguity case). §3.2's runtime half: load timed separately
from alloc; `net_time_ms` subtracts it by default (convention preserved),
`net_breakdown_ms(count_load=True)` is the RQ4 view. §4.1 doit_blocks
refactor (cinm1comparison converted; bench chain verified intact over all
46310 tasks). §4.2 pools.top_k/sample_rows + space.py. evaluation/dodo.py
with B0+B1(+B2-on-sample) tasks; B0 ran for real, B1 smoke-tested (n=3).
Remaining exit item: `doit bench_sample:prim_gemv` on the machine with
DPUs. Still open from the gating dodo TODOs: "Fix the CINM1-like flow",
"Integrate Hamid's latest cost model", "Remove the last host zero-init
loops" — the last one biases every kernel-term measurement.

**Phase 1 — the shared sample + E1 + RQ3 (highest paper value per run)**
B1/B2/B3 across all 7 benchmarks (this run IS A2's accepted-but-fails
check — paper says run it first); §4.3 fidelity frame; B4 default space;
B5 + `transcribe_atim.py`; assemble/plot e1, rq3, and §2's number.

**Phase 2 — RQ1 + RQ2 + A1**
cinm1 sweep + coverage (§4.4); rule-decision points; ablation toggles
(§3.3) + ablated searches; walltime table; offline CSVs folded in as ATiM/
PrIM results arrive.

**Phase 3a — group-residency refactor (compiler-only, no experiments)**
STATUS 2026-08-13: **landed** (was the long pole). §3.2's target design:
type interface, insert-alloc/free hook, hoisted per-group alloc, read_only
operand casts, and load hoisting (`--upmem-hoist-load-programs` after
dedup in the bench Makefile — which exposed and fixed a dedup miscompile,
see §3.2). Acceptance held: committed 3mm lowers end-to-end with one
alloc+load per group at the function top and zero defensive copies. The
runtime `load` timer landed in Phase 0, so all campaigns separate load
from alloc. Remaining: the profiling-parity spot check on hardware once
Phase 1 runs. §3.3 (ablation toggle) and §3.4's compiler half landed the
same day, so Phase 2/4's compiler prerequisites are also done.

**Phase 3b — RQ4 measurements**
§4.5 drivers + transformer block + `3mm_{seq,par}.mlir`; both arms;
fig:wholeprogram.
(Scheduled after 3a by necessity, but RQ4 is Tier-1 and the paper's C3
evidence — if earlier phases slip, RQ4 must not be the thing that gets
cut.)

**Phase 4 — conditional / polish**
A2 rejected-region sampling (§3.4); N1 only if RQ3 shows kernel-dominated
residual; numbers.py; freeze `results/` for the paper. Strictly optional
beyond that: §3.6 async lowering for concurrent group execution — only if
everything above has landed.

---

## 9. Review log (2026-08-12) — questions resolved, one open

Resolved in review; the body sections above reflect these:

1. **OPTS constants**: parameters at the top of the dodo; values +
   rationale in the §5 table. **n_sample = 300, decided in review** (3/300 =
   1% at 95%, the rounder number for the paper); k_top=200 ⇐ grounds RQ3's
   top-k overlap; n_seeds=32 ⇐ cinm1comparison precedent.
2. **CPU context bar** = TVM-autotuned CPU baselines, offline input
   (`offline/cpu.csv`, §6.1).
3. **RQ4 blocker is dialect-level**: alloc op conflates allocation with
   program load, IR allocates per compute block; whole-program arm needs
   the alloc/load split to allocate once per group (§3.2). Long pole of
   Phase 3.
4. **eval-solution** validates by materialised-set membership
   (`AcceleratorInference.cpp:326-337`) → `-force` flag is the cheap A2
   route (§3.4).
5. **Order parameter contract** confirmed: one permutation param per op,
   passed as `name[k]=place` per dimension, 1-based places (§7 step 2).
6. **Workload grid**: mixed exists only for 3MM; add `3mm_{seq,par}.mlir`
   (§4.5); adjust the paper's "sequential/parallel/mixed for both" wording.
7. **ATiM walltime + components**: their reproduction script with their
   runtime module instrumented by us, same timing convention (§6.1).
8. **B3 exhaustive sweeps** run with `eval-timeout-ms=300` (safe: only
   mis-prices configs that cannot be top-k). **B1 uses the search stack's
   own `hybrid` + `eval-timeout-ms=400`** — corrected in review: the hybrid
   timeout switches to the fast model rather than dropping the config, so
   it does not censor; only `sample-max-cost-ms` censors, and that stays
   off in B1. With no timeout, "hybrid" is just cycle-accurate.
9. **`--cinm-infer-tiling-factors` has no knobs beyond (D,T)** — the sweep
   is its full free space; the paper's "VERIFY BEFORE SWEEPING" is settled.

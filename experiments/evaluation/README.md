# Paper evaluation suite

The single pipeline behind the paper's evaluation (§8): a `doit` project
whose stages are connected by files, so partial results propagate and any
subset can be (re)run. Every assemble/plot stage tolerates missing inputs
— `doit plots` is safe at any point of the campaign and renders whatever
is derivable from the data collected so far.

## What is measured

| Exp | Question | Arms run by this pipeline | Side inputs |
|-----|----------|---------------------------|-------------|
| E1  | Does the space contain ATiM-grade points, and does the search find them? | transcribed ATiM points; the shared sample; measured top-k-by-model; every seed's search pick — "space best" is the best measured point of *any* of these arms | ATiM best + trace |
| RQ1 | End-to-end quality vs baselines | search picks; CINM 1.0 (D,T) sweep + rule-decision point | PrIM, ATiM, CPU rows |
| RQ2 | What a search costs | search `timings.csv` + space-build time | ATiM tuning wall clock |
| RQ3 | Cost-model fidelity, per term | predicted-vs-measured join over the shared sample | — |
| RQ4 | What per-operator tuning cannot see | two arms of our own compiler: per-operator (`graph-allocation` off) vs whole-program (on), broken into kernel / scatter / **program load** | — |
| A1  | What each capability is worth | search re-run over 3 restricted spaces (`use-mram-tiling=false`, `enable-scatter-specialisation=false`, both) | — |
| A2  | What the constraint system wrongly excludes | forced lowering of Cartesian draws outside the feasible set | — |
| A3  | Seed variance | free: the search stack is multi-seed | — |
| N1  | (conditional) simulator quality/speed trade | search stack × 3 `simulator=` values | — |

Benchmarks: the PrIM-derived single-operator set
`prim_{va,red,mtv,ttv,mmtv,gemv,geva}` (from `experiments/prim_*.mlir`)
for E1/RQ1/RQ2/RQ3/A1/A2, plus — still to be added, see TODO — the
multi-operator set (`2mm_{seq,par}`, `3mm_{seq,par,parseq}`, one
transformer block) for RQ4.

**The measured sample is shared.** One uniform draw of `n_sample`
feasible configs per function (fixed seed, never redrawn) serves E1's
best-of-sample, RQ3's fidelity ground truth, the best-vs-median number,
and the accepted-but-fails check at once. The rule-of-three arithmetic
(`0-of-n beaten ⇒ top 3/n of the space at 95%`) is why `n_sample=300`
reads as "top 1%". All such constants live in `OPTS` at the top of
`dodo.py`, each with its rationale; they appear verbatim in the paper, so
they are never set anywhere else.

## Layout

```
dodo.py            all tasks; OPTS at the top
assemble.py        folds data/ into results/*.csv  (schemas in its docstring)
plot_*.py table_*.py  one script per figure/table; read results/*.csv ONLY
paper_numbers.py   inline paper numbers as \newcommand defs (tables/numbers.tex)
invariants.py      recompute a transcription's D/T/bytes/WRAM from its artifact
points/            manually-authored configurations (see points/README.md)
offline/           side-input CSVs (see below); not generated here
data/{bench}/      per-stack raw data (gitignored)
  space/           space.json per function            [doit space]
  sample/          shared sample: pool.csv, compiled/, run/
  topk/            exhaustive predicted pool + measured top-k
  search/          BO dumps (seed_*/), compiled/, run/
  search_ablate_*/ same, restricted spaces
  points_*/        measured transcription candidates
  a2/              forced-lowering probes (probe.csv)
results/ plots/ tables/   derived (gitignored); frozen by hand for the paper
```

## Running

Prerequisites: `build/bin/cinm-opt` (and `cinm-translate`), the UPMEM SDK
(`UPMEM_HOME`) for compiling bench binaries, actual DPUs only for
`bench_*` tasks. Python: `doit`, `pandas`, `numpy`, `matplotlib`,
`scipy`.

CPU-only, safe anywhere (parallelize with `doit -n <N>`):

```sh
doit space split        # B0: space.json per function + per-fn modules
doit sample             # B1: draw the shared sample (simulator runs)
doit exhaust_pred       # B3: price the whole feasible set (hours)
doit search search_ablate   # B4: BO, default + 3 restricted spaces
doit a2                 # forced-lowering probes of the rejected region
doit compile_sample compile_topk compile_search compile_search_ablate compile_points
doit invariants_report  # transcription cross-check (needs compile_points)
```

Hardware (the DPU machine). Bench tasks run strictly one config at a
time, in one fixed global order — sample → topk → search → ablated
searches → points — enforced by task dependencies, so just:

```sh
doit bench_sample bench_topk bench_search bench_search_ablate bench_points
```

Interruption is fine: per-config `.done` markers make every stage
resumable. A config that failed is *recorded*, not retried implicitly —
after fixing the cause:

```sh
doit retry_failed_compiles && doit <compile tasks>
doit retry_failed_bench    && doit <bench tasks>
```

Derived outputs, at any stage:

```sh
doit assemble plots numbers   # results/*.csv -> plots/, tables/
```

Compile failures in the sample/topk stacks are data, not noise: an
eval-solution config that the space accepted but that fails to build is
the accepted-but-fails counter (the paper claims it is 0) — keep the
logs.

## Measurement conventions

- **Amortizability**: `measurements.net_time_ms` subtracts alloc, free
  and program load — a cost paid once per function invocation is init
  and amortizes over repeated inferences. RQ4 uses the *undiscounted*
  breakdown (`net_breakdown_ms(count_load=True)`): a load or rescatter
  recurring within one inference can never amortize, and pricing exactly
  that is RQ4's point.
- **The cost model is one thing**: `simulator=hybrid` +
  `eval-timeout-ms` together (cycle-accurate under the budget, fast
  model past it). The timeout switches models, it never drops a config —
  so it does not censor the sample. What would censor is
  `sample-max-cost-ms` (reject-and-resample), which stays off for the
  shared sample. The exhaustive sweep alone uses the shorter
  `exhaust_timeout_ms`, which can only mis-price configs too slow to
  compete for the top anyway.
- **Every seed's search pick is measured**; the spread of picks over
  seeds is a reported number (A3), so identical picks are not
  deduplicated. The search is *reported* as its **median seed**, never
  best-of-N — seed spreads reach ~8× (first campaign), so best-of-N would
  report the max of a wide distribution as the outcome. The scalar for
  search quality is **regret** (median seed / space best); the per-seed
  scatter is the strip in fig:sufficiency.
- **The exhibited witness**: "space best" means the best *measured* point
  from any arm — sample, top-k, or a search pick. A witness does not care
  where it came from; restricting it to sample∪topk made "best known"
  false whenever a search pick won its row. Consequently, space best is
  never compared against the search (it contains it); the search's
  internal baseline is measured top-k, which shares the search's cost
  model and so isolates the search loop from model fidelity.

## Side inputs (provided, not computed here)

1. **`offline/{atim,prim,cpu}.csv`** — measured baseline rows in the
   interchange schema (`assemble.INTERCHANGE_COLUMNS`): `benchmark,
   fn_name, system, config_label, total_ms, scatter_ms, kernel_ms,
   gather_ms, load_ms, excluded_transfer_ms, excluded_transfer_bytes,
   tuning_wallclock_s, notes`; component columns nullable, one row per
   measured point. Nothing else in the pipeline knows how these were
   obtained. Provenance requirements:
   - *ATiM*: their reproduction script, with their runtime module
     instrumented to time components under the same convention as ours;
     the same runs yield `tuning_wallclock_s`. E1's ATiM rows must be
     traceable to the trace file the transcription used. `atim_eval.py`
     records the measurements and `build_interchange.py --install
     <here>/offline` derives this file from them; regenerating it needs
     no hardware.
     `excluded_transfer_*` is what their timing convention leaves out:
     the operands named by `pragma_explicit_h2d`, device transfer only
     (their runtime's copy into a padded host buffer is staging, and is
     excluded on both sides — see `notes` for which operands).
   - *PrIM*: the hand-optimized kernels, timed under the same
     convention.
   - *CPU*: TVM-autotuned CPU baselines (the context bar is itself
     autotuned, not naive — worth a methodology sentence in the paper).
2. **`points/atim/{bench}.json`, `points/cinm1rule/{bench}.json`** —
   manual transcriptions of ATiM's tuned schedule and of CINM 1.0's rule
   decision into our parameter space. Format, workflow and the
   invariant cross-check are documented in `points/README.md`; the raw
   material is each function's `space.json` (parameter docs +
   permutation tables), which is why `doit space` is a prerequisite.
3. **The compiler itself** — everything here trusts `cinm-opt` and the
   cost model behind `--upmem-infer-accelerator`; retuning or replacing
   the cost model invalidates predicted costs (pools, top-k, searches)
   but not hardware measurements.

## Open items

Gating — must land before the real campaigns are worth hardware time:

- integrate the updated (Hamid's) cost model; every predicted cost and
  every search decision depends on it;
- fix the CINM1-like flow (`cinm_experiments/cinm1.py`);
- remove the last host zero-init loops — they bias every kernel-term
  measurement.

Then, in rough order:

- **cinm1 sweep**: (D,T) sweep tasks with coverage accounting (every
  pair attempted → lowered? compiled? ran?), so RQ1's coverage sentence
  falls out of bookkeeping. `--cinm-infer-tiling-factors` has no knobs
  besides (D,T), so the sweep covers its full free space.
- **RQ4 workloads and drivers**: `3mm_seq.mlir`/`3mm_par.mlir` next to
  the existing `3mm_parseq.mlir` (mixed exists only for 3MM), a
  transformer block with LLaMA-2-110M's shapes and dependency structure
  (synthetic, i32 — say exactly that, never "we ran LLaMA-2"), and one
  templated whole-program bench driver; then fill `RQ4_PROGRAMS` in
  `dodo.py` and add the two-arm stacks (the assemble/plot side is
  already waiting for them).
- **`transcribe_atim.py`**: parse an ATiM trace
  (`sample_perfect_tile`/`bind`/`reorder`/`rfactor` lines are regular)
  and propose a candidate `points/` JSON with per-field provenance; the
  human reviews and commits, the tool never writes `points/` directly.
- **Hardware validation**: a first `doit bench_sample:prim_gemv` smoke
  on the DPU machine; and a one-class profiling before/after spot check
  that the group-residency refactor did not leak into the profiling
  path.
- **`paper_numbers.py`**: wire the e1/rq2 numbers once those campaigns
  produce rows (the slots degrade to comments today).
- **N1** only if RQ3 shows a kernel-dominated residual.
- Freeze `results/` for the paper (they are gitignored until then).

## Possible extensions

- **Near-boundary A2 sampling.** The uniform Cartesian draw lands almost
  entirely deep in the infeasible region, so A2 as run validates the
  constraint system over the whole domain, not specifically at the
  feasibility boundary. A one-step-perturbation sampler around feasible
  points would target the boundary; the nearest-feasible machinery
  already exists in the compiler's membership diagnostics.
- **Concurrent group execution.** Execution is synchronous end to end
  (`upmem.wait_for` blocks the host), so RQ4's parallel instances show
  that whole-program allocation preserves staticity — not concurrency
  speedup, and its figure must not read as if disjoint groups overlap in
  time. The extension: lower the committed compute-block graph onto the
  upstream `async` dialect purely as a *representation* (its
  token/value types instead of homegrown futures), then lower tokens
  directly to UPMEM SDK asynchronous primitives
  (`dpu_launch(DPU_ASYNCHRONOUS)`, per-set `dpu_sync`) — one host
  thread, no MLIR async runtime, no runtime thread-safety work. Pinned
  groups are disjoint by construction, so the dependency edges plus one
  per-group serialization token are the whole correctness argument. If
  it lands, the RQ4 harness needs no changes; independent blocks' kernel
  segments simply overlap in wall clock.
- **N3**: the full 24-layer transformer end to end — demonstrates what
  one block already demonstrates, at 24× the integration cost; only
  worth it as a stretch.

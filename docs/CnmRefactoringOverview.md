# CNM refactoring: overview

A one-page map of the `cinm → linalg → cnm → upmem` refactoring: the flow
we had, the flow we are building, what was broken along the way, and what
is left. Full reasoning lives in
[CnmMemoryLevelsDesign.md](CnmMemoryLevelsDesign.md) (§A–§I); the
milestone record is in
[CnmMemoryLevelsImplementationPlan.md](CnmMemoryLevelsImplementationPlan.md).

**The thesis in one line.** Passes expose *parameters*; they do not take
decisions. Every distribution decision the old flow made by case analysis
is now a number the UPMEM inference plugin supplies, and the plugin
derives those numbers from the op's own iteration space rather than from a
per-op template.

## 1. The CINM 1.0 flow, and who decided what

Transcribed in [cinm1.py](../experiments/cinm_experiments/cinm1.py) from
`testbench/Makefile`, still live as the comparison baseline.

| Stage | Pass | Decisions it took |
|---|---|---|
| 1a | `--cinm-infer-tile-sizes` | **Tile sizes**, from the accelerator's `computeTilingFactors` — a fixed heuristic per op, no search. |
| 1a | `--cinm-tiling` | Emits the host-side loop nest over tiles, via `CinmTilingInterface` per op kind. |
| 2 | `--convert-cinm-to-cnm` | **Everything about the distribution**: how a tile's elements map onto workgroup leaves, whether an operand is broadcast / chunked / transposed, what the per-leaf buffer shape is, and what the launch body computes. |
| 3–5 | bufferize, hoist, `--cnm-ensure-scatter-gather-contiguous` | Mechanical. |
| 6 | `--convert-cnm-to-upmem` | MRAM alloc + WRAM staging around every launch, unconditionally. Transfer API selection via `use-sg-xfer-codegen` / `use-bc-xfer-codegen`. |

The weight is all in step 2. `computeShapeOfTensors` is ~200 lines of
case analysis — "tensor has exactly one parallel element, so broadcast";
"parallel elements equal `|WG|` but chunks are not contiguous, so
`linalg.transpose`, and the transpose has to happen in the caller";
"flatten trailing parallel dims until `trailing % k == 0`, then maybe emit
a reshape". Each case *guesses* a distribution from the shapes.

Three consequences shaped everything that follows:

- **The decisions are not addressable.** DSE can move the tile sizes and
  nothing else, so the search space is whatever the templates expose,
  which is why `SimulationTemplates.cpp` exists at all: to reach
  configurations the real pipeline could not express.
- **Reduction splitting is inexpressible.** A reduction dimension spread
  across the workgroup has no representation, so any configuration
  needing partial results per leaf is simply out of reach.
- **Levels are implicit.** `cnm.buffer`'s `level` field was decorative;
  `--convert-cnm-to-upmem` staged MRAM→WRAM the same way for every
  program, so on-device staging could not be tiled, hoisted or shared.

## 2. The CINM 2.0 flow

The whole picture, including the parts not yet implemented — those are
marked **[todo]** and collected with due dates in §4.

```
--convert-cinm-ops-to-linalg      cinm op -> linalg, carrying attributes
                                  and a cinm.lowered_from marker
  [todo] fusion                   producer/consumer fusion on linalg
--convert-linalg-to-cnm           distribute onto the workgroup
    cnm-buffer-level=mram         <- buffers live in MRAM
    per-dim-attrs=...             <- attrs that must survive a split
bufferize / hoist / CSE
--upmem-tile-mram-buffers         tile + promote MRAM -> WRAM, staging
                                  with cnm.local_transfer
--convert-linalg-to-affine-loops, --affine-scalrep
--cnm-ensure-scatter-gather-contiguous
  [todo] scatter specializations  broadcast, constant-scatter
--convert-cnm-to-upmem            level-aware: MRAM args bind straight to
                                  the static alloc
  [todo] occupancy check          exact, on the lowered IR
--lower-affine, --fold-memref-alias-ops, --upmem-dedup-kernels
```

### Memory levels are explicit

`#upmem.mram` / `#upmem.wram` already were the memref memory space
everywhere downstream; they now also implement a new
`CinmLevelAttrInterface`, so generic code can ask "which level is this
memref in" without hardcoding UPMEM. Capacity stays on the platform in
`CinmLevelDefAttr` — the level attribute cannot carry a size, because
then `memref<64xi32, #upmem.wram<size=…>>` would be a *different type* on
each platform.

**Yes, `--convert-linalg-to-cnm` has the same `cnm-buffer-level` option**
as `--convert-cinm-to-cnm` — the resolution logic is shared
([CnmBufferLevel.h](../include/cinm-mlir/Conversion/CnmBufferLevel.h),
extracted for exactly this). Empty means unlevelled buffers, i.e. the
backend picks the staging itself. `--convert-linalg-to-cnm` also takes
`allow-float-reassociation` and `per-dim-attrs`.

Consequences down the stack: a launch body computes on MRAM memrefs, and
`--upmem-tile-mram-buffers` tiles that body and promotes its operands
into WRAM with `cnm.local_transfer` around them. Promotion runs in two
stages so the output tile is staged *outside* the reduction loop and
written back once, as the hand-written templates do.

The launch body stays **linalg**, not `cinm` ops. That was tried and
reverted: selecting a level changes where buffers live, not what the body
is, and the body is very often a dot product that no `cinm` op can
express. Keying the staging off the memory space instead means it works
for any linalg op, and `--convert-cinm-to-cnm` goes back to deciding
nothing about the body.

### Distribution is mechanical

`--convert-linalg-to-cnm` replaces the case analysis. Its parameter is
`cnm.tile_sizes`: one **block size** per iteration dimension of the op.
Everything else follows:

- **check** `∏(E_i / b_i) == |WG|`;
- **scatter map** = `indexingMap ∘ (workgroup → tile)` — computed, not
  guessed. Contiguity is somebody else's job
  (`--cnm-ensure-scatter-gather-contiguous`);
- **per-leaf buffer shape** = the block sizes of the dims in that
  operand's own indexing map. A dropped dim *is* broadcast, so that case
  needs no special handling. (Repeating a dim is rejected: maps must be
  projected permutations.)

Starting from linalg rather than cinm is what buys this — indexing maps
already say what the case analysis was trying to reconstruct — and it is
also what makes fusion available later, essentially for free.

**Reduction splitting**, the one structural addition, comes from
`linalg::splitReduction`: it turns the reduction tile index into an extra
*parallel* dimension, so distribution applies unchanged, and it does the
identity seeding and folds the original `outs` in exactly once at the
host-side merge. Associative and commutative combiner required; floats
behind `allow-float-reassociation`, since splitting them reassociates.

Two things are fixed **by rule** rather than searched, both because they
are permutations rather than integer factors and a search space cannot
hold them without a first-class representation and distance function:

- **§G3, tile-dim → workgroup-axis order**: split reduction dims
  outermost, then the original parallel dims, then the unsplit reduction
  remainder. This is not cosmetic — which tile dim varies fastest decides
  which operands a DPU's tasklets *share*, and `--convert-cnm-to-upmem`
  reads that syntactically off the scatter map. The current rule
  reproduces the autotuner's grouping.
- **§I, trips**: when the tile counts exceed the workgroup, the tile
  space is linearized in the §G3 order, low-order digits indexing the
  workgroup and high-order digits a host trip loop. **[todo]**

### The search space is derived, not written

One space, built by walking the linalg ops: one block size per iteration
dimension per level (level 0 = what a workgroup leaf gets, level 1 = what
it walks that in at the leaf memory level), constrained by
`∏(E_i/b_i) == dpus*tasklets` and leaf-divides-block. No
`dpuRows`/`dpuCols`/`taskletRows`/`taskletCols` — those are a *reading* of
these numbers, not parameters. It generalizes to any iteration rank.

Variables are named `<op>.<dim><level>` — `gemv.M0, gemv.K0, gemv.M1,
gemv.K1` — with dimension names taken from the originating `cinm` op via
`cinm.lowered_from`, and op kinds counted first so two gemvs get
`gemv0`/`gemv1`. `eval-solution` resolves by name, not position; the old
positional encoding was an ABI for every params dict in `experiments/`
and had already been a hazard twice.

**Capacity is deliberately not modelled a priori.** Occupancy is a
property of the *lowered* program: operand sharing, `useFullTileBuffers`,
buffer hoisting, kernel dedup — and even the affine simplifier, since
`ae698cc` changed nothing but that and moved per-DPU MRAM from 9280 to
8384 elements. So the space carries only a bound that can never
over-estimate (assume maximal sharing), and the exact check reads the
real `upmem.static_alloc` sizes off the lowered IR. **[todo]**

### Where the templates fit

`SimulationTemplates.cpp` stays for now as the quality bar. Both
lowerings are selectable (`lowering=templates|generic`) against the *same*
space — the templates read its numbers back through a projection (§H5) —
so one configuration can be costed both ways. That comparison is the
criterion for deleting them.

## 3. Bug fixes

Latent ones first; several were invisible until this work reached them.

- **`cnm.launch` did not round-trip levelled buffers.** The shorthand
  buffer type printed shape and element type only, while the parser
  accepted an optional level. Any `cinm-opt | cinm-opt` on levelled IR
  failed. (`0f5dede`)
- **Tiled reductions dropped every trip but the last.** With a shaped
  result, `ReduceTilingModel` overwrote the accumulator slice instead of
  combining into it, and seeded it with `tensor.empty()` rather than the
  identity. Unreachable until something tiled a reduction dimension into
  more than one trip. (`3caea2d`)
- **Reduce scattered a zero init for every method**, so a `mul` reduction
  always returned zero. Now the method's identity. (`71cd9c1`)
- **The launch body's reduction dimension was hardcoded to 0** — correct
  only when the buffer holds nothing but the reduction.
- **`--convert-cinm-ops-to-linalg` dropped discardable attributes**, and
  **`linalg::splitReduction` builds a fresh op**, so the leaf tile sizes
  vanished *exactly* for the configurations that split a reduction:
  slower code, no diagnostic. (`1d1d094`, `190678e`)
- **`simplifyAffineExprWithBounds` could not fold a `floordiv` of a sum.**
  Without the rule the delinearized scatter index never simplified to
  something tasklet-independent, `isMramBroadcastOverThreads` said no, and
  the shared operand was replicated per tasklet. Load-bearing: it is what
  makes §G3's ordering observable at all, and worth 9280 → 8384 elements
  of per-DPU MRAM on gemv_64MB. (`ae698cc`)
- **The UPMEM C translator computed subview offsets from the subview's
  own sizes instead of the source's strides.** For
  `subview %buf[%t,0,0,%k] [1,8,1,64]` of `memref<8x8x1x128xi32>` it
  emitted `%k*64 + %t` where `%t*1024 + %k` was meant — silently wrong
  MRAM addresses. It now uses `getStridesAndOffset` and **rejects nested
  subviews loudly**; the pipeline runs `--fold-memref-alias-ops` (not
  `--canonicalize`, which does not do this) to compose them first.
  (`b6cbc0b`, `a12f33c`)
- **`affine.for` survived into DPU kernels** on the generic path.
  (`85371ca`)
- **`isZeroSplatFoldable` could not see through `linalg.fill`**, so the
  identity seed defeated the constant-scatter fold. (`eececea`)
- **A stale test asserted gating that never existed** — `cinm1-codegen`
  was documented in a comment as disabling the `upmem.broadcast`
  shortcut, which is gated on `use-bc-xfer-codegen` alone. The comment was
  wrong, not the code. (`d6b8fca`)

## 4. What is left

### 4.1 Technical debt to remove

| What | When | Blocked on |
|---|---|---|
| `cnm-buffer-level` on `--convert-cinm-to-cnm`, and `cinm-to-cnm-launch-body.mlir` / `gemv-generic-mram-pipeline.mlir` | **now** | nothing — see below |
| The templates path: `SimulationTemplates.cpp`, the `registerSimulator` bypass, `readAsGemvTemplate` and the §H5 projection, the `lowering=` option | **before the artifact** | parity on a cycle-accurate simulator, which needs 4.2's repack and broadcast items |
| `--convert-cinm-to-cnm` itself, `computeShapeOfTensors`' case analysis, and `--cinm-infer-tile-sizes`/`--cinm-tiling` on the UPMEM path | **after the artifact** | it is the CINM 1.0 baseline's lowering *and* the only cinm→cnm route for the GPU backend; needs the baseline retired or GPU moved to `--convert-linalg-to-cnm` |
| `cinm.op.reduce`'s memref/DPS mode (M3b) | opportunistic | added for the withdrawn M4; only consumer left is its own tiling model |
| `CinmLevelDefAttr`'s dead `arity`; the `getDesignParams`/`instantiateDesignParams` stubs; `CostModel.cpp` | opportunistic | — |

**On the first row — no, `--convert-cinm-to-cnm` does not need to keep
handling MRAM.** Its `cnm-buffer-level` option (milestone M1) has no live
caller: the generic path goes through `--convert-linalg-to-cnm`, and the
CINM 1.0 baseline and the GPU path both invoke `--convert-cinm-to-cnm`
with no flag, which is byte-identical to the pre-refactoring behaviour.
The option is exercised only by its own tests. It was a necessary
stepping stone — M1 through M7 built the level machinery on the old pass
before `--convert-linalg-to-cnm` existed — but it is dead now and can be
deleted well before the pass around it. What must stay is the shared
`CnmBufferLevel.h` helper and everything level-related in
`--convert-cnm-to-upmem`, which the generic path depends on.

### 4.2 Optimizations to implement

Roughly in value order; the first two are the whole measured gap.

- **Stop materializing the tiled layout on the host** (M15). `cnm.scatter`
  requires the host value's shape to *end with* the buffer shape, so
  `toTiledLayout` emits a real permuted copy for any operand tiled in ≥2
  dims — on gemv_64MB a 64 MB copy of `A`, **77.9 ms of the generic path's
  103.4 ms**. Fix: give `cnm.scatter` a general affine map into the host
  buffer (the real answer), or emit a strided view and let
  `--cnm-ensure-scatter-gather-contiguous` decide where packing is really
  needed (cheap to try, and it measures how much is necessary).
- **Broadcast detection** — a scatter map that does not depend on the PE
  dimension should specialize into a broadcast. Concretely: the template
  path uses a broadcast transfer for the gemv vector and the generic path
  does not, so it pays a full scatter for the same data.
- **Sequential trips** (M13, §I). Decided, not implemented. Needs
  `expand_shape` where the trip boundary falls *inside* a dimension, and
  an `scf.for` iter_arg where trips run over a reduction dimension.
  Blocks the `cinm2` baselines, commented out in
  [dodo.py](../experiments/gemv_microbenchmark/dodo.py) until it lands.
- **Exact post-lowering occupancy check** (M14, §H4). Until it exists,
  infeasible configurations are discovered only when the DPU binary fails
  to link.
- **Constant-scatter simplification** — the general form of the zero-seed
  problem: the identity seed is a constant buffer scattered to every leaf
  on every launch. `cnm.set_zero` is the right answer but only
  `--convert-cnm-to-gpu` lowers it, and emitting it makes
  `--convert-cnm-to-upmem` fail *silently* — so this also owes that pass a
  diagnostic for ops it cannot legalise.
- **Linalg fusion** (§G8). A no-op for today's single-op benchmarks, and
  enabling it forces search-space construction after the conversion
  (fusion changes the op count, the walk indices *and* the iteration
  space). The pipeline is arranged so it slots in right after the linalg
  conversion. It also weakens the same-configuration comparison, since the
  two paths would no longer share an op set.
- **On-device transfer coalescing** (§C) — merge sibling per-tasklet
  `cnm.local_transfer`s into one leader transfer plus a barrier. **CNM has
  no barrier primitive**; this and the row-leader merge already
  hand-written in the reduce template both need one.
- **Share the WRAM staging buffer across tasklets** when the scatter map
  is tasklet-independent (§A3); always private today.
- **Device-side tree reduction of partials** (§G10) — host merge only for
  now. Same parameters, but it is the trigger for revisiting §G3.
- **Blocked vs. multi-block transfer API selection**, decided generically
  from the scatter map rather than by pass flags.
- Smaller: `--upmem-tile-mram-buffers` promotes with
  `useFullTileBuffers=false`, so a tile that does not divide its extent
  gets a staging buffer larger than the tile.

### 4.3 Design questions still open

- **§G3 (and §I) as real search parameters.** Both are fixed rules
  standing in for a permutation-valued parameter, which needs a
  first-class representation with a sensible distance function (cf. BACO).
  Worth revisiting together, once either is shown to cost measurable
  performance.
- **A cycle-accurate cost model.** `op-count` models neither loads nor
  stores and does not report time, so it cannot answer "is the generic
  path as good as the templates" — the question that gates 4.1's second
  row.
- **Merge or keep `UpmemGenericLoweringNotes.md`** (§E): is the direct
  `linalg.generic → upmem` path a permanent parallel option, or superseded?

## 5. Where things stand

`gemv_64MB` on hardware, one configuration through both lowerings:

| | net | scatter | gather | launch | unaccounted |
|---|---|---|---|---|---|
| `atim_templateflow` | 10.782 | 9.428 | 0.555 | 0.434 | 0.365 |
| `atim_genericflow` | 103.433 | 24.557 | 0.609 | 0.390 | 77.877 |

All in ms. **Device time already matches** (launch 0.390 vs 0.434) — the
DPU program the generic path emits is as good as the template's. The
entire gap is host-side data movement: the repack and the missing
broadcast, the first two items of §4.2.

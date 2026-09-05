# The CINM 2.0 lowering pipeline: an overview

This document describes how cinm-mlir compiles a high-level tensor
operation down to code for the UPMEM processing-in-memory architecture:
the pipeline as it was, the pipeline as it is being rebuilt, why, and what
remains to be done. It is meant to be readable on its own. Detailed
argument and derivations live in
[CnmMemoryLevelsDesign.md](CnmMemoryLevelsDesign.md), and the
milestone-by-milestone record in
[CnmMemoryLevelsImplementationPlan.md](CnmMemoryLevelsImplementationPlan.md);
neither is required reading for what follows.

## 0. Background

### The dialects

cinm-mlir lowers through three of its own MLIR dialects:

- **`cinm`** — the input language. Whole-tensor operations
  (`cinm.op.gemv`, `cinm.op.gemm`, `cinm.op.reduce`, …) inside a
  `cinm.compute` region that names the accelerator it should run on.
  Nothing here is device-specific.
- **`cnm`** — "compute near memory", a device-independent model of
  *distributed* computation. Its vocabulary is a **workgroup** (a
  multi-dimensional grid of processing elements), a **buffer**
  (`cnm.buffer`: one tile of data per workgroup element), **scatter** and
  **gather** (move data between a host tensor and a buffer, addressed by
  an affine **scatter map** from workgroup coordinates into the host
  tensor), and **launch** (run a region once per workgroup element, on its
  own tiles). We call an individual workgroup element a **leaf**.
- **`upmem`** — the backend dialect. DPU allocation, kernels, DMA
  transfers, and the memory spaces `#upmem.mram` / `#upmem.wram`. It is
  translated to C and compiled by the UPMEM SDK.

### The hardware, in the four terms this document uses

A UPMEM system is a set of **ranks**; each rank holds ~64 **DPUs**
(simple in-order cores); each DPU runs up to 16 or 24 hardware threads
called **tasklets**. Each DPU has two memories:

- **MRAM** — 64 MiB of DRAM per DPU. Large, slow, reached only by
  explicit DMA. This is where the host scatters data to.
- **WRAM** — 64 KiB of SRAM per DPU (minus ~8 KiB reserved for runtime
  structures). Small, fast, directly addressable. Compute happens here,
  so a kernel must stage its working set MRAM → WRAM and back.

The workgroup shape used for UPMEM is therefore `ranks × DPUs ×
tasklets`, and a **leaf is one tasklet**. Note the asymmetry that causes
most of the subtlety below: MRAM and WRAM are physically per-*DPU*, so
the tasklets of a DPU may share a buffer or may each need their own,
and which one it is depends on how the data was distributed.

### Design-space exploration

Tile sizes and distribution choices are not fixed by the compiler; they
are searched. The `--upmem-infer-accelerator` pass hosts an **inference
plugin** that (1) builds a **search space** of integer parameters with
constraints, (2) asks a Bayesian optimizer for a configuration,
(3) lowers the program with that configuration and costs the result with
a simulator, and (4) commits the best one. A configuration can also be
pinned by hand with the `eval-solution` pass option, which is how the
tests and benchmarks work.

Historically the plugin had a second, faster route: **templates**, in
`SimulationTemplates.cpp` — hand-written generators that emit UPMEM code
for `gemv` and `reduce` directly, bypassing the real pipeline. They
existed because the real pipeline could not express the configurations the
templates could. That is no longer true: the route, the `lowering=` option
that selected it and the file itself are gone (§4.1).

### The thesis of the refactoring

> Passes expose **parameters**; they do not take decisions.

Every distribution decision the old pipeline made by case analysis on
shapes becomes a number supplied by the search, and the search derives
those numbers from the operation's own iteration space rather than from
per-operation special-casing.

## 1. The old pipeline (CINM 1.0), and who decided what

Still live: it is the comparison baseline in the experiments, transcribed
into [cinm1.py](../experiments/cinm_experiments/cinm1.py) from
`testbench/Makefile`.

| Stage | Pass | Decisions it takes |
|---|---|---|
| 1a | `--cinm-infer-tile-sizes` | **Tile sizes**, from the accelerator's `computeTilingFactors` — a fixed per-op heuristic, no search. |
| 1a | `--cinm-tiling` | Emits the host-side loop nest over tiles, through `CinmTilingInterface`, implemented once per op kind. |
| 2 | `--convert-cinm-to-cnm` | **Everything about the distribution**: how a tile's elements map onto leaves, whether an operand is broadcast / chunked / transposed, the per-leaf buffer shape, and what the launch body computes. |
| 3–5 | bufferization, hoisting, `--cnm-ensure-scatter-gather-contiguous` | Mechanical. |
| 6 | `--convert-cnm-to-upmem` | Allocates MRAM and stages MRAM→WRAM around *every* launch, identically for every program. Which transfer API each operand got was chosen here, by case analysis on the scatter map gated on the pass flags `use-sg-xfer-codegen` / `use-bc-xfer-codegen`; that choice has since moved to `--upmem-specialize-transfers`, and the flags with it. |

The weight is all in stage 2. Its core routine, `computeShapeOfTensors`,
is roughly 200 lines of case analysis — "this tensor has exactly one
parallel element, so broadcast it"; "its parallel elements equal the
workgroup size but the chunks are not contiguous, so insert a
`linalg.transpose`, which the caller has to do"; "flatten trailing
parallel dimensions until the product divides `k`, then maybe emit a
reshape". Each case *guesses* a distribution from the operand shapes.

Three consequences shaped everything that follows.

**The decisions are not addressable.** The search can move tile sizes and
nothing else. Any configuration that differs in *how* work is spread over
the workgroup is unreachable, which is precisely why the hand-written
templates exist.

**Reduction splitting is inexpressible.** Spreading a reduction dimension
across the workgroup — each leaf computing a partial result, merged
afterwards — has no representation at all. Configurations needing it are
simply out of reach. The failure was concrete: the best known `gemv`
configuration from an independent autotuner made
`--convert-cinm-to-cnm` fail with `numParallelElts (64) % numWgItems
(16384) != 0`.

**Memory levels are implicit.** `cnm.buffer` had a `level` field, but
nothing ever set it and nothing read it. `--convert-cnm-to-upmem` staged
MRAM→WRAM the same way for every program, so on-device staging could not
be tiled, hoisted, or shared between tasklets.

## 2. The new pipeline (CINM 2.0)

The full picture, including parts not yet implemented — those are marked
**[todo]** and collected with priorities in §4.2.

```
--convert-cinm-ops-to-linalg      cinm op -> linalg op, carrying its
                                  attributes and a cinm.lowered_from marker
  [todo] fusion                   producer/consumer fusion on linalg
--convert-linalg-to-cnm           distribute onto the workgroup
    cnm-buffer-level=mram           -> buffers live in MRAM
    per-dim-attrs=...               -> attributes that must survive a split
--cnm-scatter-optimizations       shrink a uniform scatter to one tile, or
                                  drop it for a fill in the launch body
bufferization / hoisting / CSE
--upmem-tile-mram-buffers         tile the launch body and promote its
                                  operands MRAM -> WRAM, staging them
                                  with cnm.local_transfer; a filled operand
                                  is written in place, not staged
--convert-linalg-to-affine-loops, --affine-scalrep
--cnm-ensure-scatter-gather-contiguous
                                  turn each leaf's tile into whole blocks,
                                  splitting a host dimension where a
                                  contiguous run does not line up; pack
                                  only what is left over
--convert-cnm-to-upmem            level-aware: MRAM buffers bind straight
                                  to the static allocation; emits only the
                                  general block transfer form
--upmem-specialize-transfers      blocks -> flat -> broadcast, where legal
--lower-affine, --fold-memref-alias-ops, --upmem-dedup-kernels
--upmem-check-occupancy           last: reject the program if its buffers
                                  do not fit a DPU -- measured, not modelled
```

### 2.1 Memory levels are explicit

`#upmem.mram` and `#upmem.wram` were already the memref memory space
throughout the backend. They now also implement a new
`CinmLevelAttrInterface`, so target-independent code can ask "which
memory level does this memref live in" without hardcoding UPMEM.

Capacity deliberately does *not* live on that attribute. It stays on the
platform, in `CinmLevelDefAttr`, because a memory space is part of a
memref's type: parameterizing it with a size would make
`memref<64xi32, #upmem.wram<size=57344>>` and `…<size=55296>` *different
types* on hardware revisions that differ only in how much WRAM they have.
So the level attribute carries identity, the platform carries size and
alignment, and `platform.getLevel(name)` connects them.

Two conversions take a **`cnm-buffer-level` option** naming the level to
allocate `cnm.buffer`s in: `--convert-cinm-to-cnm` (the old path) and
`--convert-linalg-to-cnm` (the new one). They share one resolution helper,
[CnmBufferLevel.h](../include/cinm-mlir/Conversion/CnmBufferLevel.h), so
the two cannot drift. An empty value means unlevelled buffers, which is
the historical behaviour where the backend picks the staging itself.
`--convert-linalg-to-cnm` additionally takes `allow-float-reassociation`
and `per-dim-attrs`, both explained below.

With `cnm-buffer-level=mram`, a launch body computes on MRAM memrefs, and
a new pass **`--upmem-tile-mram-buffers`** tiles that body and promotes
its operands into WRAM, wrapping them in `cnm.local_transfer` (a new
memref-to-memref `cnm` op whose direction is implicit in its operands'
memory spaces). Promotion runs in two stages, so the *output* tile is
staged outside the reduction loop and written back once rather than on
every trip — matching what the hand-written templates do by construction.

One deliberate non-choice: **launch bodies stay `linalg`**. Putting
memref-mode `cinm` ops in them was implemented and then reverted.
Selecting a memory level changes where buffers live, not what the body
computes, and the per-leaf body is very often a dot product that no
`cinm` op can express. Keying the staging off the memory space instead
means it works for any `linalg` op, and it lets the distribution pass go
back to deciding nothing at all about the body.

### 2.2 Distribution is mechanical

**`--convert-linalg-to-cnm`** replaces the case analysis. It distributes a
`linalg` op on tensors onto a workgroup, driven by one attribute:
`cnm.tile_sizes`, giving one **block size** per iteration dimension —
the extent of that dimension inside a single leaf. Everything else is
derived:

- **legality check**: `∏(E_i / b_i) == |WG|`, i.e. the tile counts must
  fill the workgroup exactly (`E_i` = dimension extent, `b_i` = block
  size). Failure reports both numbers.
- **scatter map** = `indexingMap ∘ (workgroup → tile)`. Computed by
  composition, not guessed. It is *pointwise*: it sends each element of
  each leaf's buffer to the host element it comes from, and says nothing
  about how the transfer is shaped. Turning that into contiguous blocks
  is a later pass's job, `--cnm-ensure-scatter-gather-contiguous`.
- **per-leaf buffer shape** = the block sizes of the dimensions appearing
  in that operand's own indexing map. An operand that *drops* a dimension
  is a broadcast, and needs no special case to be one.
- **launch body** = the same `linalg` op on leaf-shaped memrefs.

Starting from `linalg` rather than `cinm` is what buys this: an indexing
map already states what the old case analysis was trying to reconstruct
from shapes. (One restriction follows: indexing maps must be projected
permutations. Dropping a dimension is fine; repeating one is rejected.)
It also makes operator fusion available essentially for free later.

**Reduction splitting** is the one structural addition. Giving a
reduction dimension a block size smaller than its extent spreads it
across the workgroup: each leaf computes a partial result and a merge is
left on the host. The rewrite is `linalg::splitReduction` from upstream,
which turns the reduction tile index into an extra *parallel* dimension —
so the distribution above applies unchanged — and which also seeds every
leaf with the combiner's identity and folds the original output in
exactly once at the merge. Legality: the combiner must be associative and
commutative, and floating-point reductions require the explicit
`allow-float-reassociation` flag, since splitting them reassociates the
arithmetic and changes results.

Two things are decided **by rule** rather than derived. Both are
permutations rather than integer factors, and a search space cannot hold
a permutation without a first-class representation and a distance
function over it — so where one of them *is* searched, it is searched by
rank rather than as a permutation:

- **Tile-dimension → workgroup-axis order.** The rule: split
  reduction-derived dimensions outermost, then the original parallel
  dimensions, then any unsplit reduction remainder. This is not cosmetic.
  Which tile dimension varies fastest determines which operands the
  tasklets of one DPU *share* versus replicate, and
  `--convert-cnm-to-upmem` reads that off the scatter map syntactically.
  The rule reproduces the grouping the independent autotuner found best,
  and it is now the *default* rather than the only option: the order can
  be stated outright (`cnm.workgroup_dim_order = array<i64: 1, 0, 2>`) or
  by lexicographic rank (`cnm.workgroup_dim_order_index = N`, index 0
  being the rule), as an attribute per op or as a pass option for all of
  them. The rank counts only the dimensions actually spread over the
  workgroup, so it enumerates the distinct orders and nothing else —
  which is what makes it a space variable, `<op>.order`, at a scale
  (`k ≤ 3`) where enumerating it outright is the whole story.
- **Sequential trips.** When the tile counts exceed the workgroup size,
  the tile space is linearized in the order above; the low-order digits
  index the workgroup and the high-order digits index a host-side trip
  loop. **[todo]**

### 2.3 The search space is derived, not written

The plugin builds one space by walking the `linalg` ops of the compute
block. Per operation it declares one block size per iteration dimension
per memory level — **level 0** being what a leaf gets (its MRAM tile),
**level 1** what the leaf walks that in at the fast level (its WRAM tile)
— constrained by `∏(E_i / b_i) == dpus × tasklets` and by each level-1
block dividing its level-0 block. Plus, for ops with more than one
iteration dimension, the workgroup-axis order above as a single rank.

Notably absent: `dpuRows`, `dpuCols`, `taskletRows`, `taskletCols`. Those
are a *reading* of the block sizes, not independent parameters. Dropping
them is what lets the space generalize to any iteration rank instead of
being written out per operation.

Variables are named `<op>.<dim><level>`, so a `gemv` declares `gemv.M0`,
`gemv.K0`, `gemv.M1`, `gemv.K1`, and `gemv.order`. Dimension names come
from the originating
`cinm` op — M/K for gemv, M/N/K for gemm, and so on — recovered through a
`cinm.lowered_from` marker attribute, falling back to `D0, D1, …` for ops
with no established convention. Operation kinds are counted first, so a
block containing two gemvs gets `gemv0`/`gemv1` while the ordinary
single-op case stays unadorned. `eval-solution` resolves parameters by
name; it used to be positional, which made the variable list an implicit
ABI for every configuration recorded under `experiments/`.

**Memory capacity is deliberately not modelled in the space.** Occupancy
is a property of the *lowered* program, not of the configuration: it moves
with operand sharing, with whether promotion allocates a full-size
staging buffer, with buffer hoisting, with kernel deduplication — and even
with the affine simplifier, since one commit that changed nothing but that
moved per-DPU MRAM for a fixed configuration from 9280 to 8384 elements.
Any a-priori bound is therefore either too permissive (and the
configuration is discovered infeasible only when the DPU binary fails to
link) or too strict (and it silently deletes good configurations).

So the search space carries only a bound that can never over-estimate — it
assumes maximal sharing — and feasibility is decided by *measuring* the
lowered program. `--upmem-check-occupancy` runs last in the pipeline, after
every memory optimization, and sums what each DPU program actually
allocates: static MRAM and WRAM once per DPU, private WRAM once per tasklet
plus the runtime's stack reserve. A configuration that does not fit fails
its evaluation and the search moves on.

The two halves are deliberately far apart in accuracy. On the 64 MB gemv,
a leaf tile of 8×256 charges the cheap bound 2312 elements against the
14336 a DPU has, and the real program 82176 bytes against 57344 — because
the tasklets end up replicating the staging buffers rather than sharing
them, which is a fact about the lowering, not about the configuration.

### 2.4 Transfers are emitted general and narrowed afterwards

UPMEM offers three ways to get a host buffer onto the DPUs, and they are
the same capability at three levels of specificity: a **blocked** transfer
(`dpu_push_sg_xfer`, any number of blocks per DPU, gathered from anywhere
in the host buffer), a **flat** one (one contiguous block per DPU), and a
**broadcast** (the same block to every DPU). The dialect has an op per
level — `scatter_blocks` / `scatter_on_array` / `broadcast`, and the two
gather counterparts — and `transferCount` means "one block" in all of
them, with a DPU's blocks landing back to back in MRAM.

`--convert-cnm-to-upmem` emits **only the general form**, reading the
block count off the scatter map, and **`--upmem-specialize-transfers`**
narrows it: adjacent in-order blocks collapse to a flat transfer, and a
host buffer whose elements are all the same constant becomes a broadcast
of a constant the size of the target MRAM buffer — widening the host
constant is free (a splat attribute stores one element) and is what lets
the reduction identity seed reach the DPUs in one call. Gathers never
narrow to a broadcast: two DPUs writing one host region is a race.

The point of the split is that the emitter no longer decides. It used to
pick the API by case analysis on the map, and getting that wrong was
silent — with `use-sg-xfer-codegen=false` a map asking for host rows 0 and
2 was collapsed into a flat transfer that read rows 0 and 1. Narrowing is
now one place, guarded by one contiguity contract that all five ops'
verifiers share, and a block form that *cannot* be narrowed with the
scatter/gather API disabled is a diagnostic rather than a wrong transfer.

It is a pass rather than a canonicalization only because it has to be
disable-able while the paper's baselines are being measured; §4.1 has the
cleanup.

### 2.5 Where the templates fit

They no longer do. While both lowerings were selectable — `lowering=templates
|generic` — they read the *same* search space, the templates through a
projection:

    taskletRows = min(tasklets, mTiles)   taskletCols = tasklets / taskletRows
    mramRow     = blockM * taskletRows    mramCol     = blockK * taskletCols
    wramRow     = leafM                   wramCol     = leafK

so one configuration could be lowered and costed both ways, which is the
comparison §5 reports. The plugin now has one path: it lowers the reference
to linalg once, when the space is built, and every trial runs the pipeline
from there.

## 3. Bugs found and fixed

Several of these were latent — real defects that nothing had yet
exercised. They are worth recording because most were silent.

- **`cnm.launch` did not round-trip buffers carrying a memory level.**
  The shorthand buffer type printed shape and element type only, while
  the parser accepted an optional level, so `cinm-opt | cinm-opt` on
  levelled IR failed. (`0f5dede`)
- **Tiled reductions dropped every trip but the last.** With a shaped
  result, the tiling model overwrote the accumulator slice instead of
  combining into it, and seeded it with `tensor.empty()` rather than the
  reduction's identity. Unreachable until something tiled a reduction
  dimension into more than one trip. (`3caea2d`)
- **Reduce scattered a zero initializer for every method**, so a `mul`
  reduction always returned zero. It now scatters the method's identity.
  (`71cd9c1`)
- **The launch body's reduction dimension was hardcoded to 0**, which is
  correct only when the per-leaf buffer holds nothing but the reduction.
- **Two attribute-loss bugs.** `--convert-cinm-ops-to-linalg` dropped
  discardable attributes, and `linalg::splitReduction` builds a fresh op,
  so the level-1 (WRAM) tile sizes vanished *exactly* for configurations
  that split a reduction: slower code, no diagnostic. The
  `per-dim-attrs` option exists to carry such attributes across a split.
  (`1d1d094`, `190678e`)
- **The affine simplifier could not fold a `floordiv` of a sum.** Without
  that rule, a delinearized scatter index never simplified to something
  provably independent of the tasklet coordinate, so the check for "can
  the tasklets share this buffer" said no and the operand was replicated
  per tasklet. Load-bearing: it is what makes the ordering rule of §2.2
  observable at all, and worth 9280 → 8384 elements of per-DPU MRAM on
  the 64 MB gemv. (`ae698cc`)
- **The UPMEM C translator computed subview offsets from the subview's
  own sizes instead of the source memref's strides.** For
  `subview %buf[%t,0,0,%k] [1,8,1,64]` of `memref<8x8x1x128xi32>` it
  emitted `%k*64 + %t` where `%t*1024 + %k` was meant — silently wrong
  MRAM addresses. It now uses `getStridesAndOffset`, and **rejects nested
  subviews loudly** rather than mis-addressing them; the pipeline runs
  `--fold-memref-alias-ops` (note: `--canonicalize` does *not* do this) to
  compose chains into one first. (`b6cbc0b`, `a12f33c`)
- **`affine.for` survived into DPU kernels**, which the C translator
  cannot consume. (`85371ca`)
- **The zero-splat fold could not see through `linalg.fill`**, so the
  reduction identity seed defeated it. (`eececea`)
- **`use-sg-xfer-codegen=false` moved the wrong bytes.** It was read as
  "collapse this transfer to one flat block per DPU" rather than "this
  transfer must already be one flat block", so a map addressing host rows
  0 and 2 produced a transfer of rows 0 and 1 — no diagnostic, wrong
  results. The narrowing is now conditional and the flag only says whether
  the blocked API may be used; a transfer that needs it while it is off is
  an error naming the operand. Worth noting that the aligned contiguity
  verifier could not have caught this: the collapsed transfer is perfectly
  in bounds and contiguous, and the block dimension was thrown away before
  the op it would have checked ever existed. (`1d7e448`)
- **A test asserted gating that never existed.** A code comment claimed
  the `cinm1-codegen` option disabled the broadcast-transfer shortcut,
  which is in fact gated on `use-bc-xfer-codegen` alone. The comment was
  wrong, not the code. (`d6b8fca`)

## 4. What remains

### 4.1 Code to remove

| What | When | Blocked on |
|---|---|---|
| The `cnm-buffer-level` option on `--convert-cinm-to-cnm`, and the two tests exercising it | **now** | nothing — see below |
| ~~The templates path: `SimulationTemplates.cpp`, the simulator bypass, the projection of §2.5, the `lowering=` option~~ — **done**, along with the analytical per-shape cost estimators that only it called (`UpmemSimulator::simulateGemv`/`simulateReduction` and the two whole-program wrappers) | — | — |
| `--convert-cinm-to-cnm` itself, its `computeShapeOfTensors` case analysis, and `--cinm-infer-tile-sizes` / `--cinm-tiling` on the UPMEM path | **after the artifact** | it is the CINM 1.0 baseline's lowering *and* the only `cinm`→`cnm` route for the GPU backend; needs the baseline retired or the GPU backend moved to `--convert-linalg-to-cnm` |
| `cinm.op.reduce`'s memref (destination-passing) mode | opportunistic | built for a milestone that was withdrawn; its only remaining consumer is its own tiling model, though it does fill a real gap for memref-mode input programs |
| `--upmem-specialize-transfers` as a *pass*: the two narrowings become `hasCanonicalizer = 1` on the two `_blocks` ops, and `use-sg-xfer-codegen` / `use-bc-xfer-codegen` go with it | **after the artifact** | the flags exist only so the CINM 1.0 baseline can be measured with each transfer API in turn; nothing else reads them |
| `CinmLevelDefAttr`'s unused `arity` parameter; the unexplained `getDesignParams`/`instantiateDesignParams` stubs; `CostModel.cpp`, superseded by the Bayesian search | opportunistic | — |

**On the first row: `--convert-cinm-to-cnm` does not need to keep handling
MRAM.** Its `cnm-buffer-level` option has no live caller. The new pipeline
distributes through `--convert-linalg-to-cnm`; the CINM 1.0 baseline and
the GPU path both invoke `--convert-cinm-to-cnm` with no level, which is
byte-identical to its pre-refactoring behaviour. The option is exercised
only by its own tests. It was a necessary stepping stone — the memory-level
machinery was built on the old pass before `--convert-linalg-to-cnm`
existed — but it is dead now and can go well before the pass around it.
What must stay is the shared `CnmBufferLevel.h` helper and all the
level-awareness in `--convert-cnm-to-upmem`, on which the new pipeline
depends.

### 4.2 Optimizations to implement

Roughly in value order. Three of the items that used to head this list have
landed. **Pointwise scatter/gather maps**, which accounted for most of the
measured gap against the templates: the operand is now scattered where it
lies instead of being permuted into a tiles-outermost copy, and the
transfer's block structure is derived after bufferization rather than
assumed by the dialect. **Broadcast detection and transfer-API selection**
(§2.4): a scatter map that does not depend on the processing-element
coordinate now narrows to a broadcast, the gemv vector included, and which
API each transfer uses follows from the map instead of from a pass flag.
**Constant scatter**: a uniform value is not transferred at all. Each leaf
writes its own elements instead — in parallel, with no host bandwidth, and
without the MRAM round trip, since a tasklet initializes its WRAM tile
directly rather than loading MRAM it is about to overwrite. That is the
reduction identity seed gone entirely: on the 64 MB gemv the kernel now
opens by zeroing eight WRAM words, and `@buf` is read by nothing.

- **Sequential trips.** Decided but unimplemented (§2.2). The trip
  boundary does not always fall between two dimensions — it can fall
  inside one — and trips over a reduction dimension need a loop-carried
  accumulator. This blocks two baseline configurations, currently
  commented out in
  [dodo.py](../experiments/gemv_microbenchmark/dodo.py). Now that the
  scatter map is pointwise, a trip is an offset on the host operand rather
  than a reshape and a permutation.
- **Operator fusion on `linalg`.** A no-op for today's single-operation
  benchmarks, and enabling it forces search-space construction to move
  after the conversion, since fusion changes the operation count, the walk
  indices, and the iteration space. The pipeline is arranged so it slots
  in right after the `linalg` conversion. It also weakens the
  same-configuration comparison of §2.5, because the two paths would no
  longer share an operation set.
- **On-device transfer coalescing.** Transfers performed concurrently by
  sibling tasklets can be merged into one blocked transfer by a leader
  tasklet. Note that **`cnm` has no barrier primitive** — launch bodies
  are implicitly SPMD with no cross-tasklet coordination modelled — so
  this, and the row-leader merge already hand-written in the reduce
  template, both need one first.
- **Share the WRAM staging buffer across tasklets** when the scatter map
  is tasklet-independent. Always private today.
- **Device-side tree reduction of partial results**, instead of merging
  them on the host. Same parameters, but it is the trigger for revisiting
  the ordering rule of §2.2.
- Smaller: `--upmem-tile-mram-buffers` promotes without full tile
  buffers, so a tile that does not divide its extent gets a staging
  buffer larger than the tile it holds. Correct, just wasteful.

### 4.3 Open design questions

- **Making the two fixed rules of §2.2 searchable.** Both stand in for a
  permutation-valued parameter, which needs a first-class representation
  and a distance function to be searched sensibly (compare BACO's
  approach). Worth revisiting together, once either is shown to cost
  measurable performance.
- **A cycle-accurate cost model.** The cheap `op-count` simulator models
  neither loads nor stores and does not report time, so it cannot answer
  "is the generic pipeline as good as the templates" — the question
  gating the second row of §4.1.
- **The status of the direct `linalg.generic → upmem` path** sketched in
  [UpmemGenericLoweringNotes.md](UpmemGenericLoweringNotes.md): a
  permanent parallel option, or superseded by this pipeline and to be
  merged into it?

## 5. Current state

Test suite: 60 of 67 lit tests pass. The seven failures predate this work
and are unrelated to it (`CimToMemristor` ×2, `TorchToCinm`,
`Transform/Cim` ×2, `upmem-to-c`, `simulate-python`).

Measured on real hardware: a 4096×4096 integer `gemv`, one configuration,
lowered both ways.

| | net | scatter | gather | launch | unaccounted |
|---|---|---|---|---|---|
| templates | 10.782 | 9.428 | 0.555 | 0.434 | 0.365 |
| new pipeline | 103.433 | 24.557 | 0.609 | 0.390 | 77.877 |

All figures in milliseconds, and **the table is stale in both directions
that mattered**: it was measured before the scatter maps became pointwise
— the 77.9 ms of unaccounted time was the 64 MB host repack that change
removes — and before transfers were specialized, which was the other
named cause of the gap. **Device time already matched** — `launch` is
0.390 ms against the template's 0.434 ms, so the DPU program the new
pipeline generates is as good as the hand-written one. Re-measuring on
hardware is the outstanding item; nothing in §4.2 above is expected to
move `launch`.

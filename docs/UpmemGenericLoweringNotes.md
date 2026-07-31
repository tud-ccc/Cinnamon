# Generalizing linalg.generic → UPMEM lowering

Context: `LinalgToUpmem.cpp` currently has two hand-written patterns,
`generateGemv` (1 parallel loop M, 1 reduction loop K) and
`generateTailReduction` (1 reduction loop K only). Both follow the same
shape: tile every loop through the memory hierarchy
host → DPU-grid → MRAM → WRAM(tasklet) → register, where a parallel loop
must supply the DPU-grid trip count (= number of DPUs) and a parallel loop
inside the DPU must supply the tasklet trip count. This note is about
generalizing that to an arbitrary `linalg.generic` with any number of
parallel/reduction loops, before touching implementation.

## 1. Reduction splits are not one mechanism, they're (at least) three

Depending on which tier a reduction loop is split at, the merge strategy is
different, and this is easy to gloss over as "just another tiling level":

- **Host-sequential tier**: the same DPU/MRAM buffer persists across
  successive host loop trips; accumulation happens implicitly via
  re-scatter + `wait_for`, no separate merge op. (E.g. the M-tile's `y`
  buffer in MRAM across `K` outer iterations in both kernels.)
- **DPU-grid tier**: spreading a reduction loop across `dpuCols` produces
  *partial* results per DPU (DPUs can't talk to each other), so the host
  must gather and then explicitly combine — the trailing `linalg.reduce`
  over dim 1 in both `generateGemv` and `generateTailReduction`.
- **Tasklet tier**: if tasklets each hold a private partial, you need a
  barrier + "leader" merge (see the row-leader merge step in
  `createDpuTailReductionKernel`).

So "tile the reduction loop once per memory level" is really "at each tier,
choose spatial-split-with-merge vs sequential-split-with-persistent-
accumulator." These have different costs and different code shapes. A
general lowering needs to make (or expose) that choice per tier, not just
pick tile sizes.

## 2. Assigning loops to hardware roles is combinatorial once you have >1 parallel or >1 reduction loop

Gemv and reduce each had exactly one loop available per role (M → DPU-rows
+ tasklet-rows, K → DPU-cols + tasklet-cols/sequential), so there was no
real choice to make. The moment there's more than one loop of a kind —
e.g. a matmul-shaped generic (2 parallel loops M,N + 1 reduction K) — we
must decide which parallel loop feeds the DPU grid, which feeds the
tasklet grid, and whether the *other* parallel loop gets any hardware
parallelism at all or just becomes extra sequential tiling. The device
hierarchy really only offers two useful axes (dpu-grid, tasklet — `rank`
is pinned to 1 in both examples), so any op with more than 2
"parallelizable" loops (counting reduction loops that get spatially split)
forces some loops to be demoted to purely sequential tiling. That
demotion policy doesn't exist yet.

## 3. Multiple reduction loops compete for the same scarce resources

One DPU-grid axis, one tasklet axis, one barrier-merge budget. If a
generic has two reduction loops, at most one can get the "spatial split +
merge" treatment at a given tier; the other must be sequential there.
Which one wins, and does the answer differ per tier (e.g. reduction A
splits across DPU-cols but reduction B splits across tasklets)? This
needs a policy, likely cost-model-driven rather than purely structural.

## 4. Buffer residency/replication is a per-operand decision, not per-loop

`x` in gemv is broadcast (one WRAM copy shared across all `dpuRows`,
loaded once by tasklet 0 behind a barrier), while `A`/`y` are private per
tasklet. This falls out of which loops each operand's affine indexing map
actually depends on. For an arbitrary `linalg.generic`, we need, per
operand and per tier: does this operand depend on the loop being split
here? If not, that's a broadcast/reuse opportunity (skip re-scatter, skip
re-load, share one WRAM copy across tasklets with a barrier); if so, it
needs its own staged buffer. Currently this is just hand-derived per
kernel.

## 5. Joint capacity fitting at MRAM and WRAM is a multi-operand constraint problem

Tile sizes (`mramRows/mramCols`, `wramRows/wramCols`, tasklet-private vs
shared) must be chosen so all operand buffers alive at a tier fit
simultaneously — and "how many copies" depends on point 4. With more
loops/operands this becomes a real sizing search rather than picking two
numbers by hand.

## 6. Divisibility/remainder handling is currently assumed away

Nowhere does the code handle `M`, `K`, `mramRows`, `tasklets*wramRows`,
etc. not dividing evenly — every loop nest assumes perfect tiling. A
general lowering either needs to require pre-padding, or needs
remainder/epilogue loops at whichever tiers can't guarantee divisibility,
which multiplies code paths per tier.

## 7. Neutral-element / "first iteration" bookkeeping for nested sequential accumulators

At every sequential-split boundary for a reduction, we need to know
whether this is the first pass (zero-init the buffer) or a continuation
(load the outer level's partial and keep accumulating) — visible already
as the `FillOp` + persistent MRAM `y` buffer in the reduce kernel vs. the
`LocalTransferOp` load-before-accumulate in the gemv kernel's `mr` loop.
Generalizing this correctly across an arbitrary nesting of sequential
reduction tiers is basically a carry/scan problem, not just "emit a
fill."

## 8. Associativity/commutativity is a silent assumption

Splitting a reduction spatially (DPU-grid, tasklet) and merging afterward
in whatever order gather/barrier produces is only sound if the combiner
is associative and commutative — true for the arith reductions used now,
but `linalg.generic` in general allows arbitrary combiner regions. Worth
deciding upfront whether the pattern only targets recognized
arith-reduction bodies (as now) or needs to detect/require this property
for generic regions.

## 9. Reusing the combiner region across three different merge sites

The register-level accumulate, the tasklet barrier-merge, and the
host-side post-gather `linalg.reduce` all currently hardcode
`arith::AddIOp`/`AddFOp` or a fixed `cinm::ReduceMethod`. Generalizing
means cloning/reusing the *same* combiner region (and deriving its
identity element) at all three sites, which is more delicate for an
arbitrary `linalg.generic` body than for a known reduction kind.

## 10. Cost-model coverage

`simulateFullGemv`/`simulateTailReduction` are hand-written per op shape.
A general pattern needs a matching general cost function (or the
tile-size/role-assignment search has nothing to optimize against), and
memory-traffic estimation now depends on arbitrary operand indexing maps
rather than fixed formulas.

## Suggested next test cases

Before implementing anything, it's worth reasoning through (not yet
coding) these two shapes, since they're the smallest ones that force
decisions on points 2, 3, and 9 that gemv/reduce never had to make:

- **matmul-shaped generic** (2 parallel loops M,N + 1 reduction K)
- **double reduction** (e.g. reducing a 3D tensor over two axes)

---

<!-- Add your comments below or inline as blockquotes -->

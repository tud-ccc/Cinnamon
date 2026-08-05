# Launch fusion: consumers of a distributed op

Context: `--linalg-fuse-elementwise-ops` in the convert pipeline
([UpmemInferAccelerator.cpp:259](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L259))
fuses an elementwise *producer* into its consumer. The `prim_gemv` benchmark is
the other shape — `gemv` then elementwise — so nothing fuses, and the
elementwise gets a launch of its own with a full round trip through the host in
between.

    %4 = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    %5 = cinm.op.elementwise mul %4, %cs : tensor<4096xi32>

Whether that round trip is avoidable depends on the schedule the search picks
for the gemv, so this is not a transformation that can be decided in isolation:
it constrains, and is constrained by, the design space. This document says what
the criterion is, why it is a searched parameter rather than a rule, and what
has to change.

## A. What the IR actually shows

Both schedules, produced by hand from
[prim_gemv.mlir](../experiments/prim_gemv.mlir)'s 4MB case with the two ops in
one compute block (see §B), a `1x16x1` workgroup, and
`--convert-linalg-to-cnm=cnm-buffer-level=mram`.

**No K-split** — `gemv.K.mram = K`, so M is blocked 64 and there are 16 tiles:

```mlir
%3 = cnm.gather  %cnm_buf_1[#map] of %1 : buffer<64xi32> into tensor<1024xi32>
// ... nothing that touches %3 ...
cnm.scatter %3 into %cnm_buf_2[#map] of %4 : tensor<1024xi32> into buffer<64xi32>
```

Same map, same buffer type, same workgroup type. A pure round trip: leaf *i*
hands back exactly the elements leaf *i* is about to be handed again.

**K-split** — M blocked 256, K blocked 256, so 4×4 tiles:

```mlir
%6 = cnm.gather %cnm_buf_2[#map2] of %4 into %5 : buffer<1x256xi32> into tensor<4x1024xi32>
%7 = linalg.generic {iterator_types = ["reduction","parallel"]} ins(%6) outs(%0)  // host merge
cnm.scatter %7 into %cnm_buf_3[#map8] of %8 : tensor<1024xi32> into buffer<64xi32>
```

Different shapes, different maps, a host reduction in between —
[splitDistributedReductions](../lib/Conversion/LinalgToCnm/LinalgToCnm.cpp#L185-L267)
fired and the value the consumer wants does not exist on the device at all.

**So the criterion is a local pattern on CNM IR**: `gather` → `scatter` with
equal maps and equal buffer types on the same workgroup. That is also exactly
the legality condition for merging the two launches, so the transformation can
be written and verified without consulting the search space at all. The abstract
"tiled tree" model this design started from is unnecessary: the tree already
exists in the IR at the point where there are concrete factors to test.

## B. Blocker: the two ops are not in the same search

[AssignPlatformsPass](../lib/Dialect/Cinm/Transforms/AssignPlatformsPass.cpp#L41-L61)
wraps *each* `cinm.op.*` in its own `cinm.compute`, and
`cinm::inferAcceleratorConfig` runs once per `ComputeBlockOp`. Confirmed:
`prim_gemv.mlir` as written lowers to two `cinm.compute_block`s, so gemv and
elementwise get two independent searches picking two independent
`dpus`/`tasklets` — there is no joint space to constrain and no adjacency to
fuse.

GEVA and TTV work today because they are hand-written with an explicit
`cinm.compute` around several ops. `prim_gemv.mlir` is not.

**Decided:** step 0 is to put them in one block. **Done** — `prim_gemv.mlir`
now wraps both ops in an explicit `cinm.compute`. A
`--cinm-merge-compute-blocks` pass (merge adjacent blocks with a common
platform, no intervening op that reads a result) is the general fix and is
tracked separately.

## C. The criterion, derived

For an edge producer `p` → consumer `c` carrying value `v`, with `A_p` = p's
indexing map for the result and `A_c` = c's map for that operand. Both are
projected permutations — `--convert-linalg-to-cnm` requires that of every
operand anyway
([checkProjectedPermutations](../lib/Conversion/LinalgToCnm/LinalgToCnm.cpp#L110-L118)).

1. **Materialization.**

       ∀ d ∈ dims(p) \ range(A_p):  extent_p[d] / block_p[d] == 1

   A dimension p does not index its result with is a reduction dimension;
   tiling it over the workgroup is precisely what makes
   `splitDistributedReductions` rewrite the op and leave a host merge behind. So
   "the producer materializes full output tiles" falls out of the indexing maps,
   with no per-op knowledge of what a gemv is.

2. **Same leaf, same data.** Per dimension `k` of `v`:

       block_p[A_p⁻¹(k)] == block_c[A_c⁻¹(k)]

   plus agreement of the tile→workgroup linearization, i.e. a joint constraint
   on `p.order` and `c.order`
   ([CnmMemoryLevelsDesign §G3](CnmMemoryLevelsDesign.md#g3-tile-space--workgroup-order-a-rule-by-default-a-parameter-when-asked)).
   Together these are what make the gather and scatter maps equal.

3. **Leaf level.** Additionally `leaf_p[A_p⁻¹(k)] == leaf_c[A_c⁻¹(k)]`.

### C1. Condition 1 is not per-level

The obvious model — three memory levels, three tree levels, each needing "full
tiles at that level" — is wrong at the leaf.
[UpmemTileMRAMBuffers](../lib/Dialect/UPMEM/Transforms/UpmemTileMRAMBuffers.cpp#L328-L362)
tiles parallel dimensions in an outer loop and reduction dimensions in an inner
loop, staging the output tile outside the reduction loops. The leaf-level output
tile is therefore complete at the end of each outer-loop body **even when the
reduction is tiled in WRAM**, because that split is a sequential loop inside one
leaf rather than a spread across leaves.

Splitting a reduction *across the workgroup* blocks fusion. Splitting it *within
a leaf* does not. Condition 1 is a statement about the workgroup distribution
only; the level index adds equalities (2) → (3) and nothing else.

### C2. "Fuse at host level" is not fusion

Level 0 in the obvious enumeration means "two launches, host round trip", which
is today's behaviour and needs no parameter. The thing actually wanted in the
K-split case — *run the consumer on the host rather than offloading it at all* —
is a different decision, and one nothing in the current design can express:
every op `isDistributionCandidate` reports on gets a launch. Keep it separate; a
per-op `offload` boolean is a coherent extension, folding it into the fusion
level is not.

### C3. Why the level is searched, not fixed

Tempting shortcut: skip the parameter and require (2)+(3) unconditionally for
every producer/consumer edge. That is wrong — condition 1 forbids splitting the
producer's reduction across the workgroup, and for gemv on UPMEM that split is
the main source of parallelism. Fusion and reduction splitting are in genuine
tension, and which wins depends on the shape. So the level must be a variable.

The *transformations*, however, need not branch on it. Both fusion passes below
are opportunistic: they fire iff the pattern matches. The variable's only jobs
are (a) making the fusable corner of the joint space dense enough for the search
to find, and (b) keeping the two ops' parameters coupled. No conditional
lowering pipeline is needed, which was the main worry motivating this document.

### C4. What is already implied

For the rank-1 gemv case, condition 2 costs nothing. With K-tiles = 1 the
exact-fill constraint
([UpmemInferAccelerator.cpp:787](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L787))
gives `N / gemv.M.mram == dpus * tasklets` and `N / ew.D0.mram == dpus *
tasklets` over the *same* `dpus`/`tasklets`, hence `gemv.M.mram ==
ew.D0.mram`; and with one distributed dimension each, both orders are the
identity. Only conditions 1 and 3 need stating.

For a rank-≥2 result (gemm → elementwise) the per-dimension equalities and the
order agreement are real constraints and must be emitted.

## D. Step 1 — levels from the platform

`handleLinalgOp` hardcodes two levels by name:

```cpp
blocks.push_back(b.divisorsOf(base + ".mram", extent));
leaves.push_back(b.divisorsOf(base + ".wram", blocks.back()));
```

and reads their capacities through `platform.getMramLevel()` /
`getWramLevel()`. For a fusion *level* to be an index rather than a magic
0/1/2, this has to run over `platform.getLevels()`, which is already an ordered
`ArrayRef<CinmLevelDefAttr>` from farthest to closest
([UPMEMAttributes.cpp:156-160](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L156-L160)),
each carrying `getName()` and `getSizeInElements(Type)`.

- One `SpaceVar` per (dimension, level), named `<op>.<dim>.<level.getName()>`,
  each chained `divisorsOf` the previous level's variable.
- One capacity bound per level, `footprint(sizes_i) <= level_i.getSizeInElements(eltTy)`.
- The two stamped attribute lists become one list per level. Today
  `kOuterTileParamsAttr` and `kLeafTileParamsAttr` resolve to `cnm.tile_sizes`
  and `upmem.leaf_tile_sizes`; those two consumers are level-specific, so the
  mapping level→attribute stays in the UPMEM plugin. Only the *declaration* of
  the variables becomes generic.

**The parameter names do not change.** The UPMEM platform's levels are literally
named `mram` and `wram`, so `gemv.M.mram` stays `gemv.M.mram` and every
configuration recorded under `experiments/` keeps working. That is the point of
doing this first rather than alongside a rename.

*Done when:* the space printed for `prim_gemv` is identical to today's, and
`--upmem-infer-accelerator=eval-solution=...` accepts an existing pinned
configuration unchanged.

## E. Step 2 — `--cnm-fuse-launches` — **DONE**

[FuseLaunches.cpp](../lib/Dialect/Cnm/Transforms/FuseLaunches.cpp). A CNM-level
pass, no UPMEM dependency. Two patterns, in order:

1. **Cancel the round trip.** `cnm.scatter %v into %b2[#m] of %wg2` where `%v`
   is `cnm.gather %b1[#m'] of %wg1`, with `#m == #m'`, `b1`/`b2` of the same
   buffer type, `wg1`/`wg2` of the same workgroup type, and no op between the
   gather and the scatter that writes `%b1`. Replace uses of `%b2` with `%b1`
   and drop both ops. If `%v` has other uses (it is also the block's yielded
   value, say), keep the gather and drop only the scatter.
2. **Merge adjacent launches.** Two `cnm.launch` on the same workgroup with no
   intervening op that touches either's buffers: concatenate operands and
   splice the bodies. Pattern 1 is what makes them adjacent.

Both are checkable syntactically, so the pass is safe irrespective of what the
search does — it is a peephole, not a scheduler.

Two things came out of building it.

**The two launches are on two different workgroups.** `--cnm-hoist-workgroups`
followed by `--canonicalize --cse` leaves two distinct `cnm.workgroup` values of
the same type, plus two `cnm.free_workgroup`s; `cnm.workgroup` is not CSE-able,
and coalescing them is not obviously sound in general — acquiring a workgroup
twice asks for two sets of leaves. So `unifyWorkgroups` is *not* a
canonicalization: it runs only as part of cancelling a round trip, which is
exactly the event that establishes that the two halves want to be co-located.
It keeps the last release and refuses if any surviving use would outlive it.
Whether both halves' buffers still fit once they share leaves is a capacity
question, and §I leaves it to the occupancy check.

**Ordering has to be dominance, not block order.** Whether the workgroups end up
inside or outside the compute block depends on which one it is — `cinm.compute`
is not `IsolatedFromAbove` and its workgroups get hoisted out; `cinm.compute_block`
is, and they stay in. Either way the ops this pass compares routinely sit at
different region depths, and `DominanceInfo::properlyDominates` answers only
when the second op lies inside the first's region (it normalises that way and
gives up otherwise). Hence the `happensBefore` helper, which lifts whichever op
is deeper until the two are siblings.

*Placement:* in `buildFrontPipeline`, after `--cnm-hoist-workgroups` and before
bufferization, so `--cnm-scatter-optimizations` and the bufferizer see the
merged form.

*Tests:*
[fuse-launches.mlir](../test/Transform/Cnm/fuse-launches.mlir) — the §A no-split
IR fuses to one launch and one workgroup; the §A K-split IR is left alone; a map
mismatch is left alone; a gather with a second use keeps its gather but still
loses the scatter; `merge-launch-bodies=false` elides the transfers and keeps
two kernels. End to end,
[gemv-elementwise-fusion.mlir](../test/Transform/UPMEM/gemv-elementwise-fusion.mlir)
runs the 4MB `prim_gemv` case through the whole pipeline on two pinned
configurations: at `gemv.K.mram=1024` it comes out as one `upmem.alloc_dpus` and
one `upmem.dpu_program`, at `gemv.K.mram=256` as two, with the partials gathered
into a `memref<4x4x256xi32>`. One scatter and one gather of the 1024-element
intermediate disappear, along with a kernel dispatch.

## F. Step 3 — implication in the constraint IR — **DONE**

`ConstraintNode` had `Const, Var, Add, Mul, Sub, Div, Cmp` and no boolean
connective. Conjunction needs none — two `require` calls. Implication does.

One node kind, `Implies`, binary, both children boolean. Disjunction stays out:
no caller, and no story for the analyser. Surface syntax is a free function,
`b.require(implies(fuse >= 1, lhs == rhs))`, because no C++ operator's
precedence reads correctly for implication; the `Expr<Type::BOOL>` /
`Expr<Type::INT>` split added in `d5495e1` makes it type-check.

`evalCmpNodeInto` became `evalBoolNodeVec` / `evalBoolNodeInto` (`a => b` is
`!a | b`, both sides evaluated for every lane since a vectorized evaluation has
no short-circuit), and `cmpMayHold` became `boolMayHold` plus a new
`boolMustHold`. The dual matters: a consequent may only be acted on once its
antecedent *must* hold, since believing one that merely *might* would impose the
consequent on completions the constraint says nothing about.

**Division inside an implication** works, and getting there fixed a bug that
predates this. `a / b` in the DSL means "b divides a exactly", but the
interpreter used to compute a truncating quotient and rely on a divisibility
side condition being enforced separately — and `extractDivConstraints` cannot
extract one from under a guard, since what it reifies is unconditional. So
`implies(g, extent / block == 1)` would have silently accepted `block = 768`
against `extent = 1024`. Rather than ban the spelling, the evaluation was made
exact: each `Div` clears the lanes where it did not divide, and each comparison
ANDs that in, so an inexact division simply makes its comparison false. Per
comparison, not at the root — an inexact division in an *antecedent* falsifies
the antecedent, which satisfies the implication. See
[ConstraintAnalysisDesign.md § Division is exact division](ConstraintAnalysisDesign.md#division-is-exact-division).

The multiplied-out form is still the better spelling — it says what it means
without relying on that — but it is now a style preference rather than a
correctness requirement, and either way `matchProductEquality` cross-multiplies
it to `block == extent` before the enumerator sees it.

In `planComponents`:

- **Connectivity.** The antecedent's variables are unioned with the
  consequent's, so `fuse` lands in the same component as the tile variables it
  gates. Without this it keeps a slot of its own and the equalities are never
  absorbed.
- **Enforcement is free.** An implication goes into the enumerator's `bounds_`
  like any other comparison, and `boolMayHold(Implies)` is already exactly the
  right test: it prunes only once the guard is settled true and the consequent
  impossible. At full assignment every interval is a point, so it is exact.
- **Solving is not.** That only rejects; it never *determines*. So an
  implication whose consequent is a product equality is additionally recorded as
  a `GatedProdEq`, and `candidatesFor` / the unit-propagation tier consult it
  once `boolMustHold` settles the guard.
- **Scheduling.** `ComponentEnumerator::selectNext` tiered variables as
  unit-propagated → in a product equality → everything else. A guard variable is
  in neither of the first two, so it would be enumerated *last*, at which point
  its consequent can prune nothing. Guards now get their own tier, right after
  unit propagation: their domains are tiny (a mode switch) and until one is
  settled every equality hanging off it is inert.

*Verified* against brute force on four synthetic spaces (gated equality; two
nested gates; a gate alongside an unconditional product equality; a gate on an
inequality). The first three come out fully absorbed — the encoding offers
exactly the valid set and nothing else — with `fuse ∈ {0,1,2}` over two
7-divisor variables giving 63 = 49 unfused + 14 fused rather than the 147-point
product. The fourth is correct but unabsorbed, since an inequality consequent is
not a `ProdEq` and that space has no other structural relation to form a
component around.

There is no C++ unit-test target in this repo, so that check was a scratch
driver rather than a committed test. The committed regression test arrives with
§G, which is the first caller.

## G. Step 4 — declare the edges

A generic helper over the linalg ops of the block, called from
`initializeSpace` after the per-op `handleLinalgOp` pass:

1. For each pair (p, c) where c consumes a result of p and both are
   `isDistributionCandidate`, and both maps for that value are projected
   permutations: declare `fuse.<p>-><c>` over `[0, numLevels]`.
2. Emit condition 1 as `implies(fuse >= 1, prod over d ∉ range(A_p) of (extent_p[d] / block_p[d]) == 1)`.
3. Emit condition 2's per-dimension equalities gated on `fuse >= 1`, and the
   order agreement where both ops have ≥2 distributed dimensions.
4. Emit condition 3's per-dimension equalities gated on `fuse >= 2`, per level.

Nothing here mentions UPMEM. It belongs next to `SpaceBuilder`, or in a small
`Cnm` analysis library that the plugin calls; `handleLinalgOp` already does all
its reasoning through `linalgLoopExtents` / `linalgOperandDims`, which
generalise directly.

Note that `fuse` is *not* stamped on any op and no pass reads it. It exists
purely to shape the space. That is unusual enough to be worth a comment at the
declaration site.

## H. Step 5 — leaf-level fusion

Once §E has merged the launches, the two `linalg` ops sit in one body on
memrefs. If their leaf tile sizes agree — which is what `fuse = 2` buys —
`--upmem-tile-mram-buffers` produces two affine nests over the same range in the
same block, and `affine::createLoopFusionPass()` fuses them. That pass is
already written into `addAffineOpts` and commented out
([UpmemInferAccelerator.cpp:135](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L135)).

**Try enabling it before writing anything.** If it works, the whole of leaf-level
fusion is one uncommented line plus the constraints from §G, and the design
space parameter is carrying the entire weight of the transformation. If it does
not — most likely because the intermediate is a launch parameter in MRAM rather
than a local buffer, so the fusion has no reason to eliminate it — the fallback
is a dedicated pattern in `UpmemTileMRAMBuffers` that tiles a producer/consumer
pair jointly rather than one op at a time.

Either way this step is worthless without §E, and measurable separately from it.

## I. Capacity

`footprint(blocks) <= mram` is stated per op, and a fused launch holds both ops'
operands live at once. Leave it. [CnmMemoryLevelsDesign
§H4](CnmMemoryLevelsDesign.md) makes these deliberately *necessary-only*
conditions, with the exact test deferred to `--upmem-check-occupancy` on the
lowered program (§H3); under-approximating is safe by construction, and fusion
only ever loosens the true bound relative to the stated one. A configuration
that fuses and then does not fit is rejected late, like every other one.

## J. Staging

The order matters, because two of these produce a measurable win on their own:

| | status | depends on | independently valuable |
|---|---|---|---|
| §B one compute block | **done** | — | no (unblocks everything) |
| §D levels from platform | **done** | — | no (prerequisite for §G) |
| §E `--cnm-fuse-launches` | **done** | §B | **yes** — search finds K-tiles=1 configs and they fuse, with no space change at all |
| §F implication | **done** | — | no |
| §G edge constraints | | §D, §F | yes — makes the fusable corner reachable rather than lucky |
| §H leaf fusion | | §E, §G | **yes** |

**Next: measure.** §B, §D and §E are in, so the search is already free to find
fusable schedules and the pipeline already fuses them — with no change to the
space. What that costs and buys on the `prim_gemv` sizes is the justification
for §F/§G, and there is no point writing conditional constraints before seeing
it. If it turns out not to pay at the workgroup level, the leaf level almost
certainly does not either, and the constraint work is not worth doing.

## K. Open questions

- **Does the order agreement in condition 2 have a closed form?** For rank-1
  results it is vacuous and for the cases we lower today the producer has at
  most one distributed parallel dimension, so it is vacuous there too. Writing
  it in general means relating two `order` indices whose domains depend on which
  dimensions are distributed, which the ranking scheme (§G3) makes
  configuration-dependent. Possibly cleanest as an opaque predicate that decodes
  both permutations and compares, accepting that it will not be absorbed.
- **Chains longer than two.** `gemv → mul → add` gives two edges with a shared
  middle op. The conditions compose, but `fuse.a->b = 2, fuse.b->c = 0` is a
  configuration the passes will happily produce and whose cost model behaviour
  nobody has thought about. Probably fine; worth a test.
- **Multiple consumers.** A producer with two distributed consumers can fuse
  with at most one without duplicating work. Currently nothing forbids
  declaring both edges. Simplest rule: at most one edge per producer result may
  have `fuse > 0`, as another implication.
- **`--cinm-merge-compute-blocks`.** §B takes the hand-written route. The
  general pass has to decide what to do with non-cinm ops between two blocks
  (the `tensor.splat` in `prim_gemv` is one), which is the same question
  `--cinm-isolate-compute-blocks` answers in the other direction.

# Search space: implementation versus design

What the design documents and the paper draft say about the configuration
search space, versus what
[`SpaceBuilder`](../include/cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h)
and `ConfigSpace` actually do. Sources compared:

- **[ConstraintAnalysisDesign.md](ConstraintAnalysisDesign.md)** — the design
  the current implementation was built from. Stages 1, 2, 4 are marked done,
  Stage 5 (component enumeration) was built and is not marked, Stage 3 and 6
  are deferred.
- **The paper draft** (§ "Positioning" / "Representation" / "Design"), which
  states the intended model in its own vocabulary — *subspaces*, *density* —
  and is the more recent statement of intent.
- **[CnmMemoryLevelsDesign.md](CnmMemoryLevelsDesign.md) §G2/§G3/§H4** and
  **[LaunchFusionDesign.md](LaunchFusionDesign.md) §F/§G**, which say what
  parameters and constraints exist rather than how they are encoded.

A technical description of what is implemented now lives at the top of
`SpaceBuilder.h`, deliberately in the code so it stays with it. This document
holds only the *differences*, and dies as they are closed.

Everything below is stated as: **what is said**, **what is**, **what it costs**.

---

## 1. Planning results are not observable

**Said.** Nothing, in either source — this is a gap in the design as much as in
the code.

**Is.** `SpaceBuilder::planComponents` decides the entire shape of the space:
which parameters were merged, which relations were folded in, which fell back
to a pairwise group, which survived as filters, and which components blew a
budget and absorbed nothing. All of it is `LLVM_DEBUG` text, lost the moment
`buildInto` returns, and none of it reaches `dumpMetadataJSON`.

**Costs.** The space cannot be characterised from an experiment's artefacts:
"why is this space 3M points" is answerable only by re-running with
`--debug-only=cinm-inference` and reading prose.

**Proposed.** `SpaceBuilder` builds a metadata record and hands it to the
`ConfigSpace` as an opaque owned pointer, which the dump serialises: per
component, its parameters and its tuple count; per constraint, its rendered
form and its disposition (static filter, structural pair, folded into
component *k*, surviving filter, opaque). The constraint IR already renders
itself (`describeNode`), so the content exists — it is only ever printed.

## 2. Density is defined but never measured

**Said.** The paper makes density — "the fraction of the enumerated product
that is feasible" — the metric the partition is judged by: "we merge subspaces
as long as doing so lets a constraint be folded into the merged enumeration",
with density one at the coarse extreme and the full Cartesian product at the
fine one.

**Is.** The implementation reports, per component, `tuples (was P, Nx fewer)`
where `P` is the product of that component's *declared* domains. That is a
compression ratio against the Cartesian product, not a density: it says nothing
about how many of the enumerated tuples are feasible. Density (`feasible /
addressable`) is computable — both numbers are now printed — but is never
formed, and in particular is never attributed to the partition that produced
it.

**Costs.** The one number that would say whether a partition is good is
missing, so there is no evidence for or against the merging policy in §3.

## 3. The merging policy is not the one described

**Said.** "We merge subspaces as long as doing so lets a constraint be folded
into the merged enumeration" — merging is a decision, justified per constraint.

**Is.** Merging is unconditional. Every structural relation contributes an edge
to a union-find, and connected components fall out; a relation between two
parameters merges them whether or not folding it pays. The only feedback is
after the fact and all-or-nothing: if the component's enumeration exceeds
`kSolutionCap` (4e6 tuples) or `kNodeBudget` (5e7 nodes), it is abandoned
*entirely* and every relation in it falls back to pairwise groups and
predicates. There is no finer partition tried in between, and no cost model.

**Costs.** Fine on the spaces measured so far, because the relations really do
all interlock. It degrades badly the moment one weak relation bridges two
otherwise separate clusters: a single edge can merge two components, blow the
budget, and lose the folding of *both*, which is the worst outcome available.

**Proposed.** Nothing yet — the failure mode has not been observed. Worth
recording because the paper claims a policy the code does not implement, and
because §4 makes it likelier to be hit.

## 4. Guard variables multiply enumeration cost

**Said.** LaunchFusionDesign §F: implication support is "free" in the
enumerator — "an implication goes into the enumerator's `bounds_` like any
other comparison", plus a scheduling tier so a guard is settled early.

**Is.** Correct but not cheap. Adding one `fuse` variable with three values to
`prim_gemv`'s 4MB space (`gemv` + `elementwise`, 9 parameters) takes space
construction from **3.2s to 2m14s**, ~40×, for a component that grows from
7,982 to 10,852 tuples in the small case and 92,538 tuples at 1024². The
enumerated *result* grows by ~36%; the *work* grows by 40×.

The cause is not the per-read guard evaluation — hoisting `boolMustHold` out of
`anyActiveProdEq` into a cached `gatedActive_` refreshed on guard assignment
changed nothing measurable. What is left is structural: the guard is assigned
first (tier 0), and the unfused branch re-enumerates the entire component with
the gated equalities inert, so the space is walked once per guard value with no
pruning in the branch that has no constraints.

**Costs.** 2m14s of single-threaded work before any evaluation begins, on the
smallest of the four `prim_gemv` sizes. It scales with the space, and with the
number of edges — a three-op block has two guards, i.e. 9 branches.

**Proposed.** Not obvious. The unfused branch is *exactly* the unconstrained
enumeration, so it could be enumerated once and the fused branches computed as
a refinement rather than from scratch; that is a real change to the
enumerator's shape. Alternatively a guard could be kept out of the component
(its own slot) and its implications left as filters, trading density for
construction time — which is precisely the merging decision §3 says should
exist and does not.

## 5. `DependentGroup` was supposed to disappear

**Said.** ConstraintAnalysisDesign Stage 5: component enumeration "subsumes
`DependentGroup`, which is exactly the two-variable case".

**Is.** Both encodings exist and both are live. `planComponents` absorbs the
divisibility relations it can, and `buildInto` then commits the rest through
`addMultiplesConstraint`, with two further fallbacks (mutual divisibility, and
a chain whose parent is already another pair's child) that degrade to dynamic
predicates. `ConfigSpace` therefore carries three slot kinds, and `at()`,
`indexOf()`, `forEachChunk()` and `isEncodable()` each branch on all three.

**Costs.** Duplicated decode logic in four places, and two representations to
keep in step for anything that touches the encoding. Not incorrect.

**Proposed.** Once components handle the pairwise case with no budget risk —
which they do; a two-variable divisibility component is trivially small — a
`DependentGroup` is a `SolvedComponent` whose enumeration happens to be a pair
list. Deleting the kind is a contained cleanup.

## 6. A permutation's *neighbours* are still its rank ±1

**Said.** The paper: "Categorical and permutation-valued parameters are also
supported, by modifying the distance function for those variables as proposed
by BACO."

**Is.** The surrogate side is done. `SearchParam` carries a `ParamKind`
(`Integer`/`Permutation`), the rank/unrank encoding is shared
(`cinm-mlir/Utils/Permutation.h`), the DSL rejects arithmetic and ordering on a
rank, and `appendFeatures` presents a permutation as its position vector, so
distance between feature vectors is Spearman's rank distance rather than
distance between two arbitrary numberings.

What still treats a rank as a magnitude is **`neighborIndices`**, which steps
±1 on every parameter's sub-index. For a permutation that lands on an unrelated
permutation, so BO's local candidate generation (`fillNeighbors`) explores a
neighbourhood that is not one.

**Costs.** Bounded: it degrades candidate *proposal*, not the model. The
acquisition function still ranks whatever is proposed correctly.

**Proposed.** `neighborIndices` dispatches on the kind, and a permutation's
neighbours are the ranks reachable by one adjacent transposition — decode,
swap a pair, re-rank. Independent of everything else here.

*(A categorical kind has no caller yet, so it is not declared.)*

## 7. `discretize` is dead, and its contract disagrees with the encoding

**Said.** Nothing explicit; the paper assumes a well-defined continuous
relaxation for the surrogate.

**Is.** `SearchParam::discretize` — "map a continuous sample in `[dlo, dhi]` to
the nearest valid discrete value" — **has no callers**. Nor do `dlo()/dhi()`
outside one debug print. That matters because the contract they describe is not
the one the encoding implements: `discretize` reads its argument as an index
into the value list, while features are now the value scaled to `[0, 1]` (and
log-scaled first for a value list). Anyone who wired the continuous relaxation
back up by following these signatures would get silently wrong values.

**Costs.** None today, since nothing calls them. The risk is that they read
like a supported inverse of the encoding.

**Proposed.** Delete `discretize`, or reimplement it as the actual inverse of
`appendFeatures` if a caller appears. `dlo/dhi` should report the value range
for a value list rather than the index range, since a debug print is all they
feed.

## 8. Declared domains are much wider than reachable ones

**Said.** Implicit in the paper's model: a subspace enumerates "the tuples that
satisfy the constraints mentioning only its parameters", so a parameter's
domain is expected to be the values it can take.

**Is.** `divisorsOf(name, SpaceVar)` declares `[1, maxVal]` — every integer —
and records the divisibility as a relation. At 1024², `gemv.M.wram` is declared
with 1024 values where at most 11 are reachable. The component enumeration
never offers the other 1013, so the space is right; but the *declared* domain
is what feeds the Cartesian-product figure, `neighborIndices`, `discretize`,
and `dlo/dhi`.

**Costs.** Inflates the Cartesian figure by orders of magnitude (4.2e17 against
an addressable 185k at 1024²). Makes BO's continuous relaxation mostly-invalid
space for those parameters.

**Proposed.** A leaf-level tile divides its parent, which divides the extent,
so `keepDivisorsOf(extent)` is sound for the inner levels too and is a
one-line change at the declaration site. It narrows the declared domain to the
reachable one without touching the encoding.

## 9. Statements in ConstraintAnalysisDesign that are now stale

Not gaps — the document simply predates the code and should be corrected or
retired.

| says | actually |
|---|---|
| `ConstraintNode::Kind` includes `Sub` and a single `Cmp` kind | there is no `Sub`; the six comparisons are six kinds, and `Divides`/`Implies` were added after |
| "`neighborIndices()` needs rewriting — but it is already wrong" | rewritten: it decodes, steps one dimension, and re-encodes, exactly as the document proposed |
| the direction rule, "enumerate the side with smaller joint cardinality, solve for the other" | never implemented as such. `reportConstraintAnalysis` *prints* it; the enumerator uses a dynamic `selectNext` (determined → guard → in-equality, narrowest domain → rest) and `solveFor` fires only on the last unknown of an equality. The print degenerates when one side normalises to a bare constant — "enumerate {1048576} (1 combos), solve for {dpus * tasklets * gemv.M.mram * gemv.K.mram}" describes nothing anyone could do, and nothing the code does. The paper's version of the rule — "pick the variable with the widest domain as the determined one" — is likewise only approximated, by preferring narrow domains among the others |
| Stage 3, scratch-buffer pool for `evalNodeVec` | still deferred; every node still allocates a fresh `ParmVector` |
| Stage 6, Form C domain narrowing | not implemented |
| Form A2, `dpus == dpuRows * dpuCols` inside `readAsGemvTemplate` | still invisible to the analyser, still behind an opaque predicate |

## 10. Opaque predicates bound density from above

**Said.** The paper: a subspace enumeration produces "exactly the tuples that
satisfy the constraints mentioning only its parameters".

**Is.** Only the constraints it can *read*. An opaque predicate over parameters
that all lie inside one component is still evaluated after the fact, so the
component offers tuples that are then filtered out. Two exist today: the
workgroup order domain (`order <= (number of distributed dimensions)!`) and the
fusion order agreement. Both mention only one component's parameters.

**Costs.** Density is capped below 1 by construction on any space with an
opaque predicate, and the cap is invisible (§2). At 64², 10,852 tuples become
14,557 valid points out of 21,704 addressable — a density of 0.67, essentially
all of it the order predicate.

**Proposed.** A component could accept an opaque predicate as a full-assignment
filter, evaluated at the leaves of the backtracking walk. It cannot prune
subtrees — that is what "opaque" means — but it would remove the tuples from
the table, restoring density at the cost of enumerating them once.

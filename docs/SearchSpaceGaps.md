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

## 1. Density never drives a merging decision

**Said.** The paper makes density — "the fraction of the enumerated product
that is feasible" — the metric the partition is judged by: "we merge subspaces
as long as doing so lets a constraint be folded into the merged enumeration",
with density one at the coarse extreme and the full Cartesian product at the
fine one. Merging is presented as a decision, justified per constraint.

**Is.** Density is measured and reported — `space.json` carries the three
sizes, both ratios, and now the components and each constraint's disposition —
but nothing reads it back. Merging is unconditional: every structural relation
contributes an edge to a union-find, and connected components fall out, so a
relation between two parameters merges them whether or not folding it pays.

The only feedback is after the fact and all-or-nothing: if a component's
enumeration exceeds `kSolutionCap` (4e6 tuples) or `kNodeBudget` (5e7 nodes) it
is abandoned *entirely*, and its divisibility relations are retried one at a
time as two-parameter components, with whatever fails that left to predicates.
There is no finer partition tried in between, and no cost model anywhere.

**Costs.** Fine on the spaces measured so far, because their relations really
do all interlock. It degrades badly the moment one weak relation bridges two
otherwise separate clusters: a single edge can merge two components, blow the
budget, and lose the folding of *both*, which is the worst outcome available.

**Proposed.** Nothing concrete — the failure mode has not been observed, and
the reporting to detect it now exists. Worth recording because the paper claims
a policy the code does not implement, and because §2 makes it likelier to be
hit.

## 2. Guard variables multiply enumeration cost

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

### 2a. The branch count does not account for the magnitude

Three guard values can multiply the work by at most three if nothing else
changes. The observed factor is 40, so most of it is *not* the extra branches,
and the account above is incomplete. The other term is per-node constraint
scanning, which is linear in the number of constraints in the component:

- `selectNext` calls `solveFor` over every equality, unconditional and gated,
  for every unassigned dimension, at every node.
- `boundsMayHold` walks every node of every bound for every candidate value,
  and an implication is a deep node — a comparison wrapping a product equality.

One fusion edge emits roughly one implication per value dimension per level,
so it adds 6–12 constraints to the component, not one. Multiplying a ~3×
branching factor by a ~6–12× per-node cost lands in the right neighbourhood.

This is inferred, not measured, and one counter separates the two: `nodes_`
already exists in `ComponentEnumerator`; print it. If it grows ~3× while wall
time grows 40×, the cost is scanning, and the fix is to index `bounds_` and
`gated_` by variable so that assigning a variable rescans only the constraints
mentioning it. If it grows ~40×, the branch structure is the cause and §2c
applies. Worth doing before either proposal below is implemented, because they
address the branching only.

### 2b. Rejected: a profitability test on the antecedent's cardinality

Fold an implication only when the antecedent's joint domain is small and the
consequent is estimated to cut enough to pay for the branching. For `fuse` the
antecedent has cardinality 3 and the consequent is an equality, so the test
would say fold.

It would say fold for the case that regressed, and for four separate reasons
it is measuring the wrong things.

1. **The antecedent's cardinality is the wrong axis.** What folding costs is
   (number of antecedent assignments that *falsify* it) × (cost of enumerating
   the component with the consequent inert). `implies(fuse >= 2, …)` has one
   falsifying value out of three; narrowing `fuse` to two values would not
   make it cheaper, and widening it to ten while keeping one falsifying value
   would not make it dearer.
2. **It scores solution counts; the pathology is search-tree size.** The table
   grew 36% and the work grew 40×. Density is likewise a ratio of counts, so
   §1's "let density drive merging" inherits the blindness: a merging rule
   needs a *work* model, not only a density model. The two coincide only where
   the enumerator is output-sensitive, and it is not across a branch in which
   every gated equality is inert.
3. **Estimating the cut is the hard part, and the cheap estimate is biased.**
   `producerTile == consumerTile` cuts by the collision rate over the pairs
   that are actually *reachable*, not by `1/|dom|` — the tile variables are
   already correlated through the divisibility chain. §3 records that declared
   domains run up to ~100× wider than reachable ones (1024 declared against
   ~11 reachable), and that error lands straight in the estimate, biasing
   every equality towards "very profitable".
4. **A per-constraint score for a per-component decision.** Dropping one
   implication can leave the component split, which changes what every other
   constraint is worth. Greedy scoring is defensible, but it is greedy and
   should say so.

What survives is the one quantity it computes both cheaply and correctly: the
number of falsifying antecedent assignments. That is what §2c needs.

### 2c. Partly viable: enumerate once unguarded, refine per branch

Put the antecedent variables last and evaluate the constraints they switch on
against the solutions already generated. Stated precisely for one implication
with antecedent variables `g`, partitioning `dom(g)` into `F` (falsifies) and
`S` (satisfies): enumerate the component once with the consequent dropped,
giving `T₀`; the falsifying branches are then `T₀ × F` at no further cost, and
each satisfying branch is `T₀` filtered by the consequent.

The falsifying half is right and the satisfying half is backwards.

- **Filtering loses exactly where enumerating wins.** A filtering pass costs
  `|T₀|` per satisfying assignment, while enumerating with the consequent
  *active* costs about `|T_s|` — and the consequent is an equality, so
  `|T_s| ≪ |T₀|`. So the hybrid is the one to want: share a single `T₀` across
  all of `F`, and keep the current pruned enumeration for each `s ∈ S`. Work
  drops from `|F|` unconstrained enumerations to one.
- **On this space the saving is nil.** `fuse` ranges over `1..numLevels+1` and
  the guards are `fuse >= 2`, `fuse >= level+2`, so `|F| = 1` for the level-1
  implications and 2 for the deepest. The saving factor is `|F|`. It cannot
  explain the 40× and would not remove it — which is the same conclusion §2a
  reaches from the other direction.
- **It does not compose across implications.** Several implications over one
  guard nest rather than partition: "the unguarded enumeration" is only well
  defined per *set of simultaneously inert consequents*, so sharing tables
  across the falsifying region means up to `2^k` tables for `k` implications,
  not `|F|`. With two edges (a three-op block) the falsifying region is
  already a union of boxes rather than a slice.
- **"Antecedent variables last" is the wrong order.** A guard settled late
  leaves its consequents inert for the whole prefix, which is precisely the
  branch that prunes nothing. Tier 0 (guards first) is right for the
  satisfying branches. Under the hybrid the tension disappears: `F` is handled
  without enumerating `g` at all, and inside a satisfying branch `g` is
  already fixed before anything else is chosen.

### 2d. Not an option: demoting implications to predicates

Keeping the guard in its own slot and leaving its implications as filters
would trade density for construction time, which is the merging decision §1
says should exist. But it is strictly worse than demoting an unconditional
constraint: the consequent is an equality between tile variables, so leaving
it out of the encoding widens the addressable space by the entire cross
product of the tiles it equates, and the filter then rejects all but the
diagonal. It also removes the only reason for `Implies` to be a constraint
form rather than an opaque predicate. The path exists as the abandonment
fallback (§1) and should stay a fallback.

## 3. Declared domains are much wider than reachable ones

**Said.** Implicit in the paper's model: a subspace enumerates "the tuples that
satisfy the constraints mentioning only its parameters", so a parameter's
domain is expected to be the values it can take.

**Is.** `divisorsOf(name, SpaceVar)` declares `[1, maxVal]` — every integer —
and records the divisibility as a relation. At 1024², `gemv.M.wram` is declared
with 1024 values where at most 11 are reachable. The component enumeration
never offers the other 1013, so the space is right; but the *declared* domain
is what feeds the Cartesian-product figure and `dlo/dhi`.

**Costs.** Inflates the Cartesian figure by orders of magnitude (4.2e17 against
an addressable 185k at 1024²). Makes BO's continuous relaxation mostly-invalid
space for those parameters.

**Proposed.** A leaf-level tile divides its parent, which divides the extent,
so `keepDivisorsOf(extent)` is sound for the inner levels too and is a
one-line change at the declaration site. It narrows the declared domain to the
reachable one without touching the encoding.

## 4. Statements in ConstraintAnalysisDesign that are now stale

Not gaps — the document simply predates the code and should be corrected or
retired.

| says | actually |
|---|---|
| `ConstraintNode::Kind` includes `Sub` and a single `Cmp` kind | there is no `Sub`; the six comparisons are six kinds, and `Divides`/`Implies` were added after |
| "`neighborIndices()` needs rewriting — but it is already wrong" | rewritten: it decodes, steps one dimension, and re-encodes, exactly as the document proposed — and a step is now whatever the parameter's kind says it is |
| the direction rule, "enumerate the side with smaller joint cardinality, solve for the other" | never implemented as such. `reportConstraintAnalysis` *prints* it; the enumerator uses a dynamic `selectNext` (determined → guard → in-equality, narrowest domain → rest) and `solveFor` fires only on the last unknown of an equality. The print degenerates when one side normalises to a bare constant — "enumerate {1048576} (1 combos), solve for {dpus * tasklets * gemv.M.mram * gemv.K.mram}" describes nothing anyone could do, and nothing the code does. The paper's version of the rule — "pick the variable with the widest domain as the determined one" — is likewise only approximated, by preferring narrow domains among the others |
| Stage 3, scratch-buffer pool for `evalNodeVec` | still deferred; every node still allocates a fresh `ParmVector` |
| Stage 6, Form C domain narrowing | not implemented |
| Form A2, `dpus == dpuRows * dpuCols` inside `readAsGemvTemplate` | still invisible to the analyser, still behind an opaque predicate |

## 5. Opaque predicates bound density from above

**Said.** The paper: a subspace enumeration produces "exactly the tuples that
satisfy the constraints mentioning only its parameters".

**Is.** Only the constraints it can *read*. An opaque predicate over parameters
that all lie inside one component is still evaluated after the fact, so the
component offers tuples that are then filtered out. Exactly two survive as
filters on the `prim_gemv` space, and `space.json` now names them both: the
workgroup order domain (`order <= (number of distributed dimensions)!`) and the
fusion order agreement, each marked `"analysable": false`.

**Costs.** Density is capped below 1 by construction on any space with an
opaque predicate. At 64², 10,852 tuples become 14,557 valid points out of
21,704 addressable — a density of 0.67, essentially all of it the order
predicate.

**Proposed.** A component could accept an opaque predicate as a full-assignment
filter, evaluated at the leaves of the backtracking walk. It cannot prune
subtrees — that is what "opaque" means — but it would remove the tuples from
the table, restoring density at the cost of enumerating them once.

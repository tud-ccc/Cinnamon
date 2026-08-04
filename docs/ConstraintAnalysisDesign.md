# Analysable constraints for the accelerator search space

Context: the configuration search in
[`AcceleratorInference`](../lib/Dialect/Cinm/AcceleratorInference/) enumerates a
flat Cartesian product of search parameters and filters it with predicates
registered through `SpaceBuilder::require`. Today those predicates are opaque —
either C++ lambdas or type-level expression templates — so the framework can
only *test* a configuration, never *reason* about it.

That opacity is expensive. On a `gemv` over `tensor<8192x16384xi32>` the space
reports:

```
Config space (7 params, 2770769301995520 total points)
Screening 1238630400 configs in parallel   →   3038 valid
```

1.24e9 configurations are enumerated and evaluated to find 3038. The arithmetic
says where it goes:

```
1,238,630,400  /  (2048 dpus × 24 tasklets)  =  25,200
```

The block/leaf structure contributes 25,200 points; the remaining **49,152×** is
the full `dpus × tasklets` cross-product, enumerated in its entirety for every
block combination and then discarded by a single predicate in
`handleLinalgOp`:

```cpp
prod(extent / block) == dpus * tasklets
```

That constraint *determines* `dpus * tasklets` from the block sizes. If the
framework could see that, it would enumerate the blocks and solve for the
`(dpus, tasklets)` pairs, cutting enumeration by roughly four orders of
magnitude. Vectorising predicate evaluation (already done) buys a 2–4× constant
factor on top of work that is 99.99975% wasted.

**Goal:** make constraints a data structure the framework can analyse, so that
(a) structurally implied relationships are folded into the index encoding rather
than filtered afterwards, and (b) a vectorised implementation is generated
automatically instead of hand-written per constraint.

## Why a runtime IR rather than the type-level DSL

The current DSL in [`SpaceBuilder.h`](../include/cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h)
is CRTP expression templates: `SpaceVar`, `ConstExpr`, `BinExpr<L,R,Op>`,
`ConstraintExpr<L,R,K>`. Structure lives in the *type*, and analysis is
`if constexpr` dispatch — see `extractDivConstraints` / `addDivConstraint`,
which recognise exactly one pattern (a `/` node) and are already at the limit of
what is comfortable to write that way.

Two things have changed that make the type-level encoding a poor fit:

1. **Its main benefit is gone.** Type-level encoding buys inlined scalar
   evaluation, which mattered when predicates ran once per configuration. They
   now run once per *chunk* over a `ConfigurationVector` of thousands of lanes,
   so a virtual call per node is amortised into irrelevance.

2. **Analysis wants the opposite of what it provides.** Recognising
   `f(A) == g(B)`, partitioning variables into determining and determined, and
   choosing a direction are ordinary pattern matches over arbitrary shapes. With
   templates you can only handle shapes enumerated at compile time, and the
   logic smears across specialisations.

A third benefit falls out of the switch. Every `evalVec` node currently
materialises a fresh Armadillo temporary; profiling shows the resulting
allocation churn dominated by kernel page-fault handling (`down_read_trylock`,
`__handle_mm_fault`, `clear_page_erms` — ~55% of cycles on a 64-thread run). An
interpreter walking a node tree can own a small pool of chunk-sized scratch
buffers and reuse them across nodes and constraints. That is hard to arrange
when every intermediate is a distinct C++ type returned by value.

**The operator-overloading surface syntax stays.** `blockM * tasklets <= wram`
should still be how constraints are written. What changes is that the operators
build a runtime node instead of *being* the representation.

## The IR

A small, closed node set — deliberately not a general expression language,
because everything here must stay analysable:

```cpp
struct ConstraintNode {
  enum class Kind { Const, Var, Add, Sub, Mul, Div, Cmp };
  Kind kind;
  // Const
  ParmValue value;
  // Var: index into ConfigSpace::params, resolved at buildInto() time
  std::shared_ptr<size_t> varIdx;
  // Cmp
  CmpKind cmp;
  // Children: n-ary for Add/Mul, binary for Sub/Div/Cmp
  SmallVector<ConstraintNodePtr, 2> operands;
};
```

Notes on the shape:

- **`Add` and `Mul` are n-ary, not binary.** This is what makes
  `prod(extent/block)` a single node with one child per iteration dimension,
  rather than a left-leaning binary tree that the analyser would have to
  re-flatten. Arity is a runtime value (the number of iteration dimensions),
  which is exactly what the type-level encoding could not express.
- **`Div` stays binary and keeps its current meaning**: `a / b` asserts `b`
  divides `a` exactly. `extractDivConstraints` already relies on this; the
  normaliser inherits it.
- **`Var` holds the same `shared_ptr<size_t>` cell `SpaceVar` uses**, so handles
  keep working across `buildInto()` and node identity is pointer identity.

### Normal form

Analysis runs on a canonicalised tree, so the recogniser matches one shape
rather than many spellings:

- Flatten nested `Add`/`Mul` into their parent.
- Fold constant subtrees.
- Sort `Mul` operands: constants first, then variables by index. Makes
  `dpus * tasklets` and `tasklets * dpus` the same node.
- Rewrite `a / b` inside a `Mul` chain into a *product with a divisibility side
  condition*, so a product-equality can be read off without division.

A `Cmp` in normal form is `<product-of-factors> <op> <product-of-factors>`,
where a factor is a constant or a variable, plus a set of divisibility side
conditions.

## Constraint forms to handle

Real examples, all from
[`UpmemInferAccelerator.cpp`](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp).

### Form A — product equality (the one that pays)

```
prod(factors) == prod(factors)
```

where either side may be a single variable or a constant, and constant factors
may appear on either side.

**Example A1** — `handleLinalgOp`, the tile-count constraint (design §G2). The
tile counts must fill the workgroup exactly:

```cpp
prod over dims of (extent[d] / block[d])  ==  dpus * tasklets
```

This is the 49,152× waste described above. `extent[d]` is a constant, `block[d]`
a variable, so after normalisation the left side is a product of variables with
a constant coefficient and a divisibility condition per dimension.

**Example A2** — `readAsGemvTemplate`, reached from `registerGemvTemplate` and
`registerReduceTemplate` via the `derive(c).has_value()` predicate. Buried
inside the helper is:

```cpp
if (dpus != dpuRows * dpuCols) return std::nullopt;
```

which is the same form, one level down. It is currently invisible to the
framework because it sits inside a C++ helper behind an opaque lambda.

**What the analyser does with it:** pick one side to enumerate and solve for the
other (see *Direction*, below).

### Form B — divisibility

```
b | a          (spelled a / b in the DSL, or a % b == 0)
```

**Example B1** — `handleLinalgOp` declares leaf tiles as divisors of block
tiles:

```cpp
blocks.push_back(b.divisorsOf(base + "0", extent));
leaves.push_back(b.divisorsOf(base + "1", blocks.back()));
```

Already handled structurally: `SpaceBuilder::divisorsOf(name, SpaceVar)` records
a `multiples_` entry, and `buildInto` turns it into a
`ConfigSpace::DependentGroup` baked into the flat index. This is the existing
proof that the approach works — it is why the raw 2.77e15 product collapses to
1.24e9.

**Example B2** — `registerGemvTemplate`, "the WRAM tile divides the MRAM tile":

```cpp
d->mramRow % (d->taskletRows * d->wramRow) == 0 &&
d->mramCol % (d->taskletCols * d->wramCol) == 0
```

Divisor is a *product of variables*, not a single variable, so today it falls
through `addDivConstraint`'s `is_mul_expr_v` branch into a dynamic predicate.

**What the analyser does with it:** a single-variable divisor against an
already-enumerated dividend extends the existing `DependentGroup` path.
A product divisor is generally a filter, not a determiner, unless all but one
factor is already fixed.

### Form C — capacity bounds (prune, never determine)

```
sum of products  <=  constant
```

**Example C1** — `handleLinalgOp`, the MRAM/WRAM footprint checks, currently
written as the hand-vectorised `footprintFits` lambda:

```cpp
sum over operands of (prod over that operand's dims of block[d])  <=  mramElements
```

**Example C2** — `registerGemvTemplate`, "the gemv template's WRAM working set
fits":

```cpp
t*wramRow*wramCol + taskletCols*wramCol + t*wramRow + (t/taskletCols)*wramRow
    <= wramElements
```

These cannot fix any variable, but they are *monotone* in every variable (all
quantities are positive sizes). That admits a cheaper use: interval propagation
to shrink a variable's declared domain before enumeration begins. If the largest
value of `gemv.M0` already busts MRAM on its own, every configuration
containing it is dead and the value can be dropped from the domain outright —
`SearchParam::keepDivisorsOf` already demonstrates domain narrowing at build
time.

Worth treating as a second-phase optimisation: it is strictly weaker than Form A
and only pays when domains are wide.

### Form D — not analysable, stays a predicate

**Example D1** — `handleLinalgOp`, the workgroup dimension order:

```cpp
orderVar < factorial(number of dims with extent/block > 1)
```

`factorial` is not algebraic and the right-hand side depends on a *count* of
satisfied conditions. This must remain an opaque predicate.

The design must keep opaque predicates as a first-class option, not an
afterthought — `SpaceBuilder::require(Constraint)` and `require(VecConstraint)`
both stay. Analysis is an optimisation layered on top, never a requirement.

## Direction: which side to solve

For `prod(A) == prod(B)`, either side could be derived. The rule:

> Enumerate the side whose variables have the smaller joint cardinality; solve
> for the other.

For A1: blocks contribute 25,200 combinations, `dpus × tasklets` contributes
49,152. Enumerate blocks, derive `(dpus, tasklets)`. Total enumerated becomes
`sum over block combos of |{(dpus,tasklets) : dpus*tasklets == P}|` — bounded by
the number of factor pairs of each `P`, a handful rather than 49,152.

Ties, or a variable appearing on both sides, fall back to a dynamic predicate.

## Encoding: generalising `DependentGroup`

`ConfigSpace::DependentGroup` today handles exactly one shape — one parent
variable determining the valid values of one child:

```cpp
struct DependentGroup {
  size_t parentIdx, childIdx;
  std::vector<std::vector<int64_t>> childValues;  // by parent sub-index
  std::vector<size_t> cumCount;
};
```

`ensureEncoding` merges parent and child into a single slot of size
`totalCount()`, and `at()` decodes a slot sub-index with `upper_bound` over
`cumCount`. Form A needs the same machinery widened from *scalar key → list of
values* to *tuple key → list of tuples*:

```cpp
struct SolvedGroup {
  SmallVector<size_t> keyDims;     // enumerated
  SmallVector<size_t> solvedDims;  // derived
  /// For each joint sub-index over keyDims (mixed-radix over their
  /// cardinalities), the valid tuples for solvedDims.
  std::vector<SmallVector<SmallVector<ParmValue>>> solutions;
  std::vector<size_t> cumCount;    // prefix sums, as today
};
```

Decoding is unchanged in shape: `upper_bound` over `cumCount` gives the key
combination, the remainder indexes into that key's solution list, and the tuple
is written into the `solvedDims` positions of the configuration.

**Precomputation cost.** For A1: 25,200 key combinations, and solving each means
walking the `dpus` domain testing `P % dpus == 0` with `P / dpus` in the
`tasklets` domain — 25,200 × 2,048 ≈ 5e7 trial divisions, well under a second,
once, at space-construction time.

**Guards** (each falls back to a dynamic predicate, never to a wrong answer):

- Key cross-product above a threshold → precomputation would blow up.
- Total solution count above a threshold → the group is not actually pruning.
- A dimension already claimed by another group → `buildInto` already has this
  situation for chained and mutual divisibility (`childSet`, `equalityFallbacks`,
  `dynamicDivFallbacks`); extend that logic rather than inventing a second one.
- Any variable appearing on both sides of the equality.

## Evaluation

One interpreter replaces both `evalImpl` and `evalVecImpl`:

- **Vectorised**: post-order walk over the node tree, each node writing into a
  scratch buffer drawn from a per-chunk pool. Buffers are recycled as soon as a
  node's consumers are done, so a constraint needs about as many live buffers as
  the tree is deep, not as many as it has nodes. This is the fix for the
  allocation churn.
- **Scalar**: `ConfigSpace::isValid` keeps its current shape — build a one-lane
  `ConfigurationVector` and run the vectorised path. There is exactly one
  evaluator, so the two cannot disagree.

Constraints registered as opaque lambdas keep their current treatment: a
`VecConstraint` is stored as-is, a scalar `Constraint` is wrapped in a per-lane
loop (`ConfigSpace::addConstraint`).

## Implementation steps

Each stage is independently landable and independently measurable. The lit tests
in `test/Dialect/UPMEM/upmem-infer-accelerator*.mlir` cover the generic and
template lowering paths, `eval-solution` (which exercises the scalar
`isValid`/`debugIsValid` path), and constraint *rejection* — run them at every
stage.

**Stage 1 — n-ary product node, existing DSL.**
Add `NAryExpr<Op, Elem>` holding `std::vector<Elem>` plus a `prod(range)` helper,
and give each `Op` an `identity()`. Rewrite A1 in the DSL and delete the
hand-written `VecConstraint` for it. No analysis yet — it still becomes a dynamic
predicate, but an auto-vectorised one. *Validates the surface syntax and proves
the auto-vectorisation goal on a real constraint.*

**Stage 2 — introduce the runtime IR.**
Add `ConstraintNode`; make the DSL operators build nodes; replace the typed
`evalVec` with a tree interpreter. All constraints stay dynamic. *Pure
refactor — the valid-configuration count must not change. Assert that against a
recorded baseline.*

**Stage 3 — scratch-buffer pool in the interpreter.**
Expected to recover most of the page-fault overhead. *Measure against the
profile that motivated it; no functional change.*

**Stage 4 — normaliser + Form A recogniser.**
Canonicalise to product-of-factors, detect `prod == prod`, apply the direction
rule, and report what it found under `--debug-only=cinm-inference` — without
acting on it yet. *Lets the analysis be validated on real spaces before it can
change any results.*

**Stage 5 — `SolvedGroup` encoding.**
Precompute solution tables, wire into `ensureEncoding`/`at()`/`indexOf()`, add
the fallbacks. *This is where the 1.24e9 → ~1e5 reduction lands. Verify the set
of valid configurations is unchanged: for a small space, enumerate both ways and
compare.*

**Stage 6 (optional) — Form B beyond parent/child, and Form C domain narrowing.**
Only if measurement says they matter after Stage 5.

## Open questions

- **Where does `readAsGemvTemplate` fit?** A2 is only visible if the template
  layout derivation is expressed in the DSL rather than as a C++ helper behind
  `derive(c).has_value()`. Rewriting it is a bigger change than the rest of this
  document and may not be worth it while `lowering=generic` is the path being
  optimised.
- **Does `nValid` grow enough to make the sampler hot?**
  `CandidatePool::sampleInitialSet` does greedy nearest-neighbour selection at
  `O(sampleN × nValid × D)` on the calling thread. At `nValid = 3038` it is
  negligible; if a better encoding raises `nValid` substantially, it becomes the
  next bottleneck. Re-measure after Stage 5 rather than pre-emptively
  parallelising it.
- **Interaction between groups.** Two Form A constraints sharing a variable
  cannot both become groups. Current plan is first-wins plus fallback; whether a
  smarter choice (prefer the group that prunes more) is worth it is unknown
  until we see a space with more than one.

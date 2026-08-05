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
  enum class Kind { Const, Var, Add, Sub, Mul, Div, Cmp, Implies };
  Kind kind;
  // Const
  ParmValue value;
  // Var: index into ConfigSpace::params, resolved at buildInto() time
  std::shared_ptr<size_t> varIdx;
  // Cmp
  CmpKind cmp;
  // Children: n-ary for Add/Mul, binary for Sub/Div/Cmp/Implies
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
  divides `a` exactly. That assertion is *not* part of the node — it is
  registered separately by `extractDivConstraints`, which is the subtlety the
  next section is about.
- **`Var` holds the same `shared_ptr<size_t>` cell `SpaceVar` uses**, so handles
  keep working across `buildInto()` and node identity is pointer identity.
- **`Cmp` and `Implies` are the boolean kinds** (`isBoolKind`). `Implies` is the
  only connective: conjunction needs none, since two `require` calls are an
  `and`, and disjunction has no caller. The surface DSL splits the two worlds
  statically as `Expr<Type::INT>` and `Expr<Type::BOOL>`, so a comparison cannot
  be an operand of `*` and an integer cannot be an operand of `implies`.

### Normal form

**Nothing rewrites the tree.** This section originally described a
canonicalisation pass — flatten, constant-fold, sort, rewrite `a / b` — and no
such pass was built, deliberately. The tree `require()` is handed is the tree
the interpreter evaluates, verbatim, for the whole life of the space. What
exists instead is a *reading*: `normalizeImpl` computes a canonical form on
demand, returns it, and leaves the IR alone.

That distinction is not pedantry. A reading can disagree with the tree it reads,
and here it does; see *The division contract* below, which is the single most
important thing to know before writing a constraint.

The reading is defined only for **arithmetic** nodes, and only reaches Form A.
Every node reduces to a `Rational` — a pair of `Monomial`s, `numer / denom` —
where a `Monomial` is `coeff * prod(vars)` with `vars` a *sorted multiset* of
parameter indices (sorted, so `dpus * tasklets` and `tasklets * dpus` are the
same monomial; a multiset, so `x * x` is representable and simply not solvable):

| node | reading |
|---|---|
| `Const v` | `numer.coeff = v` |
| `Var i` | `numer.vars = {i}` |
| `Mul(a…)` | multiply the numerators, multiply the denominators |
| `Div(a, b)` | cross over: `numer = a.numer * b.denom`, `denom = a.denom * b.numer` |
| `Add`, `Sub`, `Cmp`, `Implies` | **no reading** — analysis fails |

That last row is load-bearing: a sum is not a monomial, and refusing it is
exactly what keeps capacity bounds (Form C) out of Form A rather than silently
mis-analysed as one.

`matchProductEquality` then reads a `Cmp` whose `cmp` is `Eq` and whose sides
both have readings, and clears the denominators by cross-multiplying:

```
lhsNum/lhsDen == rhsNum/rhsDen    ⟿    lhsNum * rhsDen == rhsNum * lhsDen
```

So **yes**: `extent / block == 1` reads as `extent == block`. And the tile-count
constraint, which is written

```cpp
prod over dims of (extent[d] / block[d])  ==  dpus * tasklets
```

reads, on `prim_gemv`'s 4MB gemv, as

```
1048576 == dpus * tasklets * gemv.M.mram * gemv.K.mram
```

— every constant folded into one coefficient, every block size moved to the
other side by cross-multiplication, the variables sorted. That is the form
`ComponentEnumerator::solveFor` consumes.

Form C never has a normal form at all. `boolMayHold` / `boolMustHold` walk the
raw tree with interval arithmetic (`evalNodeBounds`), which is why they handle
the `Add` and `Sub` that the reading refuses.

### The division contract

**The reading of a `Div` is not what the interpreter computes.** `evalNodeVec`
lowers `Div` to truncating integer division, guarded against a zero divisor
(`vecSafeDiv`, matching the old `b ? a / b : 0`). So for `extent = 1024`,
`block = 768`:

| | verdict |
|---|---|
| the tree, evaluated: `1024 / 768 == 1` | **true** — accepts |
| its reading: `1024 == 768` | **false** — rejects |

The reading is *strictly stronger*. The two agree only because `require()` walks
every tree it is given and reifies each `Div` it finds as a separate
divisibility constraint — `extractDivConstraints` → `addDivConstraint`, landing
as a static domain filter, a structural `mustDivide`, or a dynamic predicate,
whichever the operand shapes allow. With `block | extent` also enforced, the
truncating case cannot arise and the disagreement is unreachable.

So the contract is:

> A `Div` in the DSL means "divides exactly". The tree alone does not say that;
> the side condition extracted alongside it does. **Both must be enforced, or
> the reading and the evaluation part ways** — and they part ways silently,
> because each is individually plausible.

This matters because the two are *not* enforced by the same machinery. The
component enumerator absorbs the Form A reading and drops the predicate; the
unabsorbed path evaluates the tree. Neither is wrong so long as the side
condition survives.

**Which is why an implication may not divide.** Everything
`extractDivConstraints` reifies is unconditional, so a `Div` under a guard would
impose its divisibility on precisely the configurations the guard exists to
exclude — and *not* extracting it leaves the truncating reading of the tree in
force, which is the silent-wrong-answer case above. `extractDivConstraints`
therefore refuses to descend into an `Implies` and asserts that neither side
divides. Write the multiplied-out form:

```cpp
b.require(implies(fuse >= 1, extent / block == 1));  // rejected
b.require(implies(fuse >= 1, block == extent));      // say this
```

which costs nothing, since the multiplied-out form is what the analyser was
going to read anyway. See [LaunchFusionDesign.md](LaunchFusionDesign.md) §F.

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

**Example C1** — `handleLinalgOp`, the MRAM/WRAM footprint checks, now built
from the DSL by the `footprint` lambda:

```cpp
sum over operands of (r_O * prod over that operand's dims of block[d])
    <=  mramElements
```

where `r_O` is the operand's replication factor — see §H5 below, which is the
whole subject of how tight this bound can be made.

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

#### §H5 — tightening the capacity bounds: replication factors

A capacity bound is only useful in proportion to how tight it is, and the
version above is loose in a specific, fixable way. It charges each operand
*one* tile per DPU — maximal sharing, every leaf on a DPU reading the same
tile. That is the most optimistic assumption, which is what makes the bound
sound, but it can be far from what any lowering could achieve.

The quantity being approximated is the **replication factor** `r_O`: how many
distinct tiles of operand `O` one DPU must hold at once. The footprint is
`sum over operands of r_O * tileSize_O`, and maximal sharing is just `r_O = 1`.

**The general bound.** Let `S_O` be the iteration dimensions that index `O`, and

```
T_O = prod over d in S_O of (extent[d] / block[d])
```

the number of *distinct* tiles of `O` across the whole workgroup. Two leaves can
share a tile of `O` only if they agree on every coordinate in `S_O`. Those `T_O`
tiles are spread over `dpus` DPUs, so by pigeonhole some DPU holds at least
`T_O / dpus` of them:

```
r_O  >=  max(1, T_O / dpus)
```

This is **independent of the workgroup order**, which is what makes it usable
at space-construction time: it holds for every assignment of leaves to DPUs, so
it never has to know which dimension varies fastest.

**The exact special case — implemented.** When `S_O` is *all* iteration
dimensions, no two distinct leaves can share `O` at all: they differ somewhere,
and that dimension selects a different tile. So `r_O = tasklets`, exactly, not
merely as a bound — a DPU has exactly `tasklets` leaves and each needs its own
tile. This is a syntactic test on the indexing map (`dims.size() == numLoops`
for a projected permutation) and needs no new IR machinery, so it is what
`handleLinalgOp` implements today, for both the MRAM and the WRAM bound.

It is also the case that matters most: in a matvec it is the matrix, and in a
matmul both inputs, i.e. the terms that dominate the footprint. Cross-checked
against `registerGemvTemplate`, which models its own fixed layout by hand and
charges `mramRow * mramCol == tasklets * blockM * blockK` for A in MRAM and
`t * wramRow * wramCol` in WRAM. The formula reproduces both exactly.

**Where the general bound stays loose.** For `x[k]` in a gemv, the pigeonhole
term is `max(b_k, (K/b_k)/dpus * b_k)` while the template actually charges
`blockK * taskletCols`. The gap is the DPU-level *replication* of `x` across
DPU-rows, which no counting argument can see — pigeonhole bounds distinct
tiles, not copies. Closing it requires knowing `order`, since it is the order
that decides whether the tasklets on a DPU span the `m` fibre or the `k` fibre.
So the refinement is deliberately uneven: exact where the footprint is
dominated, `r_O = 1` elsewhere.

**Fitting the general form in the IR**, if it is ever implemented:

- `max` is not in the node set, but `sum_O max(a_O, b_O) <= C` is equivalent to
  the conjunction of the `2^|operands|` constraints obtained by picking one side
  per operand — the max selection is among them and is the strongest. Three
  operands is 8 sum-of-monomial constraints, each monotone, so `evalNodeBounds`
  and `cmpMayHold` prune them with no new machinery.
- `T_O / dpus` truncates downward in integer arithmetic, which only weakens the
  constraint. Truncation is in the sound direction here.
- For the WRAM bound the two levels mix: `r_O` is still driven by the *block*-
  level tile counts (blocks define leaf identity) while the size term is the
  *leaf* tile, so the result is a genuine mixed monomial rather than collapsing
  to a constant.

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

### Validation baselines

Recorded on `experiments/cinm1comparison/data/prim_mtv/_split/`, comparing the
sorted parameter columns of `pool.csv` (`dump-full-pool=true`), so costs do not
pollute the diff. Any stage that is meant to be behaviour-preserving must
reproduce these sets *exactly*, not merely the counts.

| input | enumerated | valid | sha256 (first 16) |
|---|---|---|---|
| `mtv_4MB`   | 428,212,224   | 17,738 | `d53f483d9ff22f23` |
| `mtv_64MB`  | 814,055,424   | 12,364 | `268031464cd7ea88` |
| `mtv_512MB` | 1,238,630,400 |  3,038 | `685bcdde4ba0f4c1` |

### Stage 1 — n-ary product node — **DONE**

Folded into Stage 2: a type-level `NAryExpr` would have been deleted immediately
by the IR migration, so the n-ary product was built directly into the IR and
exposed as `prod()` / `sum()`. A1 is now written in the DSL
(`prod(tilesPerDim) == dpus * tasklets`) and its hand-written `VecConstraint` is
gone — the vectorised implementation is generated from the tree.

### Stage 2 — runtime IR — **DONE**

`ConstraintIR.h`/`.cpp` hold the node set; the DSL operators build nodes;
`SpaceBuilder`'s CRTP templates (`SpaceExprBase`, `BinExpr`, `ConstraintExpr`,
the `Op` structs and ~30 operator overloads) are gone. `extractDivConstraints` /
`addDivConstraint` migrated from `if constexpr` dispatch to switches on node
kind, preserving all three cases. All three baselines reproduce byte-identically.

### Stage 3 — scratch-buffer pool in the interpreter — *deferred*

`evalNodeVec` allocates a fresh `ParmVector` per node, which is the allocation
churn behind the page-fault profile. Treated as an optimisation to apply after
the encoding work, since the encoding change removes most of the work being
done in the first place. *Measure against the profile that motivated it; no
functional change.*

### Stage 4 — normaliser + Form A recogniser — **DONE**

`normalizeRationalMonomial` reduces an arithmetic node to
`(coeff * prod(vars)) / (coeff * prod(vars))`, failing on anything containing
`Add`/`Sub` — which is precisely what keeps capacity bounds (Form C) out of Form
A instead of silently mis-analysing them. `matchProductEquality` cross-multiplies
an `Eq` to clear denominators:

```
lhsNum * rhsDen  ==  rhsNum * lhsDen
```

`SpaceBuilder::reportConstraintAnalysis` runs after phase 1 of `buildInto` (when
variable indices exist) and reports under `--debug-only=cinm-inference`. It does
not change the space; all three baselines reproduce exactly.

**What it finds on `mtv_512MB`:**

```
[cinm-analysis]   Form A: 134217728 == dpus * tasklets * gemv.M0 * gemv.K0
[cinm-analysis]     joint cardinality: lhs=1 rhs=10321920
```

`134217728 = 8192 * 16384 = 2^27`. This is a sharper result than assumed when
this document was first written: after normalisation A1 is not
*variables == variables* but **constant == product of all four variables**. The
consequences for Stage 5:

- There is no "enumerate one side, derive the other" split imposed by the
  equation itself. All four dimensions are *jointly* constrained, and any
  partition into key/solved enumerates the same solution set `S`. The choice is
  therefore about precomputation cost and wasted key combinations, not about the
  size of the result.
- The degenerate partition — empty key, one solution list holding all of `S` —
  is optimal in table size and simplest to build. `S` is exactly the tuples the
  encoding should offer.
- The four dimensions contribute `2048 * 24 * 14 * 15 = 10,321,920` to the flat
  product today. Replacing that factor with `|S|` is the whole win.

The direction rule in the section above still applies to the general
*variables == variables* case (Form A2, `dpus == dpuRows * dpuCols`), but is not
what drives A1.

### Stage 5 — component enumeration

**The encoding is a planning problem, not a pattern rewrite.** The original
sketch here — recognise one constraint, form a group keyed on one side — is too
local. `SpaceBuilder::divisorsOf(name, SpaceVar)` commits to a pairwise
`DependentGroup` the moment a leaf is declared, before the product equality that
also constrains those variables has even been registered. Encoding decisions
must be made once, over the whole constraint set.

The replacement:

1. Collect every structural constraint (divisibility and product equality)
   without acting on it.
2. Build a graph over search variables, with an edge for every constraint that
   mentions two variables.
3. Take connected components. Variables in different components are
   independent and keep their own slots.
4. Enumerate each component's solution set `S` — the tuples satisfying all of
   its constraints — and give the component **one slot of size `|S|`**.

This subsumes `DependentGroup`, which is exactly the two-variable case
(`|S|` = number of valid parent/child pairs).

The flat index stays mixed-radix, so every slot must keep a fixed size; one slot
per component satisfies that by construction, since `|S|` is a constant once
enumerated.

**The flat index is an identity, not a representation.** Its job is to let a
configuration be referenced as a `size_t` — in `validIndices`, `visited`, the
per-index cost maps — without passing `Configuration` objects around. The
mixed-radix arithmetic is an implementation detail of that, not something worth
preserving for its own sake: encoding and decoding on every access is fine.

Two things follow, both simplifications:

- `at()` over a materialised component is a list lookup and `indexOf()` a hash
  lookup, both cheaper than the mixed-radix decode they replace. The
  `forEachChunk` incremental-suffix walk exists specifically because `at()` is
  `O(slots)` today; that motivation largely disappears.
- Any operation that wants *semantic* structure (neighbourhood, perturbation)
  should decode, work on the `Configuration`, and re-encode — rather than doing
  arithmetic on the index and hoping it corresponds. See `neighborIndices()`
  below for what happens when it does not.

The one property the index must keep is **stability**: enumeration order has to
be deterministic across runs, or seeded sampling stops being reproducible.

**Enumeration is a tree; storage is a list.** These are separate concerns and
conflating them makes the design look harder than it is:

- *Enumerating* `S` is a backtracking walk with variable elimination. Resolving
  `C == v1 * v2 * ... * vk`, each level offers only the divisors of what
  remains: pick `M0 | C`, then `K0 | C/M0`, then `tasklets | C/(M0*K0)`
  intersected with its declared domain, at which point `dpus` is *determined*;
  then `M1 | M0` and `K1 | K0`. No level can offer a value that fails to
  complete, so the cost is `O(|S| * k)` rather than the product of the domains.
- *Storing* `S` needs no tree at all. A flat list of tuples makes `at()` an O(1)
  lookup, and a tuple→index hash map alongside gives `indexOf()` as its exact
  inverse. At `|S| = 46,820` over 6 variables that is ~1.1 MB.

**Bonus: Form C prunes during enumeration.** Capacity bounds cannot determine a
variable, but they can kill a subtree — if a partial assignment already busts
MRAM, no completion of it will fit. That turns them from post-filters into
enumeration cutoffs, which is where the gap between `|S|` and the final valid
count gets closed cheaply.

#### Worked example: `mtv_512MB`

Current slots, confirmed against the reported enumeration
(`2048 * 24 * 105 * 120 * 2 = 1,238,630,400`):
`dpus(2048)`, `tasklets(24)`, `[M1,M0](105)`, `[K1,K0](120)`, `order(2)`, where
105 and 120 are the divisibility pair counts `sum(i+1 for i in 0..13)` and
`sum(i+1 for i in 0..14)`.

The constraint graph has edges `M1—M0`, `K1—K0` (divisibility) and
`dpus—tasklets—M0—K0` (the product equality), giving **one component of six
variables**, plus `order` standalone. Enumerating that component:

| | |
|---|---|
| `\|S\|` over `{M1, M0, K1, K0, dpus, tasklets}` | **46,820** |
| times `order(2)` | **93,640** |
| currently enumerated | 1,238,630,400 |
| **reduction** | **13,228x** |
| valid after the remaining (capacity, order) constraints | 3,038 |

So the structural encoding leaves 93,640 configurations for `computeValidMask`
to filter instead of 1.24e9 — at which point the vectorised constraint
evaluation stops being on the critical path at all.

#### Implementation shape

1. Make constraint collection lazy: `divisorsOf(name, SpaceVar)` records a
   relation, it does not commit to a group.
2. Build the variable graph, take connected components.
3. Per component, choose an elimination order and enumerate `S` by backtracking,
   applying Form C bounds as subtree cutoffs.
4. Store `S` as a flat tuple list plus a tuple→index map; the component becomes
   one slot of size `|S|`.
5. Wire through `ensureEncoding` / `at()` / `indexOf()` / `neighborIndices()` —
   all four must agree, and `indexOf` is the inverse `eval-solution` depends on.

**Guards** (each falls back to dynamic predicates, never to a wrong answer):

- `|S|` cannot be checked before enumerating it, so enumerate under a hard cap
  and abandon the component if exceeded.
- Ordering is a heuristic — optimal elimination order is NP-hard in general.
  Dead-end-free enumeration is achievable when every constraint in the component
  is a divisibility or product relation, which is the case here, but the
  enumerator must not assume it.
- A variable appearing on both sides of an equality determines nothing.

**`neighborIndices()` needs rewriting — but it is already wrong.** It is
documented as returning "all flat indices one discrete step away in any
dimension", and implemented as ±1 on each *slot* sub-index using stride
arithmetic:

```cpp
size_t stride = suffixProd_[si + 1];
size_t subIdx = (idx / stride) % slots_[si].slotSize;
if (subIdx > 0)                        result.push_back(idx - stride);
if (subIdx + 1 < slots_[si].slotSize)  result.push_back(idx + stride);
```

For an ungrouped dimension a slot step *is* a dimension step. For a
`DependentGroup` slot it is not: `at()` decodes the sub-index into
`(parentSubIdx, childLocalIdx)`, so `subIdx + 1` normally advances the child,
but at a `cumCount` boundary it advances the parent and resets the child —
changing two dimensions at once. So the documented invariant already does not
hold for grouped dimensions.

Component enumeration therefore does not break an invariant; it degrades an
approximation that is already approximate. What does get worse is neighbourhood
*quality*: a `[M1,M0]` slot of size 105 makes ±1 "usually one step in M0",
whereas a component slot of size 46,820 makes it an arbitrary jump through a
six-dimensional subspace determined by enumeration order. That matters to BO's
`fillNeighbors`, which is trying to sample locally.

The fix is independent of this work and would improve things today: implement
`neighborIndices` as *decode → perturb one dimension → `indexOf`* rather than
stride arithmetic. That is encoding-agnostic, makes the function mean what its
comment says, and costs one `indexOf` per neighbour — cheap once the
tuple→index map exists. Exhaustive and random-sample paths never call it.

*Verification: the three baselines above must be reproduced exactly. A space
this size cannot be enumerated both ways, so the check is the valid-set diff,
plus `indexOf(at(i)) == i` over a sample of indices.*

### Stage 6 (optional) — Form B beyond parent/child, Form C domain narrowing

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

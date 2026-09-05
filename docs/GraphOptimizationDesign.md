# Design: graph-level DPU allocation and scope merging for the UPMEM backend

Status: 2026-08-11. The prototype (T1–T5 of §4) is implemented; §7 records
what exists and what remains. The document stands on its own.

## Background: the per-operator schedule model this builds on

Facts about the target that this design leans on:

- A DPU owns 64 MiB MRAM and 64 KiB WRAM; the host scatters array slices into
  MRAM banks and gathers results back; tasklets stage MRAM tiles into WRAM by
  explicit DMA. Nothing is cached; every placement is explicit.
- The SDK allocates DPU *counts*. Which ranks the DPUs land on is its
  business and unobservable (some ranks even carry defective DPUs and are
  smaller than their nominal 64), and measured transfer cost is a function of
  the DPU count alone — so ranks are not a dimension of anything in this
  design, and DPU-set sizes are plain integers.
- Loading a program onto a DPU array takes ~40–80 ms — 50–100× a typical
  kernel execution — and scattering weights similarly dwarfs kernel runtime.
  (Measure both on the real system early; the whole design prices these.)
- The tasklet count `T` (`NR_TASKLETS`) and all loop bounds are compile-time
  constants of the DPU binary.

Per-operator schedule model (using MMTV, `y_bm = Σ_k A_bmk · x_bk`, as the
running example): each dimension `d ∈ {B, M, K}` with extent `N_d` is split
into four tiling factors `h_d · p_d · u_d · w_d = N_d`, read outside-in:
`h_d` tiles scattered across DPUs, `p_d` tiles distributed across tasklets,
`u_d` tiles traversed sequentially per tasklet, `w_d` the extent held in WRAM
at a time. Write `m_d = p_d · u_d` for the combined MRAM-level factor. With
`D` = DPU count and `T` = tasklets per DPU, a valid single-operator schedule
satisfies:

    P1:  Π_d h_d = D                                  (one host tile per DPU)
    P2:  Π_d p_d = T                                  (one MRAM share per tasklet)
    P3:  N_d = h_d · p_d · u_d · w_d   for each d     (factors divide the extent)
    P4:  T · wram(w_B, w_M, w_K) ≤ C_WRAM             (per-tasklet working set; the
                                                       u_d sequential tiles reuse one buffer)
    P5:  mram(m_B·w_B, m_M·w_M, m_K·w_K) ≤ C_MRAM     (per-DPU working set)
    P6:  D_min ≤ D ≤ D_max,  T_min ≤ T ≤ T_max

The compiler derives P1–P6 per operator from the IR, solves them with Gecode,
and searches the feasible set with BaCO-style Bayesian optimization; a
candidate is evaluated with no hardware in the loop (kernel-IR simulation +
analytical transfer costs), so evaluation is cheap.

The problem this document addresses: an inference workload is a *graph* of
such operators, and optimizing each in isolation greedily takes
`D = D_max` for every one — which makes each operator's weights evict the
previous one's, forces program reloads between operators, and thereby
destroys the static-weights/static-program assumption that UPMEM
competitiveness depends on.

## 0. The framing insight: residency, not concurrency

The naive reading is that `D_1 + D_2 <= D_max` is a *concurrency* constraint —
ops running at the same time must not oversubscribe the grid. That reading is
wrong for UPMEM, and getting this right simplifies everything downstream.

Once an op pins weights (and its program) on a DPU set, it occupies that set
for the *lifetime of the serving process*, not for the duration of one kernel
launch. A sequential successor cannot "borrow" those DPUs without evicting the
weights and program, which is precisely the staticity violation at issue.
So the grid-capacity constraint

    C7:   Σ_s D_s <= D_max        (sum over DPU sets, i.e. partition blocks)

applies to **all pinned ops in the graph, sequential or parallel alike**. The
dependency structure of the graph affects only the *objective* (what waits for
what), never the *feasibility* of an allocation. This is the fact that makes a
clean two-level decomposition possible.

Consequently "scope merging" is exactly: **a partition of the DPU grid into
sets, and an assignment of offloaded ops to sets.** Two ops assigned to the
same set share program and MRAM; ops in different sets own their DPUs
exclusively.

## 1. Graph-level decision variables

The unit the allocation runs over — "the graph" — is a connected component
of the dataflow between offloadable regions targeting one platform: two
regions belong together when a value flows from one to the other or they
read a common producer. Control flow is not interpreted; the relation is a
deliberate over-approximation, and errs in the sound direction — the device
is shared by everything pinned on it, so a missing edge would let two graphs
hand out the same DPUs, while a spurious one only makes the allocation
problem bigger. Each component is treated as owning the whole device.

For each offload candidate `i` (a region the frontend's platform-assignment
step marked as offloadable):

- `o_i ∈ {host, device}` — offload decision. Prototype: decided up front by
  an Ideal Arithmetic Intensity ranking (operations per element of
  transferred data, discounting transfers that pinning would amortize — e.g.
  gemv with a static matrix scores `M`, with a dynamic matrix `1/K`); not
  searched. A full marginal-latency treatment of host-vs-device placement is
  out of scope; trust the backend's offloading heuristics.
- `s_i` — the DPU set op `i` is assigned to.

For each set `s`:

- `D_s` — its size in DPUs, drawn from a small per-class menu of candidate
  sizes derived from the problem itself (see Stage A): the counts that can
  divide the op's iteration space at all, thinned to a bounded list. A
  configurable granularity quantizes the menu — a *preference* for
  round-numbered allocations, not a hardware constraint, since the SDK
  accepts any count.

"Merging" is not a separate variable: it is implied by `s_i = s_j`.

### What counts as static

The whole design prices *residency*, so it needs a sound notion of which data
stays put. An operand is **static** when it holds the same data on every
inference, so that pinning it on a DPU set amortizes its transfer over the
serving lifetime — model weights, canonically. Staticness is derived, not
guessed: an operand is static iff it is

- a program input the frontend has declared static (the serving contract:
  weights are baked at deployment; an annotation on function parameters),
- a compile-time constant,
- a view (slice/subview) of a static value at compile-time-constant
  offsets and strides — the "same window of the same tensor" case; a
  dynamically-indexed view of static data is *not* static, since the data
  moved per inference varies — or
- a type cast of a static value: same data under another type.

The classification propagates into offload regions through their operands.
Two constraints consume it: C9 splits its capacity charge on it (static
footprints of co-residents sum, dynamic working sets take the max), and C8's
signature includes the per-operand staticness pattern — two shape-identical
ops whose matrices differ in staticness compile to the same binary but pay
different per-inference transfer, so their cost profiles `L(D)` differ and
they must not alias.

### Cross-op validity constraints

- **C7 (grid capacity)**: `Σ_s D_s <= D_max`. As argued above, over all pinned
  ops, not just concurrent ones.
- **C8 (program identity)**: `s_i = s_j ⇒ program_i = program_j`.
  Program identity means *every compile-time parameter is equal*: operator
  kind, shapes, tiling factors, **and the tasklet count `T`** (`NR_TASKLETS`
  is baked into the binary, like the loop bounds). The prototype implements
  this by *aliasing the two scopes' parameter vectors into one* — which
  covers `T` with no special case. This is the key trick: merging *shrinks*
  the joint search space (n ops, one space) instead of cross-producting it.
  Do **not** implement identity as "run one search per identical op and
  expect the same argmin": BO is stochastic and near-optimal ties are common,
  so independent searches on identical spaces may return different points and
  silently break identity. Instead canonicalize scopes by signature
  (op kind, shapes, dtype, and the per-operand staticness pattern defined
  above), search once per equivalence class, and stamp the winning point onto
  every member — deterministic, and it cuts Stage-A cost. Canonicalization
  happens *before* profiling: the class, not the op, is the unit the whole
  two-level solve operates on, so a class is profiled once regardless of its
  multiplicity (QKV costs one profile, not three) and §2's outer solve gets
  its factored structure from the classes directly.
  Extension (post-prototype): a shape-dynamic kernel taking tiling factors as
  runtime arguments relaxes identity to "same operator kind + same `T`" —
  `T` stays compile-time even then. The shared-parameter interface of a
  merged set then grows from `D_s` to `(D_s, T_s)`: per-op tilings stay free
  under their own P1–P5, profiles become `L_i(D, T)` (the `T` menu is
  small, ~8–16), and the outer separation argument goes through unchanged.
- **C9 (co-residency capacity)**: P5 becomes, per set,

      Σ_{i : s_i = s} mram_static_i(m_•, w_•)  +  max_{i : s_i = s} mram_dyn_i(...)  <= C_MRAM

  Static (pinned) footprints **sum**; dynamic working buffers of sequentially
  executing co-resident ops reuse one region and take the **max**. The max is
  what makes packing nearly always succeed (aggregate grid MRAM is orders of
  magnitude larger than typical weight sizes), but it must be an explicit
  constraint, not an assumption — that is the point of the whole exercise.
  The constraint is stated *per memory level* of the platform, and the right
  default is to bound every level: a level in which nothing persists between
  kernels (WRAM, whose tiles are re-staged by DMA on every use) has zero
  static footprint for every op, and its max-of-dynamic bound is already
  implied by each op's own per-op feasibility — so scratch levels are
  automatically non-binding and no per-target selection of "the" pinning
  level is needed. On UPMEM the binding instance is MRAM, as written above.
- **C10 (layout coupling — extension, not prototype)**: a producer/consumer
  pair on the same set may keep the intermediate on-device iff the consumer's
  scatter map for that operand equals the producer's gather map. Guarded by a
  boolean via the constraint system's implication form (constraints that
  apply only when a guard holds). Note that when the guard is on,
  it *aliases* the h-factors of the shared tensor's dimensions across the two
  scopes — again removing degrees of freedom, not multiplying spaces.

### Objective

Throughput: sets run concurrently, ops within a set run sequentially,
inferences pipeline across sets. Steady-state:

    minimize   max_s ( Σ_{i : s_i = s} L_i(D_s) )   +  inter-set transfer terms

**Dependencies do not enter this objective.** In steady state, inferences
stream through the graph and every set is a pipeline stage: per inference it
must execute the ops assigned to it in some dependency-respecting order, and
the total work is Σ L_i(D_s) — a sum no ordering changes. With sufficient
buffering between stages, the steady-state rate is the reciprocal of the
busiest stage's per-inference work, whatever the edge structure. Edges
determine pipeline *depth* — how many inferences are in flight before the
pipeline saturates, i.e. latency — never where the bottleneck is. The
dependency graph's only remaining roles under throughput are scoping (which
ops compete for one grid) and the transfer terms; the allocation itself is a
pure min–max resource split. The caveat is the host-bus limitation already
stated: stages share the host during transfers, so they are not perfectly
decoupled; second-order, not modeled.

This outer problem is **moldable-task scheduling** (jobs whose runtime depends
on the processor count they are allotted) with compatibility (C8) and capacity
(C9) side constraints — a known problem family, NP-hard in general. §2 shows
that under the throughput objective our instance is not merely small but
*polynomial*: C8 factors the assignment and the min–max form admits an exact
parametric solve.

### The latency variant

Single-inference latency is the makespan of one inference's DAG: node `i`
costs `L_i(D_{s_i})`, ops sharing a set serialize, ops on different sets
overlap wherever the DAG allows. Two things change against throughput.
Dependencies now enter the objective — it is a longest path, not a max of
sums, and DPUs allotted off the critical path buy nothing. And **merging
acquires a cost**: co-resident ops serialize even where the DAG would let
them run at once, so merging QKV keeps one program and one weight residency
but triples that segment of the path. Class members consequently stop being
interchangeable — *which* members share a set depends on where they sit in
the DAG — so the factoring that makes the throughput solve exact does not
survive.

This design does not chase optimality here; a static allocation that
produces a defensible number is the goal. The method is the standard
critical-path greedy of the mixed task/data-parallel scheduling literature
(CPA / CPR): start from the cheapest feasible allocation, repeatedly apply
whichever move buys the largest makespan reduction per additional DPU, and
stop when nothing improves. Two moves suffice — *grow* a set to the next
size on its menu, and *split* a member out of a set into one of its own —
and scoring both by Δmakespan/ΔD keeps the merging decision inside the same
loop instead of giving it a phase of its own. Every accepted move strictly
increases the DPUs in use, so the loop terminates against the grid budget.

**Residency makes this simpler here than in its usual setting.** Classical
moldable scheduling time-shares processors — a task releases them when it
finishes — so its allocation phase must balance the critical path against
total processor-time *area*, and a list-scheduling phase must then pack
tasks over time. Under residency nothing is ever released: the constraint is
the flat `Σ D_s ≤ D_max`, every set is a dedicated machine, and evaluating a
candidate allocation is one longest-path computation over the DAG augmented
with the serialization edges inside each set. The area term and the packing
phase both disappear. On a chain the critical-path filter is vacuous — every
node is on it — and the loop degenerates to textbook marginal resource
allocation: spend each next DPU wherever it buys the most. That is exactly
optimal for convex profiles and a heuristic otherwise; ours are not convex,
since divisibility puts holes in the menu.

**Merging is free along a chain**, which is what makes this interesting on a
real model. Layer `k`'s Q projection and layer `k+1`'s Q projection are the
same program-identity class *and* sequentially dependent: they never run
concurrently, so putting every layer's Q projection on one set costs exactly
zero latency while dividing that class's budget by the layer count. The
greedy reaches this by construction, because it starts from maximal merging
— one set per class, at the smallest size that holds all its members — and
splits only where splitting pays. The latency-driven static allocation of a
transformer is therefore *one set per distinct kernel, with every layer's
weights co-resident on it*: the same residency story §0 argues for, arrived
at from the opposite objective.

Everything else is shared with the throughput path: the same profiles, the
same menus, the same capacity terms. Only the objective evaluator differs.
Timesharing is not offered as an option under latency — it can only lengthen
the path — so a graph whose classes do not all fit at their smallest sizes
is reported infeasible rather than partially evicted.

## 2. The two-level solve

**Never search the cross-product of per-op spaces with BO.** The decomposition:

### Stage A — per-class profiling

For each signature class `c` (canonicalization precedes profiling; see C8)
and each `D` in the menu, run the *existing* single-op search with P6
pinned: `D = D_menu[k]`. Record, per (class, D), the complete outer-facing
summary:

    L_c(D)   = best cost found,
    plus     the argmin configuration,
    plus     the static and dynamic footprints at that argmin,
             per memory level (C9's terms),
    plus     the cost of re-scattering the static operands once
             (what a non-pinned placement pays per inference; the
             dynamic operands' per-inference transfer is already
             inside L).

The footprint and re-scatter fields are what let C9 and the timeshare
pricing (§2) be evaluated by the outer solve without ever touching the IR or
the backend again: the profile is the entire interface between the levels.

**The menu is derived from the problem, not the hardware.** The workgroup
must be filled exactly (P1–P3), and `T = 1` is always admissible, so the
divisibility-feasible DPU counts of one op are exactly the divisors of its
iteration-space size (any divisor of a product of extents factors into
per-extent divisors); for a multi-op region, the divisors of the sizes' gcd.
Among those, the menu prefers multiples of a configurable granularity, falls
back to plain divisors when the problem size admits no such multiple (a
non-power-of-two problem would otherwise get an empty menu), caps at the
device size, and thins to a bounded, roughly geometrically spaced list.
Divisibility is necessary, not sufficient — a menu value the pinned search
still finds infeasible (capacity, say) simply yields no point, and the
profile has a hole there.

- Affordable precisely because evaluation is device-free.
- Embarrassingly parallel: the menu points are independent searches and run
  concurrently; deduplication across class members is free — multiplicity
  costs nothing at this stage.
- Possible refinement (not implemented): one search with D free visits many
  D values; harvest best-per-D from its trace and only top up under-sampled
  D values with short pinned runs.
- No monotonicity assumption: divisibility (P1, P3) makes L_c(D)
  non-monotone; the profile is measured pointwise, so that's fine. (The
  monotonicity the outer solve *does* use is in the objective — feasibility
  of a bottleneck target, not the profile itself.)

### Stage B — exact outer solve

Solve for `{s_i, D_s}` against C7–C9 with the profile-based objective. Naive
partition enumeration is Bell(n) and dies long before real networks
(Bell(20) ≈ 5·10¹³); it is also unnecessary, because the problem has two
exploitable structures.

**C8 factors the assignment.** A set must contain identical programs, so
every set lives inside one signature class, and the global partition
decomposes into independent groupings of each class's members. Identical
members are interchangeable, so within a class only the multiset of group
sizes matters: the per-class choice is an integer partition of `n_c` with a
device size per group. The space shrinks from Bell(n) to Π_c p(n_c) (p = the
integer partition function; p(3) = 3, p(6) = 11) — already tractable at
network scale before the next reduction.

**The min–max objective admits a parametric solve.** A group of `k` identical
ops at device size `D` contributes load `k·L_c(D)`, so candidate bottleneck
values are finitely many: the products `k·L_c(D)` over `k ≤ n_c`, `D` in the
menu. For a fixed bottleneck target `T` the classes decouple completely:
each class independently computes the minimum DPU budget for which every
group's load stays ≤ T — a small dynamic program over "cheapest way to cover
j members", where covering a group of size `k` at size `D` is admissible iff
`k·L_c(D) ≤ T` and C9 holds in every capacity-bounded memory level
(`k·static_c(D) + dyn_c(D) ≤ capacity`, per level), at budget cost `D`. The target is feasible iff the class minima sum to ≤ D_max (C7);
feasibility is monotone in `T`, so binary search over the sorted candidates
finds the optimum exactly. Cost: O(#classes · n_c² · |menu|) per probe,
logarithmically many probes — exact at hundreds of ops, with **no CSP in the
outer loop at all**. (Under the shape-dynamic C8 extension — same kind, same
`T` — classes sharing an operator kind couple inside the per-target
subproblem; the parametric skeleton is unchanged.)

**Timesharing is priced inside the same solve.** Besides a device size, a
group may take the "unpinned" pseudo-allocation: zero grid budget, and its
load gains the program-reload and weight-rescatter cost per inference. The
partition-vs-timeshare comparison of §3 is then not a special case bolted on
but one more row of the DP's menu; the solver chooses eviction exactly where
residency does not pay.

Brute-force partition enumeration survives only as a validation oracle for
the solver at small n.

Output: per scope, a **budget**: `D_i := D_{s_i}` (P6 becomes an equality or
tight bound) and an MRAM reserve (P5's capacity reduced by co-resident static
footprints).

### Stage C — finalization under budgets

For the chosen points no budgeted re-check is needed at all: the allocator
admits a group only if `k` co-resident copies of the argmin's static
footprint fit beside one working region, so the Stage-A argmin at the
allotted size is feasible under the packing *by construction*. Every member
of a group is then finalized by evaluating exactly the group's argmin
configuration and lowering with it — one evaluation per member, no search —
which is also what implements C8's deterministic stamping: all members of a
group commit the same point because it is handed to them, not re-found. A
short conditioned re-search remains the fallback for extensions whose
budgets can tighten a space after profiling (the shape-dynamic and
layout-coupling variants).

### Why the interface is exact, not a heuristic cut

Check against P1–P6: P1–P5 are intra-op; the only variable of scope `i` that
any cross-op constraint mentions is `D_i` (C7) and the P5 capacity terms
(C9). Conditioning on `(D_i, capacity budgets)` therefore **separates the
joint CSP exactly** — `L_c(D)` together with the argmin's capacity
footprints is a complete summary of the class for the outer problem
(an optimal-value-function projection, Benders-style / Alpa-style). What the
decomposition loses is only:

1. C10's cost term (layout-matched forwarding) — handled, when enabled, by
   joint search of the *pair* with aliased h-factors (a reduced space, not a
   product), triggered only for partitions where the guard is on;
2. contention between sets on the host memory bus during concurrent transfers
   — second-order; state as a limitation, don't model.

### Why this beats joint BO

- The surrogate only ever sees single-op spaces — the exact shape already
  validated by the single-operator experiments. The worry that the surrogate
  gets confused by a tenuously-related product space never arises because
  that space is never constructed.
- The coupling variables live in a small discrete outer problem where an exact
  solver is the right tool; BO is at its worst on constraint-coupled
  near-decomposable spaces.
- Profiles are computed once and reused across all outer candidates; a joint
  BO would re-learn the inner response surface for every region of the outer
  space it wanders into.
- Precedent: Alpa's inter-op/intra-op split (outer DP queries inner ILP's
  optimal values), PipeDream's planner, SET's RA-tree exploration over an
  intra-layer engine, PIMCOMP's staged layer-partition → core-mapping →
  dataflow flow. Hierarchical decomposition with inner value functions is the
  *standard* shape for this problem class; joint flat search is not.

## 3. The examples, worked

**2MM sequential** (`r = A·B; r2 = r·C`, shapes differ): C8 forbids sharing
under the prototype merge. Outer choice is partition-vs-timeshare:

    partition:  max/Σ of L_1(D/2), L_2(D/2), both pinned, pipelined
    timeshare:  L_1(D) + L_2(D) + 2·program_reload + 2·weight_scatter  per inference

With reload at 40–80 ms against sub-ms kernels, partitioning wins by orders of
magnitude — this is the headline whole-program effect, and the profile-based
objective prices it explicitly rather than assuming it: both rows above are
just menu options of §2's solver (the timeshare row is the unpinned
pseudo-allocation). The prototype takes reload as a 40 ms constant (per
switch, not per byte — to be confirmed on hardware) and prices weight
rescatter with the same measured scatter cost model the per-op simulator
uses, a function of the DPU count and the byte volume.

**2MM parallel** (`r = A·B; r2 = A·C`, shared input A, independent gemms):
same allocation math; the objective's aggregation over the two sets is `max`
instead of `Σ`. The shared input A is dynamic (scattered per inference to both
sets); no residency interaction.

**Transformer block**: Q/K/V projections — same shape, same input, different
weights → C8 allows one set, one program, three pinned weight sets under C9's
sum. (Alternative that subsumes merging here: horizontal fusion / QKV weight
concat at linalg level, one bigger gemv. The merge mechanism still carries the
general case.) FFN matmuls (different shapes) → separate sets via the
2MM-sequential logic.

## 4. The independent techniques

- **T1 Offload selection**: IAI ranking; prunes candidates.
- **T2 Grid partitioning & DPU budgeting**: outer variables `{s_i, D_s}`, C7.
- **T3 Program merging by parameter aliasing**: C8; merged scopes share one
  search space, deduplicated by signature. Extension: shape-dynamic kernels.
- **T4 Weight co-residency packing**: C9, sum-of-static + max-of-dynamic,
  over the derived staticness classification (§1).
- **T5 Two-level solving**: profiles (Stage A) → exact parametric allocation
  (Stage B) → budgeted finalization (Stage C).
- **T6 (extension) On-device intermediate forwarding**: C10,
  implication-guarded h-factor aliasing.

T1–T5 are the prototype. T6 and dynamic kernels are extensions with clean
hooks; each is one guarded constraint away, which itself demonstrates the
constraint system's compositionality.

## 5. Fit with the existing machinery

- **Constraint DSL**: needs nothing new. Under the throughput objective the
  outer problem is solved parametrically outside any CSP (§2); the constraint
  system's graph-level job reduces to *instantiating* per-op systems under
  budgets (Stage C), with C7/C9 as extra linear constraints and aliasing as
  variable identification. The solver returns at the graph level only for the
  latency variant's budgeted-crashing subproblem; guarded constraints cover
  T6 later.
- **P6** per op becomes `D_i = D_{s_i}` under a fixed outer candidate; P5 gets
  the reserve term. Both are edits to the *instantiation* of the per-op
  system, not to backend-declared templates — i.e. the graph level tightens
  budgets, the operator level searches under them. The cross-op dependency
  cycle is resolved by value-function profiles where the intra-op cycle was
  resolved by joint constraint solving.
- An equivalent formulation writes C8/C9 with a boolean fuse variable and
  guarded constraints inside one big CSP; the choice here (solve the outer
  problem
  parametrically outside the CSP) is not merely simpler — it is what turns an
  NP-hard-looking joint problem into a polynomial one for the throughput
  objective, and it keeps the surrogate and the solver each on the problem
  shape they are good at.

## 6. Related work anchors

- Hierarchical inter-op/intra-op decomposition with inner value functions:
  Alpa (OSDI'22), PipeDream planner.
- Resource allocation to layers on partitionable/tiled accelerators:
  SET (ISCA'23, RA trees), TANGRAM, Stream (layer-fused DSE, multi-core),
  PIMCOMP (PIM compiler with layer-partition/replication/core-mapping stages —
  closest PIM analog, but ReRAM crossbars, no constraint derivation),
  Planaria (spatial multi-tenancy by architecture fission).
- Outer problem = moldable-task scheduling with compatibility + capacity side
  constraints (classic theory).
- Virtual PIM: runtime multi-tenant DPU allocation on UPMEM — runtime
  scheduling, not compile-time co-optimization.
- Delta kept by this design: cross-operator constraints *derived and solved
  inside the compiler's constraint system*, on a physically-allocated
  workgroup; none of the above derive the space, and IREE's constraints are
  per-dispatch.

## 7. Implementation status

Implemented (the T1–T5 prototype, end to end):

1. Staticness classification: the annotation on program inputs plus the
   derivation through constants, constant-indexed views, and casts. Consumed
   by C9 and by the signature.
2. Graph collection and canonicalization: connected components of the
   dataflow between offloadable regions (§1's over-approximation), each
   partitioned into program-identity classes by a structural signature that
   ignores constant payloads and includes the per-operand staticness
   pattern. The class is the unit everything downstream operates on.
3. Stage A: one profile per class, one pinned search per menu value, run
   concurrently; the menu derived from the iteration-space divisors as
   described there; each point recording cost, argmin, per-level
   static/dynamic footprints, and the static operands' re-scatter cost.
4. Stage B: the parametric outer solve, validated against brute-force
   partition enumeration on small instances (the worked examples of §3
   behave as predicted: different-shape ops partition the grid unevenly —
   each gets the cheapest size under the bottleneck — same-shape ops share a
   set or split it as budget and capacity allow, and timesharing wins
   exactly when the reload constant is made small).
5. Stage C: per-member finalization by direct evaluation of the group's
   argmin (no re-search needed for chosen points, as argued there).
6. Both objectives: the exact parametric solve for throughput, and the
   critical-path greedy for latency, over the same profiles. Graph
   collection records the dependency edges between offloadable regions
   (dataflow through the ops the graph does not own), which is what the
   latency evaluator walks.

The whole flow is opt-in behind an option, and the objective is a second
option; the per-op whole-device search remains the default and the fallback
for targets that declare no shared resource.

Not yet implemented: T6 layout coupling, shape-dynamic kernels, and the
offload-selection ranking (T1 currently trusts the platform-assignment
step). Risk still open: measure program-reload and weight-scatter cost on
the actual system to confirm the 40–80 ms figure (taken as a 40 ms constant
until then) and the per-op-switch (not per-byte) cost model for reload.

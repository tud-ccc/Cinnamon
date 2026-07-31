# Making CNM aware of memory levels (MRAM/WRAM)

Context: today the "optimized" UPMEM programs used for cost evaluation
(`handleGemv`/`handleReduce` in
[UpmemInferAccelerator.cpp](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L391-L622))
are not produced by the real lowering pipeline
(`cinm` → `--convert-cinm-to-cnm` → `--convert-cnm-to-upmem`). They are
produced by bespoke template generators
(`upmem::generateGemv`, `SimulationTemplates.cpp:1122`) that hand-emit
loop nests + `upmem.local_transfer` ops directly, with all tiling
decisions (rank/DPU/tasklet/MRAM/WRAM/register) baked into the
constraint-solving code that builds each template's `SpaceBuilder`.

Goal of this effort: replace the templates with the real pipeline, by
pulling the decisions the templates currently make into passes that
(a) are backend-agnostic in their *mechanism* and (b) expose the
backend-specific *decisions* as parameters, filled in by
`UpmemInferencePlugin` the same way tile sizes are today (see
`kTileParamNamesAttr` / `applyTileSizes`). This mirrors §3.2 ("CNM
changes") of the paper draft
(`cinm2-paper-overleaf/contents/03_cinm_platform.tex:52-90`), which
sketches a `cnm.transfer` op, scatter specializations, constant-scatter
optimization, and transfer coalescing as "optimizations that are
beneficial for UPMEM but implemented generically at the CNM level."

This note focuses on the piece the user flagged as the biggest
obstacle: **making CNM aware of MRAM vs. WRAM at all** (today it only
ever implicitly represents WRAM-resident computation). The other
optimizations (§B/§C below) are more local and are sketched only
briefly, to be designed once §A is settled. A related but distinct
effort, [UpmemGenericLoweringNotes.md](UpmemGenericLoweringNotes.md),
generalizes the *same* templates by staying at the `linalg.generic` →
`upmem` level, without CNM; see §E for how the two connect.

## Inventory: the pieces already exist, but are disconnected

There is more scaffolding for this than it looks like at first glance
— it's just entirely inert:

- `cnm.buffer`'s type already carries a `level: Attribute` parameter
  ([CnmTypes.td:36-41](../include/cinm-mlir/Dialect/Cnm/IR/CnmTypes.td#L36-L41)).
  Nothing ever sets it to a non-null value — `--convert-cinm-to-cnm`
  always builds buffers with `level = {}`
  ([CinmToCnm.cpp:309](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L309),
  `:886-893`).
- `cnm.launch`'s verifier *already* interprets a set `level` as a
  memref memory space: it requires each region argument to have type
  `memref<shape x elt, level>`
  ([CnmOps.cpp:229-249](../lib/Dialect/Cnm/IR/CnmOps.cpp#L229-L249)).
  So structurally, a `cnm.launch` body could already receive an
  MRAM-tagged memref as an argument today — the type-level plumbing
  §A3 below ends up relying on is already sitting there unused.
- `CnmAcceleratorAttrInterface::getWorkgroupMemoryLevels()` is
  implemented for UPMEM
  ([UPMEMAttributes.cpp:246-253](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L246-L253)):
  it returns, per tier of the workgroup shape (rank / dpu / tasklet),
  the list of memory levels available at that tier — `[]`, `[mram,
  wram]`, `[]`. Nothing calls this method outside its own definition.
- The platform/level infrastructure it's built on
  (`CinmLevelDefAttr`: name + size + alignment + arity,
  `CinmPlatformAttrInterface::getLevel(name)`,
  `UpmemPlatformAttr::getMramLevel()/getWramLevel()`) is real and
  already used — but only by the hand-written `SpaceBuilder` DSE code
  in `handleGemv`/`handleReduce`, not by any pass.
- `--convert-cnm-to-upmem` already does an MRAM↔WRAM split, but as one
  fixed, hardcoded strategy, not driven by `level` at all: every
  `cnm.alloc` unconditionally gets an MRAM buffer
  (`upmem::StaticAllocOp`) plus a WRAM buffer (private
  `upmem.pwram_alloc`, or a single shared static alloc if
  `isMramBroadcastOverThreads` holds), wired together with explicit
  `upmem.local_transfer` ops inserted right before/after `cnm.launch`
  ([CnmToUPMEM.cpp:424-602](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L424-L602)).
  There's no type converter for `cnm.buffer` in this pass at all — it
  pattern-matches `cnm.alloc`/`scatter`/`gather`/`launch` directly and
  builds the memrefs itself via an `IRMapping`, so `getLevel()` is
  never even read here. Crucially, **the `cnm.launch` body always
  computes over WRAM**; there is no way today to have a launch body
  operate on an MRAM-resident buffer with its own internal tiling into
  WRAM — that's exactly the `mramRow/mramCol` vs. `wramRow/wramCol`
  split `generateGemv` does by hand. (§A3 below settles how this
  changes: the launch body gets the MRAM buffer directly, and the
  internal WRAM tiling is done via a second application of the
  existing `CinmTilingInterface` tiling mechanism rather than being
  hardcoded per op.)

So the work isn't "invent MRAM-awareness from scratch," it's "decide
the semantics of the `level` field and the launch-body contract, then
wire the existing interface methods into passes that currently ignore
them."

## A. CNM-level design

### A1. What does `level` hold?

**Decided:** introduce a `CinmLevelAttrInterface` (name TBD — lives
next to `CinmPlatformAttrInterface`/`CinmAcceleratorAttrInterface`,
same pattern) and make all *generic* code (the interfaces in §A5, the
new MRAM-tiling pass in §A4, `getMemrefMemspace`, capacity constraints)
speak in terms of the interface rather than the concrete
`CinmLevelDefAttr`. Backend dialects are then free to define their own
attributes implementing it — `#upmem.mram`/`#upmem.wram` — and attach
backend-specific behavior directly to them (the motivating case: SDK
transfer-call selection per level, see §B).

**Correction (implemented).** This section originally had
`CinmLevelDefAttr` become one implementation of the interface. It
doesn't, and can't: the level attribute doubles as a *memref memory
space*, so parameterizing it with a capacity would make
`memref<64xi32, wram>` two distinct types on UPMEM v1A and v1B, whose
WRAM sizes differ. The two are therefore different concepts —
`CinmLevelAttrInterface` carries level *identity* only (a single
`getLevelName()`), and `CinmLevelDefAttr` stays a platform-side
capacity *descriptor* that does not implement it. `getLevelOfMemspace`
on the platform interface maps one to the other, so generic code can
still reach a level's size from a memref's memory space.

Concretely this made the change much smaller than anticipated:
`upmem::DpuMemSpaceAttr` (`#upmem.mram`/`#upmem.wram`) already existed
and was already the memref memory space throughout the UPMEM backend,
so it just implements the interface. `upmemLevels()`
([UPMEMAttributes.cpp:145-163](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L145-L163))
keeps building `CinmLevelDefAttr`s unchanged, and
`getWorkgroupMemoryLevels()`/`CinmLevelArrayAttr` need no change at
all.

Two reasons for this over reusing `CinmLevelDefAttr` directly: (1) it
lets a level attribute drive backend-specific lowering decisions
(§B/§A4) instead of just describing capacity; (2) `CinmLevelDefAttr`'s
struct-typed assembly format is verbose to spell out at every call
site compared to a lightweight custom attribute.

Per your note, **arity drops out of the interface** — it was only
used for the commented-out parametric-indexing scheme
(`toMemSpace(indices)` in
[CinmAttributesBase.td:105-109](../include/cinm-mlir/Dialect/Cinm/IR/CinmAttributesBase.td#L105-L109),
already dead code) and isn't needed by any live consumer. A concrete
attribute like `#upmem.mram` can still privately track how it's
indexed if it needs to; that's no longer the generic interface's
concern.

Follow-on implications to work out (not blocking the decision, just
consequences of it):

- **Interface method surface**: at minimum `getName()` +
  `getSizeInBytes()` (used today for capacity constraints via
  `getSizeInElements(t)`) and `getMemrefMemspace()`-equivalent
  (currently `UpmemPlatformAttr::getMemrefMemspace(level)` takes a
  `CinmLevelDefAttr` and returns `level.getName()` — does this move
  onto the level interface itself, e.g. `level.getMemrefMemspace()`,
  now that behavior can live on the level attribute rather than the
  platform?). Alignment: keep on the interface (used for allocation)
  or push to the concrete attribute alongside arity? Leaning toward
  keeping it, since `EnsureScatterGatherContiguous.cpp`-style
  reasoning needs it generically, but flagging since it wasn't called
  out explicitly.
- **Ripple into existing signatures**: `CinmPlatformAttrInterface::
  getLevels()`/`getLevel(name)` currently return
  `ArrayRef<CinmLevelDefAttr>`/`CinmLevelDefAttr`
  ([CinmPlatformAttrInterface.td:29-44](../include/cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td#L29-L44)),
  and `getWorkgroupMemoryLevels()` returns
  `SmallVector<CinmLevelArrayAttr>`, where `CinmLevelArrayAttr` is a
  tablegen `ArrayOfAttr` generated over the *concrete*
  `CinmLevelDefAttr` class
  ([CinmAttributesBase.td:123-124](../include/cinm-mlir/Dialect/Cinm/IR/CinmAttributesBase.td#L123-L124)).
  `ArrayOfAttr` doesn't support being parameterized over an interface
  the same way, so this becomes either `ArrayRef<Attribute>` (with
  `cast<CinmLevelAttrInterface>` at use sites) or a hand-written array
  attribute wrapping the interface. Small mechanical change, but every
  current call site that assumes `CinmLevelDefAttr`-specific accessors
  (e.g. `UpmemAcceleratorAttr::getMramLevel()` returning it by value)
  needs to switch to the interface type.
- `UpmemPlatformAttr`'s `levels` parameter
  ([UPMEMTileFirstAttributes.td:23-29](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMTileFirstAttributes.td#L23-L29))
  is currently typed as `CinmLevelArrayAttr` specifically (not
  generic) — needs the same widening.

### A2. Where does the level come from as a pass parameter?

**Decided:** no per-operand granularity for now. `--convert-cinm-to-cnm`
gets a single pass-wide option (`buffer-level=mram|wram|...`), matching
today's "one strategy for the whole trial" style of the inference
plugin — for UPMEM that's always MRAM. Per-operand residency
(`UpmemGenericLoweringNotes.md` §4 — `x` broadcast/shared vs. `A`/`y`
private) is deferred; §A3 below actually ends up giving each operand
its own answer anyway, just decided later in the pipeline (during the
launch-body tiling step) rather than by this pass parameter.

### A3. What can a `cnm.launch` body do with a non-leaf-level buffer?

This was the crux question, and staging transfers *outside*
`cnm.launch` (my original option (a)) turns out not to be viable:
`cnm.launch`'s contract is the host/device boundary — everything
around it is host code, everything inside is on-device (possibly
threaded) code. On-device transfers are *requested by device code*;
there's no way to represent "transfer a tile into WRAM" as host-side
IR sitting outside the launch, because the DPU/tasklet is the one
issuing it.

**Decided design:**

- At the `cnm.launch` boundary, every outer `cnm.buffer`'s `level` is
  reflected onto the corresponding region argument's memref, exactly
  as the verifier already requires today (§ inventory above). So a
  launch body can legitimately see a `memref<..., #upmem.mram>`
  argument now — this was previously true only on paper (nothing ever
  set `level`); now it's the normal case for the MRAM path.
- That block argument is a *logical* per-tasklet view, same as for
  WRAM today — CNM's existing semantics are that each SPMD "thread" in
  the launch body sees its own private slice of the workgroup buffer.
  For MRAM specifically this is only a logical view: UPMEM's `Cnm →
  upmem` lowering requires MRAM buffers to be **statically allocated**
  (one allocation, DPU-wide) for now, matching what
  `upmem::StaticAllocOp` already does for the buffers `CnmToUPMEM.cpp`
  materializes today. So for an MRAM-level block argument, the
  lowering doesn't allocate a separate memref per tasklet — it
  replaces uses of the block argument with a `memref.subview` of the
  single static MRAM allocation, indexed by the tasklet's position.
  This is exactly the `mramHasTaskletDim` subview logic
  `createTransfer` already has
  ([CnmToUPMEM.cpp:335-367](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L335-L367)),
  generalized to apply to every use of the argument, not just the
  transfers this doc adds around it. WRAM-level block arguments (the
  tile buffers allocated by §A3's cleanup pass) don't need this: they
  really are private per tasklet, one small allocation each, as today.
- If the buffer is broadcast across tasklets (the scatter map doesn't
  depend on the tasklet dimension — same tile for every tasklet on a
  DPU), there's no subview at all: it's one shared MRAM buffer, used
  as-is by every tasklet. This is already detected today by
  `isMramBroadcastOverThreads`
  ([CnmToUPMEM.cpp:424](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L424))
  purely from the scatter map, at the `Cnm → upmem` lowering stage —
  the same place `isWramShared` decides WRAM sharing. Both carry over
  unchanged: this is backend-lowering detection, not something the
  CNM-generic side needs to model or decide.
- Inside the launch body, `--convert-cinm-to-cnm` doesn't emit lowered
  loops directly — it emits a structured op over those memrefs, which a
  later pass tiles down to the leaf level and wraps in transfers.

  **Revised (implemented).** This section originally required those to be
  **cinm ops**, so that the second staging round could reuse
  `CinmTilingInterface` and `--cinm-tiling`. That does not work, and the
  revision generalizes rather than narrows:

  - `--convert-cinm-to-cnm` maps each *leaf* of the workgroup to a slice
    of the computation. For `cinm.op.gemm` that slice is a single output
    element, so the per-leaf body is a **dot product** — there is no
    shape in which `cinm.op.gemm` could stand inside its own launch. The
    same happens to gemv whenever a leaf gets one row. So the cinm-op
    requirement would have excluded gemm entirely.
  - The launch body is already linalg today, and staging it down is
    exactly what `linalg::promoteSubViews` does: allocate in a faster
    memory space, copy in, compute, copy out — with hooks for the memory
    space, the allocation, and the copy op, the last of which is where
    `cnm.local_transfer` goes.

  So the launch body **stays linalg**, and the staging pass keys off the
  memref's memory space alone. It then applies to *any* `LinalgOp` with
  buffer semantics rather than to the two or three cinm ops that happen
  to have a usable per-leaf shape, and `--convert-cinm-to-cnm` goes back
  to making no decision at all about the body.

  `CinmTilingInterface` keeps its original role: the *host-side*
  `--cinm-tiling` round that produces the workgroup tile. Only the
  on-device staging round is linalg's.

- The staging pass (`--upmem-tile-mram-buffers`) tiles the body with
  `scf::tileUsingSCF` at the leaf tile size, then promotes the resulting
  subviews into the leaf level. This is the piece that replaces the
  hand-coded `mr`/`mc` loop body of `generateGemv`.
- **Deferred optimization:** when the scatter map shows a buffer is
  broadcast across tasklets of a DPU (same tile for every tasklet on a
  DPU, DPUs may still differ), the private-WRAM buffer above can
  instead become a single shared static allocation guarded by a
  barrier (mirrors `isWramShared`/`PrivateWRAMAllocOp` vs. static
  alloc in `CnmToUPMEM.cpp` today). Left for later — it may end up
  living at the UPMEM dialect level rather than CNM, partly because
  CNM has no barrier primitive yet (§C already flags this gap).

Both levels of tiling reuse existing, working machinery instead of
inventing a new mechanism — `CinmTilingInterface`/`--cinm-tiling` on the
host side, `scf::tileUsingSCF`/`linalg::promoteSubViews` on the device
side. That is a considerable simplification over my original framing of
this as a from-scratch design.

One known gap to close afterwards: promotion places copy-in and copy-out
immediately around the op, so an output tile inside a reduction loop
round-trips to the outer level on every trip, where the hand-written
templates write it back once. See M5b in the implementation plan.

Both remaining implementation questions from the previous draft are
resolved:

- `--cinm-tiling` already works properly on memref operands (not just
  tensors), so no separate DPS-typed variant of `cinm` ops is needed
  for them to appear inside a `cnm.launch` region.
- The alloc+transfer+op+transfer rewrite isn't a pattern lurking inside
  a generic pass waiting for a trigger condition — it's its own
  explicit step, contributed by the UPMEM dialect (not CNM-generic),
  run at a fixed point in the pipeline:

  ```
  --convert-cinm-to-cnm=cnm-buffer-level=mram
      → [upmem-provided cleanup pass: second --cinm-tiling pass down to
         the WRAM tile size, then rewrite leaf-tiled ops onto new
         private-WRAM buffers with cnm.local_transfer around them]
      → further CNM-level optimizations (§B: scatter specializations,
         constant-scatter simplification, etc.)
      → --convert-cnm-to-upmem
  ```

  This fits the framing from the start of this doc: the pipeline stays
  orchestrated by the UPMEM dialect (via the inference plugin), CNM
  passes stay generic/parameterized, and backend-specific *sequencing*
  of those generic passes (including this cleanup step) is the
  backend's job, not something CNM needs to decide for itself.

### A4. `cnm.local_transfer`

**Decided:** named `cnm.local_transfer` (mirroring
`upmem.local_transfer`, which it plays the same role as one level up
the stack). It operates on **memrefs, not `cnm.buffer`s** — `cnm.buffer`
represents a buffer distributed across a whole workgroup; inside a
launch body we're only ever manipulating one PE's local, ordinary-
shaped view of a tile, which a plain memref already models correctly.
This settles A4's original open question too: since it's memref-to-
memref, ordinary `memref.subview` is used to carve the tile out of the
MRAM-tagged memref — no new CNM-level subview op needed.

`cnm.local_transfer` is a CNM-dialect op (so it can be emitted by the
launch-body tiling step in §A3, which runs before any backend-specific
conversion), but *how* it's actually realized — which SDK call, block
vs. multi-block transfer, etc. — is left to the `Cnm → backend`
conversion pass, same division of labor as the rest of CNM. For UPMEM
this presumably lowers close to 1:1 onto `upmem.local_transfer`,
which absorbs the `useSgXferCodegen`/`useBcXferCodegen`-style choices
that live in `CnmToUPMEM.cpp`'s `Opts` today.

### A5. Does `getWorkgroupMemoryLevels()` carry enough information?

It currently returns, per workgroup tier, a `CinmLevelArrayAttr`
(size/alignment/arity per level) — that answers "how big" and "how
aligned," which is enough for capacity-fitting constraints (as
`handleGemv` already does with `mramLevel.getSizeInElements(eltTy)`).
It does **not** currently answer "how do you move data between level
N and level N+1" — e.g. whether a transfer must be contiguous, what
its DMA granularity/alignment is, or its cost per byte/per
transaction. That information currently lives implicitly in
`CnmToUPMEM.cpp` (the "must be contiguous" logic in `ScatterOp`,
`EnsureScatterGatherContiguous.cpp`) and in the SDK-selection knobs in
`Opts` (`useSgXferCodegen`, `useBcXferCodegen`).

**Resolved:** not needed at the CNM level, at least for now. The "how
do you move data" question is answered structurally by
`cnm.local_transfer` (§A4); its actual realization (contiguity
handling, SDK call selection, cost) is entirely the `Cnm → backend`
pass's problem, same as today. The cost model itself is not a CNM-level
concern at all: it operates exclusively at the UPMEM-dialect level and
assumes CNM ops have already been converted — which matches how DSE
already works (`UpmemInferAccelerator.cpp` runs a trial all the way
down to `upmem` dialect IR and simulates that, per the code excerpt
that kicked off this doc). So no cost/shape metadata needs to be added
to the level attribute for this effort; `getWorkgroupMemoryLevels()`'s
existing size/alignment info is sufficient for the CNM-level capacity
constraints it's used for today.

## B. Scatter/gather specializations (later)

From the paper draft, sketched only for completeness — design to
happen after §A:

- **Broadcast detection**: when a `scatterMap` doesn't depend on the
  processing-element dimension, specialize `cnm.scatter` into a
  broadcast. This is a canonicalization purely on today's CNM IR
  (scatter map affine structure), independent of MRAM/WRAM — although
  which buffer `level` is being broadcast into changes which UPMEM
  API/codegen path applies downstream.
- **Constant-scatter simplification**: detect a constant (e.g.
  zero-init) being scattered; if homogeneous, rewrite to a broadcast of
  a smaller tile; if small enough, rewrite further into an on-device
  init loop or a static array in the device binary. Same story —
  orthogonal analysis, but the "small enough" threshold and "on-device
  init loop" choice presumably differ by target level (WRAM zero-init
  vs. MRAM).
- **Blocked/multi-block transfer API selection**: UPMEM offers both a
  uniform per-DPU block transfer and a per-DPU arbitrary-block-count
  transfer; choosing between them is a codegen decision that (per the
  draft) should be "implemented generically at the CNM level" using
  the scatter map as infrastructure. Needs its own design pass once
  §A's `cnm.local_transfer`/buffer-level story is settled, since
  MRAM-level scatters and WRAM-level scatters may want different
  answers.

## C. On-device transfer coalescing (later)

"Transfers executed concurrently by different tasklets can be coalesced
into blocked transfers executed by a leader tasklet" (paper draft). Once
§A4's `cnm.local_transfer` exists inside/around launch bodies, this
becomes a rewrite that merges sibling per-tasklet transfers into one leader
transfer plus a barrier. Note: **CNM has no barrier/synchronization
primitive today** — `cnm.launch` bodies are implicitly SPMD with no
cross-tasklet coordination modeled in the dialect. This coalescing
(and the row-leader merge pattern already hand-written in the
reduce template) can't be expressed generically without one. Worth
flagging now even though the design is deferred: **a barrier op may be
a second CNM primitive this effort needs**, not just `cnm.transfer`.

## D. Parameterization / DSE hookup

**Confirmed**: this is exactly the existing mechanism, applied twice.
`SpaceBuilder` solves constraints, then stamps a `cinm.tile_sizes`
attribute (`TILING_FACTORS_NAME`) that `--cinm-tiling` reads via
`CinmTilingInterface::convertToTiledOps`
([UpmemInferAccelerator.cpp:361-388](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L361-L388),
[TilingPass.cpp](../lib/Dialect/Cinm/Transforms/TilingPass.cpp)). §A3's
design reuses this unchanged: the UPMEM inference plugin computes
*both* the MRAM tile sizes (today's role) *and* the WRAM tile sizes
(new), and both get consumed by the same `--cinm-tiling` pass — once
before bufferization/`--convert-cinm-to-cnm` (host → MRAM tile), once
again inside the launch body post-materialization (MRAM tile → WRAM
tile). All tiling decisions are resolved before `--convert-cnm-to-upmem`
runs; that pass stays a comparatively mechanical lowering. No new
parameterization mechanism needed for this effort.

`CinmAcceleratorAttrInterface::getDesignParams()` /
`instantiateDesignParams()` are already stubbed for exactly this kind
of thing but currently return `{}` / identity for UPMEM
([UPMEMAttributes.cpp:230-257](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L230-L257)) —
worth checking whether these are meant to *replace* the
`SpaceBuilder`/attribute-stamping mechanism eventually, or whether
they're for a different (declarative, not search-based) class of
parameters. Not blocking for §A, but the two mechanisms coexisting
long-term without a stated boundary between them seems likely to
cause confusion.

## E. Relationship to `UpmemGenericLoweringNotes.md`

That note's ten problems (reduction-split taxonomy, loop→hardware-role
assignment, competing reductions, per-operand residency, joint
capacity fitting, divisibility/remainder, carry/neutral-element
bookkeeping across nested sequential accumulation, associativity
assumptions, combiner-region reuse, cost-model coverage) were derived
by generalizing `generateGemv`/`generateTailReduction` directly at the
`linalg.generic` → `upmem` level, without CNM in the picture at all.
Every one of them resurfaces verbatim in the CNM-mediated flow — CNM
doesn't dissolve them, it just changes where the answer gets recorded
(as `cnm` ops/attributes instead of raw memref + `affine.for` loops).
In particular:

- §A3's launch-body-tiling design is exactly where that note's
  points 1-3 (reduction taxonomy, role assignment, resource
  competition) and point 6 (divisibility) live — running
  `--cinm-tiling` a second time inside the launch body *is* "tile
  every loop through the memory hierarchy," just phrased in
  `CinmTilingInterface`/`cnm.local_transfer`/`cnm.launch` terms.
- §A2's per-operand level question is that note's point 4
  (buffer residency/replication is per-operand) restated.
- §D's parameterization question is that note's point 10 (cost-model
  coverage) — a generic cost function needs to exist before there's
  anything for tile-size/role-assignment search to optimize against.

**Open question:** should these become one document once §A3 is
settled, since at that point the "generic tiling" problem is the same
problem regardless of whether it's phrased as CNM ops or raw
memref+loops? Or is keeping them separate useful because one targets
"bypass CNM, go straight to upmem" as a possibly-permanent fallback
path (not just a stepping stone)? Worth clarifying whether the
`linalg.generic → upmem` direct path is meant to keep existing
alongside the CNM-mediated one, or whether it's superseded once this
effort lands.

## F. Migration / validation

Not designed here, just flagged: once §A produces *some* end-to-end
`cinm.gemv → cnm → upmem` path, how do we know it reproduces
`generateGemv`'s output (or at least its cost/perf characteristics)?
**Decided:** the template path stays *indefinitely*, not just as a
development oracle — `SimulationTemplates.cpp` and the
`registerSimulator` bypass are the quality bar the generic pipeline
must reach, and `evaluate()` selects between the two so their costs can
be compared directly on the same configuration. Use gemv specifically
as the first target for the new pipeline (it's the smallest existing worked example with an actual
MRAM/WRAM split, and `UpmemGenericLoweringNotes.md` already flags
matmul/double-reduction as the next-hardest cases — no need to jump
there first).

## G. Parameterizing the distribution: `linalg → cnm`

**Motivating failure.** M7's plugin fed the template search's
`mramRow`/`mramCol` into `cinm.tile_sizes` and produced a tile far too
small, then failed in `--convert-cinm-to-cnm` with `numParallelElts
(64) % numWgItems (16384) != 0`. Two separate causes:

1. `mramRow`/`mramCol` are *per-DPU* block sizes, but `--cinm-tiling`
   must produce the *per-workgroup* tile that `--convert-cinm-to-cnm`
   then distributes over all `dpus * tasklets` leaves.
2. More fundamentally, `computeShapeOfTensors`
   ([CinmToCnm.cpp:143-340](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L143-L340))
   maps *parallel* dims onto workgroup elements and places the *whole*
   reduction extent in each leaf's buffer. It never splits a reduction
   across the workgroup. So `dpuCols`/`taskletCols` — K-splitting with
   partial-sum merging, which the templates do — have **no
   representation at all** in the CNM path. For the logged
   configuration (M=4096, K=4096, 2048 DPUs × 8 tasklets) the
   constraints force `dpuCols >= 32`; the CNM path would need 4096 rows
   spread over 16384 leaves, a quarter row each. The search space was
   describing a decomposition the generic pipeline cannot express.

**Decided:** give CNM reduction splitting, and take the opportunity to
make the distribution mechanical rather than heuristic. This is the
same theme as the rest of this effort — `--convert-cinm-to-cnm` takes
too many decisions; they should be parameters the DSE supplies.

### G1. The model

Treat the op as a loop nest over its iteration space. Tiling each
dimension yields outer loops over tiles and inner loops within a tile.
The inner loops become the `cnm.launch` body; the outer loops are never
emitted — they *are* the workgroup, and they determine the
scatter/gather maps. Everything the conversion needs follows from one
block-size vector.

### G2. The parameter

A per-iteration-dimension **block size** `b_i`, from which the tile
count `f_i = E_i / b_i` is derived.

Block sizes, not factors, deliberately: `--cinm-tiling`'s
`cinm.tile_sizes` are block sizes (loop steps — verified: `array<i64:
16, 32>` on `memref<64x128xi32>` gives `step 16`/`step 32`), and the
search's natural variables (`mramRow`, `mramCol`) are block sizes too.
One unit throughout the stack; the M7 failure above was precisely a
units confusion.

Constraint: `∏(E_i / b_i) == |WG|` **exactly**. Sequential outer trips
stay upstream in `--cinm-tiling`, so this conversion distributes one
workgroup-sized tile and never emits a host loop of its own.

This lines up term-for-term with the template constraints: M-tiles ↔
`dpuRows`, K-tiles ↔ `dpuCols`. The `numParallelElts % numWgItems`
failure class disappears — the constraint is stated in the parameters
and checked in `SpaceBuilder` rather than discovered mid-conversion.

### G3. Tile-space → workgroup order: fixed by rule, not searched

Mapping tile coordinates to workgroup coordinates needs an *order*, not
just sizes, and the order is not neutral: it decides which dim varies
fastest across tasklets vs DPUs, hence scatter contiguity.

**Decided:** linearize both sides (the workgroup already is —
`mlir::linearizeIndices(ctx, wgShape)`) and fix the tile-side order by
rule: **parallel dims outer, reduction dims inner, op dim order within
each group.** No search variable.

Under this formulation the template's `dpuCols` and `taskletCols`
collapse into a single `k_tiles = K / b_k`: the DSE picks how many
reduction tiles there are, not where the DPU/tasklet boundary falls
inside them.

Note that linearization is not *bad* at locality. With reduction dims
innermost, the partials of one output region occupy consecutive linear
leaf indices, and leaves within a DPU are consecutive blocks of `T`, so
all `k_tiles` partials share a DPU exactly when `k_tiles | T`. When
`k_tiles > T` the split degrades gracefully into a two-level merge:
`T` partials merged locally, then `k_tiles / T` groups merged across
DPUs.

**What is actually given up is the ratio.** Linearization is greedy —
the innermost tile dim fills the fastest hardware axis completely
before spilling to the next — so it forces `taskletCols = min(k_tiles,
T)`. The template space can choose `taskletCols = 2` with `T = 8`
(leaving `taskletRows = 4`); linearization with the same `k_tiles = 8`
must give `taskletCols = 8, taskletRows = 1`.

**Why the ratio matters, and it is not merge cost.** MRAM is per-DPU
and shared across tasklets, and whether an operand is *replicated per
tasklet* or *shared* is decided by whether its indexing map varies
along the tasklet axis — i.e. by the assignment. The lowering shows
this directly: a scatter map depending on the tasklet dim produces a
leading tasklet dimension plus per-tasklet subviews, one that does not
produces a single shared buffer
([cnm-to-upmem-mram-level.mlir:73-84](../test/Conversion/CnmToUpmem/cnm-to-upmem-mram-level.mlir#L73-L84)).
For gemv (`A[m,k]`, `x[k]`, `y[m]`) the two ends are:

- *Reduction innermost* (what linearization gives): tasklets share
  `m_tile`, differ in `k_tile`. `y` becomes `T` locally-mergeable
  partials, but `x` is replicated `T` times in MRAM.
- *Parallel innermost*: tasklets share `k_tile`, differ in `m_tile`.
  `x` is one shared slice — a `T`× MRAM saving — but every partial goes
  to the host.

Neither dominates; which wins depends on the broadcast operand's size
relative to the merge traffic, i.e. on the problem shape. That is
search-variable material.

**So this is not purely a performance knob**: per-tasklet replication
changes per-DPU MRAM footprint, so a fixed order can change
*feasibility* under capacity constraints. Every factorization
`(m_tiles, k_tiles)` stays reachable, but not every factorization stays
*fittable*.

**Why deferring is still right, and when to revisit.** For gemv the
effect is second-order: `A` dominates the footprint and is indexed by
both dims, so it is replicated under every assignment and the `x`
saving is small. Good enough for M8/M9. The revisit trigger is **a
broadcast operand large relative to the per-leaf tile** — gemm with a
shared operand will hit this well before device-side merging (G10)
does.

Note the scale when revisiting: iteration rank is 1 (elementwise), 2
(gemv), 3 (gemm), so there are at most `3! = 6` orders. Enumerate them
(a small categorical variable, or one space instance per order); a
first-class permutation representation with a permutation-aware
distance function, BACO-style, is not warranted at this size and
probably never will be.

### G4. Reduction splitting: the one structural addition

If reduction dim `d` has `f_d > 1`, leaves differing only in `d`'s tile
coordinate compute partials of the *same* output region, so the gather
map is no longer injective unless the host destination gains a
dimension of size `f_d`. The conversion emits `cnm.gather` into
`tensor<... × f_d × outTile>` followed by a host-side `linalg.reduce`
over that dim using the op's combiner.

This is genuinely new IR, not bookkeeping — everything else in G1
follows mechanically from the tiling.

Reuse candidate: `linalg::splitReduction`
([Transforms.h:1271](../third-party/llvm/mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h#L1271))
performs exactly this partial+merge transformation and creates the
neutral-element fill itself, covering G5 too. To check: whether its
extra-dim placement convention matches what the gather wants, or
whether `PartialReductionOpInterface`
([TilingInterface.td:404](../third-party/llvm/mlir/include/mlir/Interfaces/TilingInterface.td#L404))
is the better route.

### G5. Identity seeding

Every leaf starts from the reduction identity, never from the incoming
`out` value — otherwise the init is applied `f_d` times. The original
`out` is folded in exactly once during the host merge. Commit 71cd9c1
already seeds the scatter init with the method's identity for the
unsplit case; splitting makes it mandatory. Seeding *all* leaves with
identity (rather than one with `out` and the rest with identity) also
composes with the broadcast-of-constant reduction optimization.

### G6. Legality

Splitting requires an associative and commutative combiner.
`add`/`mul`/`min`/`max` qualify; an arbitrary `linalg.reduce` region
does not. The op must answer "may dim `d` be split?", and the
conversion must refuse `f_d > 1` when it cannot. On floats even `add`
is a numerics change, so float reassociation is an **explicit opt-in
flag**, not something the search does silently (expected to be enabled
most of the time in practice).

### G7. What the parameter set does *not* need

- **Broadcast operands** come free: an operand whose indexing map omits
  dim `d` gets a scatter map constant in `d`'s workgroup coordinate —
  the existing `#bcast` shape.
- **Buffer level** stays the separate `cnm-buffer-level` parameter (§A2).
- **Leaf tiling** stays `upmem.leaf_tile_sizes`, consumed by
  `--upmem-tile-mram-buffers` (M5).
- **Operand placement** is derived from indexing maps.

### G8. Prerequisite: distribute `linalg`, not `cinm`

The conversion needs, per op: iteration domain, per-operand indexing
maps, reduction dims, combiner. That is exactly the `LinalgOp`
contract, and M5 already committed to keying off linalg indexing maps
rather than op identity.

**Decided:** convert to linalg first and add a
`--convert-linalg-to-cnm` pass. `--convert-cinm-to-linalg` already
exists and covers gemv/gemm/batch variants/reduce/elementwise/transpose
([CinmToLinalg.cpp](../lib/Conversion/CinmToLinalg/CinmToLinalg.cpp)),
though it may need updating.

This also gives **operator fusion for free** — linalg fusion on tensors
before distribution.

Consequences to plan around:

- **Fusion breaks op identification.** `recordParams`/`walkIndexOf`
  index ops by walk position; fusion changes the op count, the walk
  indices, *and* the iteration space (a fused op's dim set is neither
  producer's). Required ordering: convert to linalg → fuse → build the
  search space by walking the fused linalg ops → stamp. This is also a
  simplification: `handleGemv` and the other per-op handlers become one
  generic handler driven by indexing maps and `getReductionDims`.
- **The two paths fork earlier.** The templates path stays on cinm ops,
  so `--convert-cinm-to-linalg` sits on the generic branch only. §F's
  "same configuration, two lowerings" comparison weakens — after fusion
  the generic path may not have the same op set to configure. **Open:**
  should the comparison baseline be the *unfused* generic path?

### G9. Simplification payoff

Most of
[CinmToCnm.cpp:143-340](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L143-L340)
deletes. The "case 0 / case 1 / case 2, flatten trailing parallel dims
until `trailing % k == 0`, maybe emit a reshape, and the transpose has
to happen in the caller" block exists to *guess* a distribution. Here
the scatter map is `indexingMap ∘ (workgroup → tile)`, computed
directly, and contiguity is already somebody else's job
(`--cnm-ensure-scatter-gather-contiguous`).

### G10. Deferred

Device-side tree reduction of partials — host merge only for now. A
pure optimization over the same parameterization; it can land later
without changing the parameter set. It is, however, the trigger for
revisiting G3.

## Summary of open questions

1. ~~§A1: reuse `CinmLevelDefAttr` for `cnm.buffer`'s `level`, or
   introduce backend-specific marker attributes?~~ **Decided:**
   introduce a `CinmLevelAttrInterface`; `CinmLevelDefAttr` becomes one
   concrete implementation of it, backends may define their own
   (`#upmem.mram`/`#upmem.wram`). Arity dropped from the interface.
   Follow-ups: pin down the method surface (does
   `getMemrefMemspace` move from platform onto the level itself?
   does alignment stay generic?), and work out the mechanical ripple
   into `getLevels()`/`getWorkgroupMemoryLevels()`/`UpmemPlatformAttr`
   now that levels aren't one concrete attribute type.
2. ~~§A2: per-operand granularity for the buffer-level pass parameter?~~
   **Decided:** no, single pass-wide level for now (always MRAM for
   UPMEM); per-operand residency falls out later, during §A3's
   launch-body tiling, rather than from this parameter.
3. ~~§A3: what can a `cnm.launch` body do with a non-leaf-level
   buffer?~~ **Decided:** the level is reflected onto the region
   argument's memref (as the verifier already required); inside, cinm
   ops implementing `CinmTilingInterface` operate directly on it
   (`--cinm-tiling` already works on memref operands, no DPS variant
   needed), and a second, UPMEM-provided cleanup pass runs
   `--cinm-tiling` again (MRAM tile → WRAM tile) and then rewrites the
   resulting leaf-tiled ops onto new private-WRAM buffers with
   `cnm.local_transfer`s around them. This pass is its own explicit
   pipeline step (after `--convert-cinm-to-cnm`, before §B's CNM
   optimizations and `--convert-cnm-to-upmem`), not a trigger-condition
   pattern inside a generic pass.
4. ~~§A4: `cnm.transfer` operand shape?~~ **Decided:** renamed
   `cnm.local_transfer`, operates on memrefs (not `cnm.buffer`), tiling
   done via ordinary `memref.subview` — no new CNM-level subview op
   needed. Its backend realization (SDK call selection etc.) is a
   `Cnm → backend` conversion concern, not decided generically.
5. ~~§A5: transfer cost/shape metadata on level attributes?~~
   **Resolved:** not needed at the CNM level. The cost model runs
   exclusively at the UPMEM-dialect level, after CNM ops have already
   been converted; CNM only needs `getWorkgroupMemoryLevels()`'s
   existing size/alignment info for capacity constraints.
6. §D: are `getDesignParams()`/`instantiateDesignParams()` meant to
   eventually replace the `SpaceBuilder` + attribute-stamping
   mechanism, or serve a different class of parameters? (Still open —
   §D confirms the *existing* mechanism, reused twice, is sufficient
   for this effort, but doesn't say what these stubs are for.)
7. §G8: with fusion on the generic branch only, the two paths no longer
   share an op set, so §F's same-configuration cost comparison weakens.
   Should the comparison baseline be the *unfused* generic path? (New,
   open.)
8. §G4: reuse `linalg::splitReduction` for the partial+merge rewrite, or
   go through `PartialReductionOpInterface`? Depends on whether
   `splitReduction`'s extra-dim placement matches what the gather needs.
   (New, open — resolve empirically during M8.)
9. §E: merge with `UpmemGenericLoweringNotes.md` once §A3 is settled,
   or keep the direct `linalg.generic → upmem` path as a permanent
   parallel option? (§A3 being settled now makes this more concrete:
   the note's points 1-3 map onto the two `--cinm-tiling` passes'
   role-assignment/reduction-split policy — worth a pass to check how
   much of the note is now answered.)

---

<!-- Add your comments below or inline as blockquotes -->

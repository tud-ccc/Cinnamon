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
  the scatter map as infrastructure. **Answered by §J**: the number of
  device dimensions the map retains *is* the block count, so the
  selection is a reading of the map rather than a separate decision.
  §K takes the remaining step and moves the *op* selection out of the
  conversion entirely — the conversion emits the general form and the
  UPMEM dialect specializes it, so the first two bullets above no longer
  have to predict which backend op they will produce.

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
rule: **reduction-derived (split) dimensions outermost, then the
original parallel dimensions, then the unsplit reduction remainder**,
op dim order within each group. No search variable.

Under this formulation the template's `dpuCols` and `taskletCols`
collapse into a single `k_tiles = K / b_k`: the DSE picks how many
reduction tiles there are, not where the DPU/tasklet boundary falls
inside them.

**Why the split dimension goes outermost.** Whichever tile dimension
varies fastest across leaves is the one that the leaves sharing a
hardware node differ in. MRAM is per-DPU and shared across its
tasklets, and `--convert-cnm-to-upmem` decides *syntactically* — from
whether the scatter map mentions the tasklet dimension
(`isMramBroadcastOverThreads`) — whether a buffer is stored once per DPU
or replicated per tasklet. So:

- *Split dimension innermost*: a DPU's tasklets differ in their
  reduction tile. Any operand indexed only by reduction dimensions —
  gemv's vector — is replicated once per tasklet.
- *Split dimension outermost*: a DPU's tasklets differ in their parallel
  tile and share their reduction tile, so that operand is stored once
  per DPU.

The evidence picks the second. The gemv_64MB optimum found by an
independent autotuner
([dodo.py](../experiments/gemv_microbenchmark/dodo.py)) has
`taskletCols = 1` — tasklets splitting rows, sharing the vector — and
with the split dimension innermost that configuration is not reachable
at all: the linearization forces `taskletCols = min(k_tiles, tasklets)`.
With it outermost, the per-DPU footprint comes out at
`8192 + 128 + 64` elements, which is exactly the template's own MRAM
constraint `mramRow*mramCol + mramCol + mramRow`. Pinned by
[gemv-split-k-grouping.mlir](../test/Transform/UPMEM/gemv-split-k-grouping.mlir).

This depends on the scatter map actually *simplifying* to something free
of the tasklet dimension. Delinearizing a workgroup index yields
`(dpu * T + tasklet) floordiv S`, which is constant in `tasklet`
whenever `T | S` but still names it; `simplifyAffineExprWithBounds`
gained the rule that discharges this (`ae698cc`). Without it the
ordering decision above has no observable effect.

**What is given up.** Partials of one output region are no longer
adjacent in the leaf index — they are `m_tiles` apart — so device-side
merging (§G10) will have nothing local to merge. That costs nothing
while merging is host-side, and it is precisely the point at which this
order should stop being a rule and become a parameter. Note the
symmetry: the two orders trade broadcast-operand sharing against merge
locality, and only one of the two is currently exploitable.

The general loss remains the *ratio*: linearization is greedy, so it
cannot independently choose how a hardware level splits between two tile
dimensions. With the split dimension outermost the reachable corner is
`taskletCols = max(1, tasklets / m_tiles)`, which covers the
configurations we care about today but is not the full template space.

**This is not purely a performance knob**: per-tasklet replication
changes the per-DPU MRAM footprint, so the order can change
*feasibility* under capacity constraints, not just speed. Every
factorization `(m_tiles, k_tiles)` stays reachable, but under the wrong
order not every one stays *fittable*. That is why the rule above is
settled by evidence rather than by taste.

**When to revisit.** Two triggers, either of which makes the order a
parameter rather than a rule:

- **Device-side merging (§G10)**, which is what the current order gives
  up.
- **An op where both trades bite at once** — one large broadcast operand
  *and* a reduction worth merging locally. gemm with a shared operand is
  the likely first case.

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

That last clause was aspirational as first implemented: `cnm.scatter`'s
shape rule still forced this pass to materialize a tiled layout of its
own. §J removes the rule and makes it literally true.

### G10. Deferred

Device-side tree reduction of partials — host merge only for now. A
pure optimization over the same parameterization; it can land later
without changing the parameter set. It is, however, the trigger for
revisiting G3.

## H. Capacity constraints, and the generic op handler

### H1. What the current constraints encode

`handleGemv` states the MRAM budget by hand
([UpmemInferAccelerator.cpp:597](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L597)):

    mramRow*mramCol + mramCol + mramRow <= MRAM

That expression is not a generic sum of buffer sizes. It encodes a fact
about the layout the lowering produces: the vector is counted **once**
because the tasklets of a DPU share it, while `A` and `y` are counted
per-DPU. It is correct only for that layout.

### H2. A generic handler cannot honestly reproduce it

Deriving the same bound from indexing maps alone leaves two options,
and both are bad:

- **Duplicate the sharing rule in the constraint.** Whether an operand
  is shared follows from §G3's ordering rule plus the block sizes, so it
  *is* computable at space-build time — by restating the lowering's
  decision in a second place, with nothing checking the two agree. That
  is exactly the coupling that produced the M7 defect.
- **Be conservative and count every leaf's buffers.** This over-counts
  shared operands by a factor of `tasklets`. For gemv_64MB it would
  charge the vector 1024 elements instead of 128 and reject the
  autotuner's optimum — a constraint that quietly deletes the best known
  configuration is worse than none.

### H3. The real problem: occupancy is a property of the lowered program

Sharing is only the first of several things that move occupancy and are
decided *during lowering*, not by the configuration:

- whether a scatter map ends up independent of the tasklet dimension —
  which depends on §G3's order **and** on how well the affine simplifier
  folds the delinearized index;
- `--upmem-tile-mram-buffers` promoting with `useFullTileBuffers=false`,
  which can allocate a staging buffer larger than the tile it holds;
- buffer hoisting, promotion to stack, and where deallocations land;
- kernel dedup.

This is not hypothetical. Commit `ae698cc` changed only the affine
simplifier — no parameter, no constraint, no pass ordering — and took
per-DPU MRAM for gemv_64MB from 9280 to 8384 elements. Any a-priori
constraint would have been wrong before it, after it, or both.

So an a-priori capacity constraint is necessarily either **unsound**
(too permissive, and the configuration is discovered to be infeasible
only when the DPU binary fails to link) or **incomplete** (too strict,
silently shrinking the space). It cannot be exact, because the quantity
it constrains does not exist yet.

### H4. Cheap necessary condition, exact late check

**Implemented** as milestone M14. The proposal follows, with one
correction from implementing it.

**A priori, keep only what can never over-estimate.** Assume maximal
sharing. Such a bound is sound as a filter — it prunes configurations
that cannot fit under *any* layout, and never rejects a feasible one.

**Then measure.** Standing decision 8 already has `evaluate()` run the
real lowering, so the lowered IR is in hand: sum what the program
allocates per memory space and reject the trial if it exceeds the level's
capacity. Exact by construction, and it stays correct when the lowering
changes — which is the property the current scheme lacks. This is
`--upmem-check-occupancy`, run last in the back pipeline.

**Correction: "sum the `upmem.static_alloc` sizes" is not the
measurement.** Private WRAM is `upmem.pwram_alloc`, which the translator
emits as a *stack* array, so WRAM costs
`numTasklets * (runtime reserve + Σ private allocs)` on top of the static
buffers, which are shared by the DPU's tasklets. Counting allocations
alone under-counts by a factor of `tasklets` — the opposite error to the
one H2 warns about, and in practice the binding one: the case that
motivated the check is a configuration whose leaf tiles fit WRAM once and
overflow it eight times over. The same computation sizes the stack the SDK
compiles the kernel with, so the two now share one definition
(`UPMEMOccupancy.h`) rather than agreeing by coincidence.

The cost is that infeasible configurations are no longer excluded from
the space, so the optimizer spends trials finding them. That is the
honest trade: an exact late check beats an approximate early one, and
the necessary condition above keeps the waste bounded. The same
measurement also gives the cost model real occupancy rather than
modelled occupancy.

### H5. The generic op handler

Wanted **alongside** the per-op handlers, not replacing them. Its space
is the one §G2 describes and nothing more:

- one block-size variable per iteration dimension, a divisor of that
  dimension's extent;
- `∏(E_i / b_i) == dpus * tasklets`;
- one leaf block size per dimension, dividing the block size;
- the maximal-sharing capacity bound from H4.

It records those directly — no projection, because these *are* the
pass's parameters. It generalizes to any iteration rank and drops
`dpuRows`/`dpuCols`/`taskletRows`/`taskletCols` entirely, which is the
interpretable space to land on once the templates go.

**Decided (option 3).** `eval_solution` used to serialise positionally
([cinmopt.py:214](../experiments/cinm_experiments/cinmopt.py#L214)), so
the space's variable list was an ABI for everything in `experiments/`:
adding generic variables next to the template ones *for the same op*
would shift positions and silently reinterpret every existing params
dict. The three options were:

1. **Generic handler only for ops that have no template handler** —
   gemm, elementwise, anything new. Existing benchmarks untouched.
2. **Handler chosen by `lowering=`**, giving the two paths different
   spaces. Loses the same-configuration comparison that §F makes the
   quality bar.
3. **One space, named rather than positional.** Breaks the positional
   ABI; requires `eval-solution` to accept names.

**Option 3 is what landed**, and it went further than "both sets of
variables": the generic space is the *only* space, and the templates read
its numbers back through the projection above. The reasoning is that the
templates are meant to go, so preserving compatibility with code that
will be removed — for the sake of benchmarks that can simply be rerun —
buys nothing. Named parameters were independently worth it: the
positional coupling had already been a hazard twice.

The variables are named `<op>.<dim><level>` — `gemv.M0`, `gemv.K0`,
`gemv.M1`, `gemv.K1` — with the dimension names taken from the
originating `cinm` op through a `cinm.lowered_from` marker, and op kinds
counted first so a block with two gemvs gets `gemv0`/`gemv1`.

## I. Sequential trips

§G2 required the tile counts to fill the workgroup *exactly*, and parked
sequential passes over a larger problem in `--cinm-tiling`, upstream.
M10 then removed `--cinm-tiling` from the generic pipeline, because
`cnm.tile_sizes` are block sizes and the workgroup takes the whole tile
space at once. Nothing has expressed trips since.

That is not hypothetical: the `cinm2` gemv_64MB configuration needs them.
With `dpus=256, tasklets=4` (1024 leaves) it covers 4096x4096 in 4 passes
over M and 4 over K.

**Decided (option 2 of three):** derive the trips rather than
parameterize them. The space keeps exactly the parameters §H5 gives it,
and the requirement relaxes from

    prod(E_i / b_i) == leaves          to          prod(E_i / b_i) == leaves * trips

with the distribution emitting a host loop of `trips` iterations. The
alternative -- an outer block size per dimension, consumed by a host-side
tiling pass -- is more faithful to §G2 but adds *n* parameters per op for
something the tile counts already determine.

The cost is a second implicit rule alongside §G3: the tile space is
linearized in the §G3 order, the low-order digits index the workgroup and
the high-order digits index the trip loop. Same trade as §G3 -- no new
parameters, at the price of a fixed factorization -- and the same revisit
trigger.

Two consequences that are *not* obvious, both forced by that same
configuration:

### I1. A trip boundary can fall inside a dimension

After the reduction split its §G3-ordered tile counts are `[4, 4096, 1]`
against 1024 leaves. No suffix of that product equals 1024, so the trip
boundary cannot be placed between two dimensions: `m`'s 4096 tiles have
to split into 4 outer and 1024 inner.

So the "trips consume whole outermost dimensions" simplification does not
survive contact with a real configuration. The tiled layout's tile
dimension is instead `tensor.expand_shape`d into `[tripPart, leafPart]`,
the trip parts are permuted to the front, and each trip
`tensor.extract_slice`s its own chunk before scattering.

**Superseded by §J4.** That mechanism is built on the tiled-layout
machinery §J deletes. With a pointwise map a trip is simply an offset:
`extract_slice`/`subview` the host operand per trip and keep the map.
The boundary can still fall inside a dimension — that fact is about the
tile counts, not about the representation — but it no longer needs a
reshape or a permutation to express. M13 therefore depends on the §J
milestone.

### I2. Trips over a reduction dimension accumulate across launches

4 of those 16 trips come from `k_outer` -- a reduction-derived dimension.
Each trip's leaves are seeded with the combiner's identity and merged by
the gather (§G5), but that merge must then accumulate into the *running*
total rather than overwrite it.

So the host loop carries the output as an `scf.for` iteration argument,
and the original `outs` is folded in exactly once at the end -- the same
rule as §G5, one level up. A trip loop whose dimensions are all parallel
needs none of this, and is the easy case; it is not the case the
benchmarks need.

## J. Scatter/gather maps: pointwise into the host buffer

`cnm.scatter`/`cnm.gather` currently require the host value's shape to
*end with* the per-leaf buffer shape, with the map naming only the
leading tile coordinates
([CnmOps.cpp:335-349](../lib/Dialect/Cnm/IR/CnmOps.cpp#L335-L349)). That
one line of the verifier is doing more work than it looks: it hard-codes
a transfer model in which each leaf receives exactly one contiguous run
of `∏B` elements. Three costs follow.

1. **It forces data movement to satisfy a type rule.** An operand tiled
   in two or more dimensions has to be physically permuted into
   tiles-outermost order before it can be scattered, because a `tensor`
   has no layout and the only way to change a tensor's layout is to copy
   it. On gemv_64MB that is a 64 MB allocation plus a 64 MB strided
   copy — see M15.
2. **It understates what the hardware can do.** The UPMEM SDK's scatter
   transfer API takes an arbitrary number of blocks per DPU, and
   `upmem.scatter_blocks` already exposes it: `numBlocksPerDpu` is
   documented as "independent of the number of tasklets declared by
   `hierarchy`: blocks are just units of transfer"
   ([UPMEMOps.td:331-335](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.td#L331-L335)).
   A leaf whose tile is *k* contiguous runs is perfectly transferable as
   *k* blocks. The CNM contract cannot say so, so a copy happens instead.
3. **It is a backend assumption stated at the device-independent
   level.** §G9 already assigns contiguity to
   `--cnm-ensure-scatter-gather-contiguous`; the shape rule pre-empts
   that pass by making non-contiguity unrepresentable.

The templates avoid the copy by hand: `scatterATile` builds a permuted
`memref.reinterpret_cast` view of `A` and picks between a flat
`upmem.scatter_on_array` and `upmem.scatter_blocks`
([SimulationTemplates.cpp:164-231](../lib/Dialect/UPMEM/Transforms/SimulationTemplates.cpp#L164-L231)).
That is sound as far as addressing goes — the lowering composes the
scatter map with the memref's `StridedLayoutAttr` and applies the offset
by GEP
([UPMEMToLLVM.cpp:142-158](../lib/Conversion/UPMEMToLLVM/UPMEMToLLVM.cpp#L142-L158),
[:529-536](../lib/Conversion/UPMEMToLLVM/UPMEMToLLVM.cpp#L529-L536)) — but
it is sound *by argument, not by construction*: building the view on
`extract_strided_metadata`'s base buffer severs provenance, so alias
analysis and buffer deallocation see no link back to `A`. It works only
because `A` is read-only for the lifetime of the nest. It is not a trick
the generic flow should imitate.

### J1. The decision

**The map becomes pointwise into the host buffer**, with a block form
that leaves a suffix of the buffer's dimensions implicit.

Let the workgroup shape be `W` (`n` dims), the `cnm.buffer` shape `B`
(`m` dims) and the host value's shape `H` (`k` dims). The map's domain is
the workgroup dimensions followed by the first `p` of the buffer's own,
for any `0 <= p <= m`. The `m - p` dimensions left out of the domain are
transferred as a **block**, and the same number of host dimensions are
left out of the results — they are the block's own shape, so they have to
match extent for extent:

    (w_0..w_{n-1}, i_0..i_{p-1}) -> (h_0..h_{k-(m-p)-1})

    buf[w][i_0..i_{p-1}, rest] = host[map(w, i_0..i_{p-1}) ++ rest]

`p = m` is the **pointwise form**, naming a host element for every buffer
element and leaving nothing implicit. `p = 0` is one whole-buffer block
per leaf — which spelled out is `numResults == k - m` and `H[k-m..] == B`,
i.e. **today's contract exactly**. The change is therefore purely
additive: existing IR is already well-formed under it, and no migration
shim is needed.

The transfer unit falls out: **one block of shape `B[p..m)` per (leaf,
retained index) pair**, so blocks per DPU = `tasklets * ∏B[0..p)`.
Choosing the transfer API stops being a codegen guess and becomes a
reading of `p`: `p = 0` is a flat `upmem.scatter`, `p > 0` is
`upmem.scatter_blocks` with that block count. This answers §B's
third bullet.

**Why the block is index-structured rather than linear.** The tempting
alternative keeps all `k` results as the *origin* of a run of `∏B[p..m)`
consecutive elements. That is strictly more expressive — it describes a
contiguous run at an arbitrary offset inside a dimension, where the form
above needs the block to cover whole dimensions — but the meaning of such
an op depends on the host value's strides, so the same IR denotes
different transfers under different layouts. That is exactly the kind of
backend assumption this section exists to remove from a
device-independent dialect. Where a maximal contiguous run does not align
with dimension boundaries, a free `memref.expand_shape` makes it align,
with the side benefit of putting the block's shape in the type.

What is and is not a well-formedness condition follows from that choice.
The block is a sub-array named by indices, so nothing about contiguity is
checkable — or needs to be — at this level. Contiguity is a question only
when *deriving* a larger block from a pointwise map, and that is an
analysis (§J2). What the verifier does owe is bounds: the old shape rule
gave them away for free, and they now have to be computed.

### J2. Deriving the block form

`--convert-linalg-to-cnm` emits the pointwise form and nothing else:
choosing a transfer shape is a layout question, and it runs before layouts
exist. After bufferization,
`--cnm-ensure-scatter-gather-contiguous` becomes the pass that earns its
name — for each scatter/gather it computes the **largest implicit
block**, i.e. the largest `m - p` such that composing the map with the
host memref's layout makes those `m - p` dimensions one contiguous run.
Where the run does not align with host dimension boundaries, a
`memref.expand_shape` makes it align at no cost. That is `linearizeToElementOffset` +
`taskletBlocksAreContiguous`
([CnmToUPMEM.cpp:114-180](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L114-L180))
generalized from a yes/no answer to a count, in the same spirit as
`getContiguousSuffixSize`. Only if the resulting block count exceeds
what the backend will accept does it fall back to inserting a packing
buffer — which is what it does unconditionally today.

**Normalize before analyzing.** Fold `memref.subview` / `expand_shape` /
`collapse_shape` chains into the map and point the scatter at the base
memref, so the analysis sees one identity-layout buffer and one map
rather than a composition over a chain of views. This makes the
criterion a property of the map alone, and it is what lets the pointwise
map *subsume* `memref.reinterpret_cast`: any strided view of the host
buffer is expressible as an affine map into the unviewed buffer, with
the stride arithmetic living in an attribute instead of in a forged
memref type. The provenance problem in `scatterATile` disappears rather
than being reproduced.

One boundary, and the existing lowering already draws it in the right
place: **static structure folds into the map, dynamic offsets do not.**
The map is outlined as a function of the DPU (and block) index alone
([UPMEMToLLVM.cpp:410-446](../lib/Conversion/UPMEMToLLVM/UPMEMToLLVM.cpp#L410-L446)),
with the base offset applied by GEP — that is what "offset is calculated
outside of the affine map" means. A subview with a loop-variant offset
therefore keeps its offset in the memref's dynamic offset field. Putting
dynamic values *inside* the map would need symbol operands on the op and
a wider runtime callback signature; stay on the near side of that line.

### J3. Scatter and gather are not symmetric

Under the shape rule this was implicit. Stated explicitly:

- A **scatter** map may be non-injective. Two leaves reading the same
  host element is a broadcast, which is the point.
- A **gather** map must be **injective** over the workgroup × buffer
  index box, or two leaves race to write the same host element.

Verified where it can be. Linearize the host index so the map becomes one
expression over the box of workgroup × retained-device dimensions, with
the block sweep folded in as one more dimension of stride 1 — then both
this obligation and the bounds one below are questions about a single
expression.

**Injectivity** is decided when that expression is a plain `Σ c_j·d_j +
c`: any dimension of extent > 1 with a zero coefficient is a collision
outright, and otherwise, sorted by `|c_j|`, each term must clear the
previous ones' reach (`c_{j+1} > Σ_{j' <= j} |c_{j'}|·(extent_{j'} - 1)`)
— the mixed-radix non-overlap condition, `O(n log n)`.

**Bounds** are computed by interval arithmetic instead, which also copes
with floordiv and mod. That matters: §G3's tile linearization puts a
`floordiv`/`mod` pair in essentially every map this flow produces, so a
plain-sum matcher would decline on exactly the maps that occur. The
requirement is `upperBound(origin + sweep) < hostElementCount`.

Both checks decline rather than reject when they cannot analyze the
expression — the same stance `getContiguousSuffixSize` takes for
unsupported layouts. **The consequence is that injectivity is usually
*not* checked in practice**, since the maps that occur carry floordiv/mod.
That is tolerable because injectivity there is a property of how
`--convert-linalg-to-cnm` builds the map (§G3's mixed-radix
decomposition of the leaf index is a bijection by construction); the
verifier check is a safety net for hand-written IR and for the simpler
maps the specializations produce. If it ever needs to be exact, the
route is to inflate the tile linearization back into per-dimension
coordinates before matching, not to teach the matcher about floordiv.

### J4. What this changes elsewhere

- **`--convert-linalg-to-cnm` shrinks.** `toTiledLayout`,
  `fromTiledLayout`, `tiledLayoutShapes` and `isLayoutPreserving`
  ([LinalgToCnm.cpp:64-151](../lib/Conversion/LinalgToCnm/LinalgToCnm.cpp#L64-L151))
  exist only to satisfy the shape rule and all delete. The map it emits
  becomes the composition §G9 already describes, extended with `+ i` on
  the intra-tile dimensions.
- **§I1 simplifies, and M13 must come after this.** §I1's mechanism —
  `expand_shape` the tile dimension into `[tripPart, leafPart]`, permute
  the trip parts to the front, `extract_slice` per trip — is built on
  exactly the machinery this section deletes. Under a pointwise map a
  trip is an offset: `extract_slice`/`subview` the host operand per trip
  and keep the map, with the dynamic trip offset riding in the memref's
  offset field, the same way `mOff`/`kOff` do in the templates. No
  expand_shape, no permutation. **Sequence M13 after this milestone** or
  it will be written against machinery that is about to disappear.
- **`--convert-cinm-to-cnm` is adapted, not rewritten.** A helper
  inflates an old-style map into the block form (it is the `p = 0` case),
  so only one representation exists in the IR and the compatibility lives
  in a function rather than in the verifier.
- **The uniform-scatter rewrite is reformulated.** "Shrink the value and
  empty the map" becomes "replace the host-index results with
  constants". Note that on tensors it detaches the scattered value from
  the buffer a later gather writes into, which is precisely the
  split-reduction accumulator case — so it is likely to end up as a
  memref-level rewrite, where that identity is explicit, or to be
  superseded by a device-side fill.

## K. UPMEM transfer ops: one general form, specialized late

§J made the transfer *shape* a reading of the CNM map. This section is
about the other half: which of the four UPMEM transfer ops that shape
should be spelled with, and where that choice is made.

### K1. The general form is the only form the conversion emits

`upmem.scatter_on_array`, `upmem.gather_on_array`, `upmem.scatter_blocks`,
`upmem.gather_blocks` and `upmem.broadcast` are not five capabilities, they
are one capability at three levels of specificity:

```
scatter_blocks(host, map(r,d,b), blockSize, numBlocks)   -- dpu_push_sg_xfer
  |  blocks are adjacent, in order  ==>
scatter(host, map(r,d), numBlocks*blockSize)              -- flat memcpy per DPU
  |  map is constant zero and covers all of host  ==>
broadcast(host)                                           -- one broadcast call
```

Today the choice is made *during* `cnm → upmem`, by
`convertCnmScatterToUpmem`
([CnmToUPMEM.cpp:259-330](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L259-L330)),
and independently again by hand in the simulation templates
([SimulationTemplates.cpp:215-232](../lib/Dialect/UPMEM/Transforms/SimulationTemplates.cpp#L215-L232)
and [1229-1232](../lib/Dialect/UPMEM/Transforms/SimulationTemplates.cpp#L1229-L1232)).
Two emitters, one rule, no shared code.

**Decision: the conversion emits only the general form, and the UPMEM
dialect specializes it.** The conversion then has no branch: block size
and block count both come straight off the CNM map, and
`taskletBlocksAreContiguous`, `getAffineExprDimCoefficient` and
`isGloballyBroadcast` move into the dialect, where they get *shorter* —
they currently have to rebuild the `(r, d, t)` map with
`keepTaskletDimAffineMapCnmToUpmem` before they can reason about it,
and after the move that map is the op's own attribute.

The `scatter → broadcast` step gets cheaper in the same way. It is
currently gated on `isBroadcast`, a fact about the `cnm.alloc`'s users
(does the MRAM buffer have a per-tasklet leading dimension). By the time
there is a `upmem.scatter`, that fact is already baked into
`transferCount`, so the condition is local: constant-zero map, and
`transferCount == host.getNumElements()`.

**Naming, as landed.** `scatter_on_tasklets` was a misnomer: its own
description said blocks "need not correspond 1:1 to actual DPU tasklets",
and under §J a leaf routinely takes several. The general pair is
`upmem.scatter_blocks` / `upmem.gather_blocks`, with the map's third
dimension named `block` to match `numBlocksPerDpu`. (`_sg` was the other
candidate; it names the SDK call, `dpu_push_sg_xfer`, but reads as
"scatter scatter-gather" in a dialect that already has `scatter` and
`gather`. The SDK call belongs in the op description.)

The one-block-per-DPU pair is `upmem.scatter_on_array` /
`upmem.gather_on_array`: it spreads one block across the array, and its
plain name would otherwise suggest it was the general case. The runtime's
`kind` column (timers.h) uses the same three words -- `on_array`, `blocks`,
`broadcast` -- so a CSV row names the op that produced it.

**Gather needs the general form too.** There was no counterpart to
`scatter_blocks` on the gather side, so
`convertCnmGatherToUpmem`
([CnmToUPMEM.cpp:227-257](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L227-L257))
emits a flat `upmem.gather` at `transferCount * numTasklets`
unconditionally and checks nothing. `dpu_push_sg_xfer` runs both
directions, so `upmem.gather_blocks` is the symmetric fix and belongs in
the same change — otherwise "the conversion only emits the general form"
stays a scatter-only claim and the two paths keep diverging.

### K2. All five ops share one contiguity contract

The verifiers do not currently agree on what they are checking.
`verifyScatterGatherContiguity`
([UPMEMOps.cpp:134-146](../lib/Dialect/UPMEM/IR/UPMEMOps.cpp#L134-L146))
asks a question about the *type*: is `transferCount` at most the memref's
contiguous suffix. But whether a transfer is contiguous depends on where
the map *starts*, not only on how long the run is. For
`memref<4x3x8xi32, strided<[100, 8, 1]>>` the suffix is 24, so a
`transferCount` of 24 passes — and it is wrong for any map whose last two
results are not `(0, 0)`, because the run then crosses the gap at
element 24.

The honest condition is the same one §J1 gave the CNM ops, and it is the
same for all five:

> Decompose the host memref into its regular grid of contiguous runs of
> `C = getContiguousSuffixSize` elements. Linearize the map's trailing
> results against that packed suffix, take the affine upper bound over
> the op's own index box, and require `maxInnerOffset + blockSize <= C`.

The index box is known in every case: the hierarchy type gives
`ranks × dpus`, and the `_blocks` forms add `numBlocksPerDpu`. The
block is `transferCount` in all four ops — that is what `transferCount`
already means for `scatter_blocks`, and after specialization it is
what it means for `scatter`, since a specialized flat transfer *is* one
block. So one helper serves all four and the arity of the box is the
only difference.

This also supplies the **bounds** check none of the four has today — the
transfer staying inside the memref at all — which is the check M16 added
to `cnm.scatter`/`cnm.gather`. The UPMEM ops end up verifying the
UPMEM-level statement of the same property, which is the point.

`getAffineUpperBound` and `linearizeToElementOffset` are both
dialect-neutral (one is interval arithmetic on `AffineExpr`, the other is
memref-layout linearization) and currently sit in
`Dialect/Cnm/IR/CnmScatterMap.cpp` and `Conversion/CnmToUPMEM/` respectively.
They move to `Utils/` so the UPMEM dialect can use them without
depending on CNM.

### K3. A pass, not a canonicalization — for now

Specialization is *not* a canonicalization, because it has to be
switchable: `use-sg-xfer-codegen` and `use-bc-xfer-codegen` are the
CINM 1.0 baseline's controls, both driven from one `use_upmem_scatter_api`
knob ([cinm1.py:111-129](../experiments/cinm_experiments/cinm1.py#L111-L129)).
A canonicalization pattern fires in every `--canonicalize` and cannot be
turned off. So the patterns go in `--upmem-specialize-transfers`, carrying
the two existing option names.

**This is temporary and should be recorded as such.** The switch exists
only to reproduce a baseline for the paper; once that measurement is
locked, the pass becomes `hasCanonicalizer = 1` on the two `_blocks` ops
and the flags disappear. Listed with the other post-paper cleanups in the
implementation plan.

**The "off" switch is currently a miscompile, and the move fixes it.**
`use-sg-xfer-codegen=false` today does not mean "do not specialize", it
means "collapse to the flat form *even when that is wrong*" — the
`NOSG` line of
[cnm-to-upmem-sg-xfer.mlir:22](../test/Conversion/CnmToUpmem/cnm-to-upmem-sg-xfer.mlir#L22)
pins a 16-element flat transfer for a map whose two tasklet blocks have a
one-row gap between them. It reads the wrong bytes, deliberately, to
reproduce pre-sg-xfer behaviour.

Under K1 that cannot happen by construction: a pattern only fires when
its precondition holds. The correct reading of the flag is "the sg
runtime call is not available; make the transfers flat upstream" — which
is what `cinm1.py` already does, running
`--cnm-ensure-scatter-gather-contiguous` in step 5 exactly when the flag
is off. So with the flag off and specialization impossible, the pass
should **emit a diagnostic**, not a wrong transfer: the packing that was
supposed to happen upstream did not. The `NOSG` expectation is rewritten
to that diagnostic. This should be inert for the CINM 1.0 configurations,
whose shape-suffix distribution packs upstream anyway — worth confirming
by running them, since it is the one behavioural change visible to the
baseline.

### K4. What this changes elsewhere

- **The simulation templates stop choosing.** Both hand-written branches
  emit the general form and let the pass specialize. They were the
  second, uncoordinated copy of the rule.
- **`--cnm-scatter-optimizations` composes.** Its broadcast rewrite
  (§B, second bullet) can rewrite CNM-level scatters freely without
  reasoning about which UPMEM op will result; whether a
  `upmem.broadcast` comes out is settled downstream by K1's third step.
- **`adaptAffineMapCnmToUpmem` survives only in the launch-argument
  path.** Both transfer paths use `keepTaskletDimAffineMapCnmToUpmem`
  (renamed for the block dimension) once the conversion stops collapsing.

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
7. §I: trips are derived from the tile counts by a fixed rule, which is
   now the second such rule after §G3. Worth revisiting both together if
   either turns out to cost real performance. (Open; the rule itself is
   decided and is implementation milestone M13.)
8. ~~§H5: how should the generic handler coexist with the per-op ones,
   given that `eval-solution` is positional?~~ **Decided:** option 3, and
   further than stated — the generic space is the *only* space,
   `eval-solution` takes names, and the templates read the same numbers
   back through §H5's projection. Milestone M11b.
9. ~~§H4: adopt the "maximal-sharing necessary condition + exact
   post-lowering occupancy check" scheme, and retire the hand-written
   per-op capacity constraints?~~ **Decided: yes, and done** (M14). The
   necessary condition is in the space and `--upmem-check-occupancy`
   measures the lowered program. The hand-written per-op constraints
   survive only on the templates path, which does not run the pipeline and
   is scheduled for removal.
10. §G8: with fusion on the generic branch only, the two paths no longer
   share an op set, so §F's same-configuration cost comparison weakens.
   Should the comparison baseline be the *unfused* generic path? (New,
   open.)
11. ~~§G4: reuse `linalg::splitReduction` for the partial+merge rewrite,
   or go through `PartialReductionOpInterface`?~~ **Decided:**
   `linalg::splitReduction`. It turns the reduction tile index into an
   extra *parallel* dimension and leaves every indexing map a projected
   permutation, so §G1's distribution applies unchanged and no
   `PartialReductionOpInterface` is needed. It covers §G5 for free
   (neutral-element fill, merge accumulating into the original `outs`).
   Its extra-dim placement is controllable and §G3 pins it outermost.
12. §E: merge with `UpmemGenericLoweringNotes.md` once §A3 is settled,
   or keep the direct `linalg.generic → upmem` path as a permanent
   parallel option? (§A3 being settled now makes this more concrete:
   the note's points 1-3 map onto the two `--cinm-tiling` passes'
   role-assignment/reduction-split policy — worth a pass to check how
   much of the note is now answered.)
13. ~~§J: keep `cnm.scatter`'s shape-suffix contract, or give the map a
   general index into the host buffer?~~ **Decided:** pointwise maps
   `(*wgDims, *bufferDims) -> (*hostDims)`, with a block form that leaves
   a suffix of the buffer dimensions — and the host dimensions they cover
   — implicit. Today's contract is the `p = 0` case, so the change is
   additive. §B's third bullet (transfer API selection) falls out of how
   many buffer dimensions the map retains. Milestone M16, and M13 is now
   sequenced behind it.

---

<!-- Add your comments below or inline as blockquotes -->

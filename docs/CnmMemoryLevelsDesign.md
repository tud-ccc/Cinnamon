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
  loops directly — it emits actual **cinm ops** operating on those
  memrefs (e.g. a smaller `cinm.gemv` over `memref<mramRow x mramCol,
  #upmem.mram>`), as long as they implement the existing
  **`CinmTilingInterface`**
  ([TilingInterface.td](../include/cinm-mlir/Dialect/Cinm/IR/TilingInterface.td)).
  This interface and its pass (`--cinm-tiling`,
  [TilingPass.cpp](../lib/Dialect/Cinm/Transforms/TilingPass.cpp))
  already exist and already do exactly "tile this op down into a loop
  nest around a smaller instance of itself, given a `cinm.tile_sizes`
  attribute" — today it's used once, pre-bufferization, to tile a
  host-level tensor op down to what becomes the workgroup tile. The
  new idea is to **apply it a second time**, post-materialization,
  *inside* the launch body: run `--cinm-tiling` again (now over
  memref-operand ops) with tile sizes equal to the WRAM tile
  (`wramRow`/`wramCol`), producing a loop nest around an
  innermost `cinm.gemv` that is exactly WRAM-sized, still typed
  `memref<wramRow x wramCol, #upmem.mram>` (tiling only changes the
  loop bounds/subview offsets, not the memory level of the operand).
- A new pattern then closes the gap: when it sees a tileable op whose
  operand memref is already at the desired leaf tile size but tagged
  with a non-leaf level, it allocates a new **private WRAM** buffer of
  that tile size, inserts a `cnm.local_transfer` (§A4) from a
  `memref.subview` of the MRAM memref into it before the op, rewrites
  the op to use the WRAM buffer, and inserts the reverse transfer
  after (for outputs). This is the piece that replaces the hand-coded
  `mr`/`mc` loop body of `generateGemv`.
- **Deferred optimization:** when the scatter map shows a buffer is
  broadcast across tasklets of a DPU (same tile for every tasklet on a
  DPU, DPUs may still differ), the private-WRAM buffer above can
  instead become a single shared static allocation guarded by a
  barrier (mirrors `isWramShared`/`PrivateWRAMAllocOp` vs. static
  alloc in `CnmToUPMEM.cpp` today). Left for later — it may end up
  living at the UPMEM dialect level rather than CNM, partly because
  CNM has no barrier primitive yet (§C already flags this gap).

This reuses existing, working machinery (`CinmTilingInterface` +
`--cinm-tiling`) for both levels of tiling instead of inventing a
second mechanism, which is a nice simplification over my original
framing of this as a from-scratch design.

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
7. §E: merge with `UpmemGenericLoweringNotes.md` once §A3 is settled,
   or keep the direct `linalg.generic → upmem` path as a permanent
   parallel option? (§A3 being settled now makes this more concrete:
   the note's points 1-3 map onto the two `--cinm-tiling` passes'
   role-assignment/reduction-split policy — worth a pass to check how
   much of the note is now answered.)

---

<!-- Add your comments below or inline as blockquotes -->

# CNM refactoring: overview

A short index of the `cinm → linalg → cnm → upmem` refactoring. The full
reasoning is in [CnmMemoryLevelsDesign.md](CnmMemoryLevelsDesign.md)
(§A–§I) and the milestone-by-milestone record is in
[CnmMemoryLevelsImplementationPlan.md](CnmMemoryLevelsImplementationPlan.md).
This file is only the map: what changed, what was broken, what is left.

**The thesis.** Passes expose *parameters*; they do not take decisions.
Every distribution decision that `--convert-cinm-to-cnm` used to make by
case analysis is now a number the UPMEM inference plugin supplies, and
the plugin derives those numbers from the op's own iteration space rather
than from a per-op template.

## 1. Design changes

### Memory levels

- **`CinmLevelAttrInterface`** identifies a level; capacity stays on the
  platform in `CinmLevelDefAttr`. `#upmem.mram`/`#upmem.wram` implement
  the interface — no new attributes, and the level attribute doubles as
  the memref memory space it already was. (§A1)
- **`--convert-cinm-to-cnm=cnm-buffer-level=<name>`** — one pass-wide
  level, resolved against the platform. `cnm.buffer`s and `cnm.launch`
  block arguments carry it. Empty = the old behaviour, byte-identical.
  (§A2)
- **`cnm.local_transfer`** — memref-to-memref, direction implicit in the
  operands' memory spaces. Tiling uses ordinary `memref.subview`; no new
  CNM subview op. (§A4)
- **Launch bodies stay `linalg`.** The original plan put memref-mode
  `cinm` ops in them; that was implemented and reverted. Selecting a
  level changes *where buffers live*, not what the body is — and the body
  is often a dot product no `cinm` op can express anyway. Staging keys
  off the memory space instead.
- **`--upmem-tile-mram-buffers`** (new UPMEM pass) — tiles any
  buffer-semantics `LinalgOp` whose operands are in a non-leaf level,
  then `linalg::promoteSubViews` with a custom allocator (UPMEM needs
  static shapes) and `cnm.local_transfer` as the copy op. Promotion runs
  in two stages so the output tile is staged outside the reduction loop
  and written back once, as the templates do.
- **`--convert-cnm-to-upmem`** branches on the buffer's level: MRAM-level
  launch arguments bind directly to the static alloc (whole, or a
  tasklet-indexed subview) with no second staging round.

### Distribution

- **`--convert-linalg-to-cnm`** (new) replaces the heuristics. Parameter:
  `cnm.tile_sizes`, one **block size** per iteration dimension. The pass
  checks `∏(E_i / b_i) == |WG|`, builds each scatter map as
  `indexingMap ∘ (workgroup → tile)`, and gives each operand the block
  sizes of the dims in its own map. No case analysis, no reshape
  guessing — contiguity is `--cnm-ensure-scatter-gather-contiguous`'s
  job. (§G1–G3)
- **Reduction splitting** across the workgroup, via `linalg::splitReduction`.
  It turns the reduction tile index into an extra *parallel* dimension and
  keeps every map a projected permutation, so distribution applies
  unchanged; it also does the identity seeding and folds the original
  `outs` in exactly once at the merge. Associative+commutative combiner
  required; floats behind an opt-in flag. (§G4–G6)
- **§G3's ordering rule** — split reduction dims outermost, then the
  original parallel dims, then the unsplit reduction remainder. Fixed by
  rule, not searched. Which tile dim varies fastest decides which
  operands a DPU's tasklets share, and `--convert-cnm-to-upmem` reads
  that syntactically off the scatter map. The current rule reproduces the
  autotuner's grouping (`taskletCols = 1`).
- **Linalg-first pipeline.** The generic branch is
  `--convert-cinm-ops-to-linalg` → `--convert-linalg-to-cnm`, not
  `--cinm-tiling` → `--convert-cinm-to-cnm`. The host-side tiling round
  disappears (the workgroup takes the whole tile space at once) and with
  it `stampLeafTileSizes` — both levels are stamped up front on the same
  op and ride the conversions.

### Search space

- **One generic space, derived from the linalg op.** One block-size
  variable per iteration dimension per level; no
  `dpuRows`/`dpuCols`/`taskletRows`/`taskletCols`. Generalizes to any
  iteration rank. (§H5)
- **Named `<op>.<dim><level>`** — a gemv declares `gemv.M0, gemv.K0,
  gemv.M1, gemv.K1`, level 0 being the block a workgroup leaf gets and
  level 1 the block it walks that in at the leaf memory level. Dimension
  names come from the originating `cinm` op via a new `cinm.lowered_from`
  marker, falling back to `D0, D1, …`.
- **`eval-solution` takes names, not positions.** The positional
  encoding was an ABI for everything in `experiments/`: adding one
  variable silently reinterpreted every existing params dict. It had
  already been a hazard twice.
- **The templates read the same numbers.** `lowering=templates|generic`
  select the two lowerings of *one* space, so the same configuration can
  be costed both ways — that comparison is what says when the templates
  can go. The projection is (§H5):

      taskletRows = min(tasklets, mTiles)   taskletCols = tasklets/taskletRows
      mramRow     = blockM * taskletRows    mramCol     = blockK * taskletCols
      dpuRows     = mTiles / taskletRows    dpuCols     = kTiles / taskletCols
      wramRow     = leafM                   wramCol     = leafK

- **Capacity is not knowable a priori** (§H3). Occupancy is a property of
  the *lowered* program: sharing, `useFullTileBuffers`, hoisting, dedup
  and even the affine simplifier move it. `ae698cc` changed only the
  simplifier and took per-DPU MRAM from 9280 to 8384 elements. So the
  plan is a cheap necessary condition (assume maximal sharing — can never
  over-estimate) plus an exact post-lowering check. Only the first half
  exists today.

## 2. Bug fixes

Latent ones first — several were invisible until this work reached them.

- **`cnm.launch` did not round-trip levelled buffers.** The shorthand
  buffer type printed shape and element type only, while the parser
  accepted an optional level. Any `cinm-opt | cinm-opt` on levelled IR
  failed. (`0f5dede`)
- **Tiled reductions dropped every trip but the last.** With a shaped
  result, `ReduceTilingModel` overwrote the accumulator slice instead of
  combining into it, and seeded it with `tensor.empty()` rather than the
  identity. Unreachable until something tiled a reduction dimension into
  more than one trip. (`3caea2d`)
- **Reduce scattered a zero init for every method**, so a `mul`
  reduction always returned zero. Now `arith::getIdentityValueAttr` for
  the method. (`71cd9c1`)
- **The launch body's reduction dimension was hardcoded to 0.** Correct
  only when the buffer holds nothing but the reduction; wrong as soon as
  the per-leaf tile has a parallel dimension too.
- **`--convert-cinm-ops-to-linalg` dropped discardable attributes**, and
  **`linalg::splitReduction` builds a fresh op**, so the leaf tile sizes
  vanished *exactly* for the configurations that split a reduction —
  slower code, no diagnostic. (`1d1d094`, `190678e`)
- **`simplifyAffineExprWithBounds` could not fold a `floordiv` of a sum.**
  Without the rule the delinearized scatter index never simplified to
  something tasklet-independent, so `isMramBroadcastOverThreads` said no
  and the shared operand was replicated per tasklet. Load-bearing: it is
  what makes §G3's ordering observable at all, and it is worth 9280 →
  8384 elements of per-DPU MRAM on gemv_64MB. (`ae698cc`)
- **The UPMEM C translator computed subview offsets from the subview's
  own sizes instead of the source's strides.** For
  `subview %buf[%t,0,0,%k] [1,8,1,64]` of `memref<8x8x1x128xi32>` it
  emitted `%k*64 + %t` where `%t*1024 + %k` was meant — silently wrong
  MRAM addresses. It now uses `getStridesAndOffset`, and **rejects nested
  subviews loudly** instead of quietly mis-addressing them; the generic
  path runs `--fold-memref-alias-ops` (not `--canonicalize`, which does
  not do this) to compose them first. (`b6cbc0b`, `a12f33c`)
- **`affine.for` survived into DPU kernels** on the generic path;
  `--lower-affine` now ends the back pipeline. (`85371ca`)
- **`isZeroSplatFoldable` could not see through `linalg.fill`**, so the
  identity seed defeated the constant-scatter fold. (`eececea`)
- **A stale test asserted gating that never existed** — the
  `cinm1-codegen` option was documented (in a code comment) as disabling
  the `upmem.broadcast` shortcut, which is gated on `use-bc-xfer-codegen`
  alone. The comment was wrong, not the code. (`d6b8fca`)

## 3. Deferred optimizations and remaining TODOs

Nothing here blocks the refactoring; this is the list to work through
once it lands. Roughly in priority order.

### Correctness / completeness gaps

- **Sequential trips** (§I) — decided (derive from the tile counts), not
  implemented. Needs `expand_shape` on the dimension the trip boundary
  falls inside, and an `scf.for` iter_arg for trips over a reduction
  dimension. **Blocks the `cinm2` baselines**, which are commented out in
  `experiments/gemv_microbenchmark/dodo.py` until this exists.
- **Exact post-lowering occupancy check** (§H4) — a transform that sums
  the real `upmem.static_alloc` sizes per memory space and returns
  `DiagnosedSilenceableFailure::silenceableFailure`. Without it,
  infeasible configurations are only discovered when the DPU binary fails
  to link.
- **The 64 MB host repack.** `--convert-linalg-to-cnm`'s `toTiledLayout`
  materializes a permuted copy of any operand tiled in ≥2 dimensions,
  because `cnm.scatter` requires the host value's shape to *end with* the
  buffer shape. On gemv_64MB that is a full copy of `A` and **77.9 ms of
  the generic path's 103.4 ms**. Two candidate fixes: give `cnm.scatter` a
  general affine map into the host buffer, or emit a strided view and let
  `--cnm-ensure-scatter-gather-contiguous` decide where packing is really
  needed.
- **The zero-seed scatter** (`0d9413d`) — the identity seed is a constant
  buffer scattered to every leaf on every launch. `cnm.set_zero` is the
  right answer but only `--convert-cnm-to-gpu` lowers it; emitting it
  makes `--convert-cnm-to-upmem` fail **silently**. So this also wants
  *`--convert-cnm-to-upmem` to diagnose ops it cannot legalise*.
- **`--upmem-tile-mram-buffers` promotes with `useFullTileBuffers=false`**,
  so a tile that does not divide its extent gets a full-size staging
  buffer and a partial view of it. Correct, but larger than the tile.
- **`--convert-linalg-to-cnm` requires projected permutations.** Dropping
  a dimension is fine (that is broadcast); repeating one is rejected.

### Deferred optimizations (§B, §C, §A3, §G10)

- **Broadcast detection** — a scatter map that does not depend on the PE
  dimension should specialize `cnm.scatter` into a broadcast. Pure
  canonicalization on today's CNM IR. Concretely outstanding: the
  template path uses a broadcast transfer for the gemv vector
  (`scatter:bc`) and the generic path does not, so it pays a full scatter
  for the same data.
- **Constant-scatter simplification** — a homogeneous constant becomes a
  broadcast of a smaller tile, or an on-device init loop / static array
  when small enough. This is the general form of the zero-seed item above.
- **Blocked vs. multi-block transfer API selection**, decided generically
  at the CNM level from the scatter map.
- **On-device transfer coalescing** (§C) — merge sibling per-tasklet
  `cnm.local_transfer`s into one leader transfer plus a barrier. **CNM
  has no barrier primitive**; this and the row-leader merge already
  hand-written in the reduce template both need one, so a barrier op is
  likely a second CNM primitive this effort owes.
- **Share the WRAM staging buffer across tasklets** when the scatter map
  is tasklet-independent (§A3) — static alloc plus a barrier instead of
  private WRAM per tasklet. Always private today.
- **Device-side tree reduction of partials** (§G10) — host merge only for
  now. A pure optimization over the same parameters, but it is the
  trigger for revisiting §G3.

### Design work still owed

- **§G3 as a real search parameter.** The tile-dim → workgroup-axis
  assignment is a *permutation*, not an integer factor, so surfacing it
  needs a first-class representation with a sensible distance function
  (cf. BACO). Fixed by rule for now. §I adds a second such fixed rule
  (the trip/leaf factorization); worth revisiting both together if either
  costs measurable performance.
- **Linalg fusion on the generic branch** (§G8). Deliberately not added:
  the current benchmarks are single-op compute blocks so it would be a
  no-op, while enabling it forces search-space construction to move after
  the conversion (fusion changes the op count, the walk indices *and* the
  iteration space). The pipeline is arranged so it slots in right after
  the linalg conversion. It also weakens §F's same-configuration
  comparison, since the two paths would no longer share an op set —
  open whether the baseline should then be the *unfused* generic path.
- **Cycle-accurate cost comparison.** `op-count` models neither loads nor
  stores, so it cannot answer "is the generic path as good as the
  templates" — which is the criterion for deleting the templates.
- **Retire `--convert-cinm-to-cnm`'s heuristics** (M12) when the CINM 1.0
  baseline is retired or the GPU backend moves to
  `--convert-linalg-to-cnm`. Not doing it now: that pass is the CINM 1.0
  baseline's lowering and the only cinm→cnm route for GPU.
- **`getDesignParams()`/`instantiateDesignParams()`** are stubbed and
  unexplained. Do they eventually replace `SpaceBuilder` + attribute
  stamping, or serve a different (declarative) class of parameters?
- **Merge or keep `UpmemGenericLoweringNotes.md`** (§E) — is the direct
  `linalg.generic → upmem` path a permanent parallel option, or
  superseded by this one?

## 4. Where things stand

`gemv_64MB` on real hardware, same configuration through both lowerings:

| | net | scatter | gather | launch | unaccounted |
|---|---|---|---|---|---|
| `atim_templateflow` | 10.782 | 9.428 | 0.555 | 0.434 | 0.365 |
| `atim_genericflow` | 103.433 | 24.557 | 0.609 | 0.390 | 77.877 |

All in ms. **Device time already matches** (launch 0.390 vs 0.434) — the
DPU program the generic path emits is as good as the template's. The
whole gap is host-side data movement: the 77.9 ms repack and the missing
broadcast, both listed above.

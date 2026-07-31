# Implementation plan: CNM memory-level awareness (§A)

This is the concrete, code-level plan for [CnmMemoryLevelsDesign.md](CnmMemoryLevelsDesign.md)
§A. It assumes that document's decisions as given and does not re-litigate
them. Two things this pass over the design surfaced that aren't in that
document yet:

1. A **consistency review** (§0 below) — gaps between §A's decisions and
   the current code, found by reading `UpmemInferAccelerator.cpp`,
   `CinmToCnm.cpp`, `CinmTilingImplementations.cpp`, and the relevant
   `Passes.td` files in full rather than the excerpts the design doc
   quotes. Two of these are real blockers, not polish.
2. A **milestone breakdown** (§1 onward) — one shippable unit per
   milestone, each ending in working, tested code, in dependency order.
   Every new pass and every new pass option gets a test file specified
   as part of its milestone, per existing lit conventions (`cinm-opt`
   + `FileCheck`, `--split-input-file` where a file holds multiple
   cases).

Reassuring finding first: `CinmToCnm.cpp` already has a comment block
sketching almost exactly this design
([CinmToCnm.cpp:376-411](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L376-L411)) —
down to `cnm.buffer<128xi32 on 8, "wram">` syntax and a `cnm.launch`
whose block arguments are MRAM buffers. This plan is the realization of
that TODO, not a new direction.

## 0. Consistency review

### 0.1 `cinm.op.reduce` has no memref/DPS mode — blocks §A3 for `handleReduce`

`Cinm_GemmlikeOp` (backing `gemv`/`gemm`/`batch_gemv`/`batch_gemm`) already
has a dual tensor/memref mode via `GemmlikeOpInterface::isTensorVariant()`
([CinmGemmlikeOpInterface.td](../include/cinm-mlir/Dialect/Cinm/IR/CinmGemmlikeOpInterface.td),
[CinmOps.td:104-109](../include/cinm-mlir/Dialect/Cinm/IR/CinmOps.td#L104-L109)):
`lhs`/`rhs`/`bias`/`out` are all `AnyShaped`, and in memref mode the op
accumulates into `out` in place with no result. `GemmLikeTilingModel`
already branches on `isa<MemRefType>(out.getType())`
([CinmTilingImplementations.cpp:178-183](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L178-L183)).
This is exactly what §A3 needs for a smaller `cinm.op.gemv` to live inside
a `cnm.launch` body operating on memrefs — **no changes needed for
gemv/gemm/batch_gemv/batch_gemm.**

`Cinm_ReduceOp` does not have this. It's declared `Pure`
([CinmOps.td:191-192](../include/cinm-mlir/Dialect/Cinm/IR/CinmOps.td#L191-L192)),
has only `input: AnyShaped` and a tensor/scalar `result` — no `out`
operand. `ReduceTilingModel::convertToTiledOps` only builds a
`tensor::EmptyOp` or scalar `arith::ConstantOp` accumulator
([CinmTilingImplementations.cpp:293-302](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L293-L302)) —
there is no path that produces a memref-mode reduce. So `handleReduce`'s
launch body cannot contain a smaller `cinm.op.reduce` today; §A3 as
designed does not yet apply to reduce.

**This needs its own sub-step before milestone M3** (below): give
`Cinm_ReduceOp` a memref/DPS mode mirroring `GemmlikeOpInterface` —
add an optional `out: AnyShaped` operand, drop (or conditionally relax)
`Pure`, add `isTensorVariant()`, and extend `ReduceTilingModel` to
accumulate into `out` when present, the same way `GemmLikeTilingModel`
does. Scope note: this only needs to support the last-dimension
reduction case `handleReduce` already restricts itself to
([UpmemInferAccelerator.cpp:497-500](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L497-L500)).

### 0.2 Tiled-op attribute propagation: the WRAM tile-size handoff has no carrier

This is the sharpest gap and worth spelling out precisely, because it's
easy to design around without noticing it's broken.

`SpaceBuilder`-driven tile sizes reach a pass today via a two-attribute
relay: `handleGemv` stamps `kTileParamNamesAttr` (`upmem.tile_param_names`,
an array of SpaceVar *names*) on the **original, untiled** op
([UpmemInferAccelerator.cpp:420-423](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L420-L423)),
and later `applyTileSizes()` reads those names, looks up their *values*
for the current trial configuration, and writes `cinm.tile_sizes`
(`CinmDialect::TILING_FACTORS_NAME`) — which `--cinm-tiling` then
consumes via `CinmTilingInterface::convertToTiledOps`
([UpmemInferAccelerator.cpp:372-388](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L372-L388)).

§A3's design runs this same tiling mechanism **twice**: once on the
original op (host tile → MRAM tile), once again inside the `cnm.launch`
body, on a **different op instance**, (MRAM tile → WRAM tile). But
`convertToTiledOps` implementations build their inner op with a plain
builder call and copy no attributes from the original — e.g.
`GemmLikeTilingModel`'s `buildTileOp` is just
`Op::create(b, loc, lhs, rhs, acc, out).getResult()`
([CinmTilingImplementations.cpp:485-488](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L485-L488)).
So any attribute placed on the **outer** op before the first
`--cinm-tiling` run (including a hypothetical
`upmem.tile_param_names` naming the WRAM SpaceVars) is simply gone by
the time the inner op exists — there is nothing on the inner op for a
second `applyTileSizes()`-style pass to read.

Two ways to fix this, evaluated:

- **(Recommended) Teach the tiling mechanism to propagate a fixed
  attribute allowlist.** Add an allowlist parameter (or a fixed,
  documented set of attribute name prefixes, e.g. everything under
  `upmem.`) to `CinmTilingPass`/`CinmApplyTilingInterfacePattern`
  ([TilingPass.cpp:26-51](../lib/Dialect/Cinm/Transforms/TilingPass.cpp#L26-L51)),
  copied from the original op onto every op `convertToTiledOps`
  produces of the same op type (i.e. onto the new inner `cinm.op.gemv`,
  not onto loop ops). `handleGemv` then stamps a *second*
  tile-param-names attribute for the WRAM SpaceVars (analogous to the
  existing MRAM one) directly on the original op; it rides along
  through the first tiling pass for free and is present on the inner
  op for the second tiling pass to consume via the same
  `applyTileSizes()` logic. Minimal new surface, reuses the existing
  relay idiom exactly.
- **(Rejected for v1)** Have the cleanup pass recompute WRAM tile sizes
  from scratch (e.g. via `computeTilingFactors`) instead of relying on
  a propagated, DSE-searched value. This throws away the ability to
  search over WRAM tile size independently of MRAM tile size — a real
  capability regression relative to today's `handleGemv`, which
  searches `wramRow`/`wramCol` and `mramRow`/`mramCol` as independent
  `SpaceVar`s. Only acceptable as a stopgap if the propagation fix is
  deferred.

This is a prerequisite for milestone M5 (the cleanup pass) and needs to
land in M2 alongside the `--cinm-tiling` changes, since it's a change
to `--cinm-tiling` itself, not to any MRAM-specific pass.

### 0.3 `evaluate()`'s real-pipeline path is dead code — needs an explicit decision, not a silent assumption

`UpmemInferencePlugin::evaluate()` **always** takes the "bypass" branch:
it calls each op's `registerSimulator` lambda (which invokes
`generateGemv`/`generateTailReduction` directly). The entire
`buildPipeline()` / `runPipeline()` / `applyTileSizes()` path — which
already exists, already runs `--cinm-tiling` → `--convert-cinm-to-cnm`
→ bufferize → `--convert-cnm-to-upmem`, i.e. exactly the pipeline this
whole effort is about — is commented out
([UpmemInferAccelerator.cpp:343-368](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L343-L368)).
`buildPipeline()` is otherwise unused dead code today.

The design doc doesn't say what happens to the *search-time* cost
evaluation once §A ships — it only talks about `cnm.local_transfer`
lowering and the cost model running "at the UPMEM-dialect level" in
the abstract. Concretely there are two different things that could mean
"run the real pipeline":

- **(a) Final codegen only.** DSE keeps using the fast per-op template
  simulators for search (thousands of evaluations; the templates exist
  *because* running the full pipeline per candidate is presumably too
  slow), and the real pipeline is invoked exactly once, after search
  picks a winner, to materialize the actual output program. This
  matches §F's "keep the template path alive as an oracle" framing.
- **(b) Search-time evaluation too.** `evaluate()` switches to always
  running `buildPipeline()` and simulating its output, replacing the
  template bypass entirely. More accurate (no more risk of the
  template's hand-derived cost diverging from what real codegen
  produces), but only viable if the real pipeline is fast enough to run
  O(maxEvals) times per op — unverified, and `buildPipeline()` includes
  heavyweight passes (one-shot bufferize, full affine optimization,
  `createUPMEMDedupKernelsPass`, loop unrolling) that were presumably
  never tuned for being on this hot path.

**Recommendation: (a) for the milestones in this plan.** Wire the new
passes into `buildPipeline()` (M6), add a flag to opt into running it
(default off, so `evaluate()`'s behavior — and therefore existing BO
results/tests — doesn't change), and treat "switch DSE's inner loop to
the real pipeline" as a separate, later decision once (a) has validated
that the real pipeline's output is correct and roughly cost-equivalent
to the templates'. This needs sign-off since it affects search runtime
and isn't purely mechanical — flagging rather than assuming.

### 0.4 `useMRAMTiling` is already spoken for — don't reuse it as the new pipeline's on/off switch

`opts.useMRAMTiling` (`--upmem-infer-accelerator=use-mram-tiling=...`)
already has a meaning: inside `handleGemv`/`handleReduce`'s constraint
building, it toggles whether `mramRow`/`mramCol` are forced equal to
`wramRow * tasklets/taskletCols` (i.e. CINM-1.0-style, single-level
tiling) or free to differ (real two-level tiling)
([UpmemInferAccelerator.cpp:427-440](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L427-L440)).
That's a statement about the **shape of the search space**, orthogonal
to *which code path realizes a given configuration* (hand-written
template vs. this effort's generic pipeline). Conflating them would
mean "generic pipeline" and "single-level tiling" become impossible to
select independently, which is a real combination someone will want
(e.g. for validating the generic pipeline against the CINM-1.0-shaped
template before trusting it with two-level tiling). Introduce a
separate option, e.g. `use-generic-pipeline` (default `false` per §0.3),
for M6's pipeline switch.

## 1. Dependency graph

```
M0  CinmLevelAttrInterface + #upmem.mram/#upmem.wram         (§A1)
 |
M1  cnm-buffer-level pass option on --convert-cinm-to-cnm    (§A2)
 |
M2  cnm.local_transfer op  +  --cinm-tiling attribute         (§A4, §0.2)
 |   propagation fix
 |
M3  cinm.op.reduce memref/DPS mode                            (§0.1)
 |
M4  --convert-cinm-to-cnm emits inner cinm ops inside          (§A3, stage 1)
 |   cnm.launch instead of lowering straight to linalg
 |
M5  upmem-tile-mram-buffers cleanup pass (2nd --cinm-tiling    (§A3, stage 2)
 |   round + private-WRAM alloc + cnm.local_transfer insertion)
 |
M6  --convert-cnm-to-upmem: MRAM block-arg lowering (static     (§A3 backend,
 |   alloc + subview, broadcast detection) + cnm.local_transfer  §A4 backend)
 |   -> upmem.local_transfer
 |
M7  UpmemInferAccelerator plugin wiring (buildPipeline update,  (§D, §0.2-0.4)
     WRAM tile-size stamping, use-generic-pipeline flag)
```

M0-M1 and M2-M3 are independent of each other and could be parallelized;
everything from M4 onward depends on all four.

## 2. Milestones

### M0 — `CinmLevelAttrInterface` + `#upmem.mram`/`#upmem.wram`

**Goal:** implement §A1's decision.

**Changes:**
- New `include/cinm-mlir/Dialect/Cinm/IR/CinmLevelAttrInterface.td`
  (or fold into `CinmPlatformAttrInterface.td`, matching where
  `CinmPlatformAttrInterface`/`CinmAcceleratorAttrInterface` already
  live): `AttrInterface<"CinmLevelAttrInterface">` with `getName()`,
  `getSizeInBytes()`, `getAlignment()`, and a `getMemrefMemspace()`
  default-implemented as returning `getName()` as a `StringAttr`
  (moves `UpmemPlatformAttr::getMemrefMemspace`'s current one-line body
  onto the level itself, per §A1's follow-up note). No `arity` method —
  per the design decision, that's dropped from the generic surface.
  Keep `getSizeInElements(Type)` as an `extraClassDeclaration` helper
  (pure function of `getSizeInBytes()`, no need for it to be virtual).
- `CinmLevelDefAttr` gains
  `DeclareAttrInterfaceMethods<CinmLevelAttrInterface>` in its trait
  list ([CinmAttributesBase.td:70](../include/cinm-mlir/Dialect/Cinm/IR/CinmAttributesBase.td#L70)).
  Leave its `arity` parameter in place (dead, but removing a
  tablegen-generated attribute parameter ripples into its
  `struct(params)` assembly format and every existing user for no
  functional gain — out of scope for this effort; note as optional
  follow-up cleanup).
- `CinmPlatformAttrInterface::getLevels()`/`getLevel(name)`
  ([CinmPlatformAttrInterface.td:27-44](../include/cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td#L27-L44))
  change return type from `ArrayRef<CinmLevelDefAttr>`/`CinmLevelDefAttr`
  to `ArrayRef<CinmLevelAttrInterface>`(-compatible, e.g.
  `SmallVector<Attribute>` cast at use sites) /`CinmLevelAttrInterface`.
- `CnmAcceleratorAttrInterface::getWorkgroupMemoryLevels()`
  ([CnmInterfaces.td:27-33](../include/cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.td#L27-L33))
  and its concrete `CinmLevelArrayAttr` return type: replace with an
  `ArrayRef<Attribute>`-based signature (or a small hand-written array
  attr over the interface) — `CinmLevelArrayAttr`'s `ArrayOfAttr`
  tablegen helper can't parameterize over an interface.
- New UPMEM attrs, in `UPMEMTileFirstAttributes.td` next to
  `UpmemPlatformAttr`: `UpmemMramAttr` (mnemonic `mram`) and
  `UpmemWramAttr` (mnemonic `wram`), each
  `DeclareAttrInterfaceMethods<CinmLevelAttrInterface>`, parameters
  `(ins "int64_t":$sizeInBytes, "int64_t":$alignment)` (mirrors what
  `upmemLevels()` already computes per `isV1a`
  ([UPMEMAttributes.cpp:145-163](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L145-L163))).
  `UpmemPlatformAttr::getMramLevel()`/`getWramLevel()`
  ([UPMEMTileFirstAttributes.td:40-46](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMTileFirstAttributes.td#L40-L46))
  now return these instead of indexing into a `CinmLevelDefAttr` array;
  `upmemLevels()` constructs `UpmemMramAttr`/`UpmemWramAttr` instead of
  `CinmLevelDefAttr`.
- Ripple: every call site currently assuming `CinmLevelDefAttr`
  specifically (`UpmemAcceleratorAttr::getMramLevel()`/`getWramLevel()`,
  `handleGemv`/`handleReduce`'s `platform.getWramLevel()`/
  `getMramLevel()` calls) keeps compiling unchanged as long as they
  only use interface methods (`getSizeInElements`, etc.) — verify none
  of them pattern-match on the concrete `CinmLevelDefAttr` type
  directly (a quick `grep -n "CinmLevelDefAttr"` after the change
  should only show the definition site and `CinmLevelArrayAttr`'s
  removal).

**Tests:**
- `test/Dialect/UPMEM/upmem-level-attrs.mlir` (new): round-trip parse/print
  of `#upmem.mram<...>` and `#upmem.wram<...>` standalone (`cinm-opt %s |
  cinm-opt | FileCheck %s`, mirroring `test/Dialect/Cnm/cnm-ops.mlir`'s
  double round-trip style), plus a case using them inside
  `#upmem.platform<..., levels = [...]>`'s optional `levels` parameter
  ([UPMEMAttributes.cpp:184-191](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L184-L191)
  already supports an explicit `levels=` override — extend that case to
  use the new attrs instead of `#cinm.level<...>`).
- `test/Dialect/UPMEM/upmem-ops.mlir`: add a regression case confirming
  `#upmem.platform<type=v1A, dimensions=...>` (no explicit `levels=`)
  still round-trips to the *same* printed form as before (default levels
  now built from `UpmemMramAttr`/`UpmemWramAttr`, printed form should be
  unaffected since `getDefault()`'s levels aren't printed unless they
  differ — verifies the migration didn't change default printing).

### M1 — `cnm-buffer-level` pass option

**Goal:** implement §A2. `--convert-cinm-to-cnm` (`ConvertTiledCinmToCnm`)
gets a level knob; default preserves today's behavior exactly.

**Changes:**
- `CnmPasses.td`... actually `CinmPasses.td`
  ([CinmPasses.td:10-15](../include/cinm-mlir/Conversion/CinmPasses.td#L10-L15)):
  add `Option<"bufferLevel", "cnm-buffer-level", "std::string", "\"\"",
  "Name of the platform memory level (e.g. \"mram\") that newly
  allocated cnm.buffers should target. Empty = unset level (today's
  behavior).">`. String-keyed against `platform.getLevel(name)` rather
  than a fixed enum, matching the design's backend-agnostic framing —
  the pass doesn't need to know level names in advance.
- `CinmToCnm.h`: add
  `std::unique_ptr<Pass> createConvertTiledCinmToCnmPass(ConvertTiledCinmToCnmOptions)`
  overload, mirroring `CnmToUPMEM.h`'s
  `createConvertCnmToUPMEMPass(ConvertCnmToUPMEMPassOptions)` pattern
  ([CnmToUPMEM.h](../include/cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h)).
- `CinmToCnm.cpp`: thread the resolved level attribute (via
  `workgroup.getType().getAccelerator().getPlatform().getLevel(bufferLevel)`,
  empty string → null attribute, unresolved name → pass failure with a
  diagnostic) into both `cnm::BufferType::get(...)` call sites,
  replacing the hardcoded `nullptr`/`0` (`convertInputIntoAlloc`,
  [CinmToCnm.cpp:309-311](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L309-L311),
  and the two calls around
  [CinmToCnm.cpp:886-893](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L886-L893)).
  This also directly resolves the long-standing `// todo level is
  hardcoded` comment at line 311.
- No changes needed to `cnm.launch`'s verifier or `createLaunchOp`'s
  block-argument typing
  ([CinmToCnm.cpp:324-357](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L324-L357)) —
  it already does `MemRefType::get(inputTy.getShape(),
  inputTy.getElementType())` with no memory space; this needs to pick up
  `inputTy.getLevel()` as the memory space, matching what the verifier
  already independently requires. This is a one-line fix
  (`MemRefType::get(shape, elt, /*layout*/{}, inputTy.getLevel())`)
  needed regardless of `bufferLevel`'s value, since it's currently
  inconsistent with the verifier whenever `level` is non-null.

**Tests:**
- `test/Conversion/CinmToCnm/cinm-to-cnm-mram-buffers.mlir` (new):
  `// RUN: cinm-opt --convert-cinm-to-cnm=cnm-buffer-level=mram
  --canonicalize %s | FileCheck %s` on a small `cinm.op.gemv`, checking
  `!cnm.buffer<... , #upmem.mram>` on the allocated buffers and
  `cnm.launch ... { ^bb0(%arg: memref<..., #upmem.mram>, ...)` on the
  block argument types.
- Same file, second case (`--split-input-file`) with an unrecognized
  level name (`cnm-buffer-level=nonexistent`) checking the pass reports
  a clear error (`// CHECK: error: ...`, using
  `-verify-diagnostics` per existing error-path test conventions if any
  exist in this repo — check `test/Dialect/UPMEM/verifier.mlir`'s style
  and match it).
- Regression: confirm `test/Conversion/CinmToCnm/cinm-to-cnm.mlir` and
  `test/Conversion/CinmToCnm/cinm-to-cnm-difficult.mlir` pass unmodified
  (no `cnm-buffer-level` flag passed → empty string → null level → byte
  identical to today, since `CnmTypes.cpp:88-89` only prints `level`
  when non-null). This is a required check, not a new file.

### M2 — `cnm.local_transfer` + `--cinm-tiling` attribute propagation

**Goal:** implement §A4, and fix §0.2.

**Changes — `cnm.local_transfer`:**
- Add to `CnmOps.td`, mirroring `upmem.local_transfer`
  ([UPMEMOps.td:190-210](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.td#L190-L210))
  exactly: `arguments = (ins Arg<AnyMemRef, ..., [MemReadAt<0,
  FullEffect>]>:$source, Arg<AnyMemRef, ..., [MemWriteAt<0,
  FullEffect>]>:$target)`, no results, a verifier requiring matching
  shape/element type (direction — which side is MRAM/WRAM — is implicit
  in the operand types' `level`, not a separate attribute, consistent
  with "memrefs already carry their level as memory space"). Assembly
  format modeled 1:1 on `upmem.local_transfer`'s.
- `CnmDialect` dependent-dialect list already includes `memref` (used by
  `EnsureScatterGatherContiguousPass`); no new dependency needed.

**Changes — attribute propagation (§0.2 fix):**
- `TilingPass.cpp`'s `CinmApplyTilingInterfacePattern`
  ([TilingPass.cpp:26-51](../lib/Dialect/Cinm/Transforms/TilingPass.cpp#L26-L51)):
  after `op.convertToTiledOps(...)` succeeds, walk the newly produced
  `results`/the region the op was replaced in for freshly created ops of
  the *same op name* as `op`, and copy over attributes from `op` whose
  name matches an allowlist. Concretely: a fixed prefix list
  (`"upmem."`, extensible) checked via `StringRef::starts_with`, since
  this needs to stay generic (not UPMEM-specific) even though today's
  only user is UPMEM. Alternative considered: make the allowlist a pass
  option (`ListOption<"propagatedAttrPrefixes", ...>`) instead of a
  hardcoded constant — slightly more flexible, low cost, recommended if
  time allows since it keeps `--cinm-tiling` fully backend-agnostic
  rather than hardcoding a `upmem.` prefix into cinm-dialect code.
- `handleGemv`: add a second attribute analogous to
  `kTileParamNamesAttr` for the WRAM SpaceVars — reuses the same
  key/mechanism, just needs a distinct attribute name (e.g.
  `upmem.wram_tile_param_names`) so both survive independently and
  `applyTileSizes()` (or its M7 successor) can find the right one at
  each of the two tiling rounds.

**Tests:**
- `test/Dialect/Cnm/cnm-ops.mlir`: add `cnm.local_transfer` round-trip
  cases to the existing double-parse test (`memref<...,#upmem.mram>` to
  `memref<...,#upmem.wram>` and the reverse), plus a verifier-failure
  case (shape mismatch) in `test/Dialect/UPMEM/verifier.mlir`-style
  (or a new `test/Dialect/Cnm/verifier.mlir` if one doesn't exist yet —
  check first).
- `test/Dialect/Cinm/cinm-tiling-attr-propagation.mlir` (new): a
  `cinm.op.gemv` (memref mode, from M3 — or gemv directly, since gemv
  already supports memref mode, no need to wait for M3 here) carrying
  both `cinm.tile_sizes` and a dummy `upmem.test_marker` attribute, run
  through `--cinm-tiling`, checking the marker attribute survives onto
  the inner `cinm.op.gemv` inside the generated loop nest but *not* onto
  the `affine.for` ops themselves.

### M3 — `cinm.op.reduce` memref/DPS mode

**Goal:** fix §0.1, the reduce-specific prerequisite for §A3.

**Changes:**
- `CinmOps.td`: add `Optional<AnyShaped>:$out` to `Cinm_ReduceOp`'s
  arguments, relax `Pure` (drop it, or make it conditional the way
  `Cinm_GemmlikeOp` uses `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>`
  instead of `Pure` — match that pattern for consistency with the other
  DPS-capable cinm ops), add `isTensorVariant()` /
  `CinmGemmlikeOpInterface`-style accessors, or a small bespoke
  interface if reusing `CinmGemmlikeOpInterface` doesn't fit reduce's
  shape (it has no `rhs`/`bias`).
- `CinmTilingImplementations.cpp`'s `ReduceTilingModel::convertToTiledOps`
  ([CinmTilingImplementations.cpp:270-354](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L270-L354)):
  when `out` is present and a memref, accumulate into slices of it
  instead of building a `tensor::EmptyOp`/`arith::ConstantOp`
  accumulator — mirrors `GemmLikeTilingModel`'s `outBuf` branch
  ([CinmTilingImplementations.cpp:177-183](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L177-L183)).
- `--convert-cinm-to-cnm`'s reduce-handling path (wherever it currently
  builds the linalg reduction inside `cnm.launch` — same place M4
  touches for gemv) needs the equivalent branch for reduce once M4 is
  in place; listed here for completeness but implemented as part of M4.

**Tests:**
- `test/Dialect/Cinm/cinm-ops.mlir`: round-trip parse/print of the new
  memref-mode `cinm.op.reduce ... into %out : ...` syntax.
- `test/Dialect/Cinm/cinm-reduce-memref-tiling.mlir` (new): `--cinm-tiling`
  on a memref-mode `cinm.op.reduce`, checking the generated loop nest
  accumulates into slices of `%out` rather than building a fresh tensor
  accumulator (structurally analogous to what a `--cinm-tiling` test for
  memref-mode gemv would check, were one to exist — flag as a gap that
  M2 or this milestone should also add a basic
  `test/Dialect/Cinm/cinm-gemv-memref-tiling.mlir` if the existing test
  suite has no coverage of memref-mode `--cinm-tiling` today; confirm by
  checking `test/Dialect/Cinm/` — currently only `cinm-ops.mlir`,
  `cinm-parse-memrefs.mlir`, `cinm-verifier.mlir` exist, none of which
  are `--cinm-tiling` pass tests, so this is a genuine pre-existing
  coverage gap, not just new-feature coverage).

### M4 — inner `cinm` ops inside `cnm.launch` bodies

**Goal:** implement §A3's first bullet list item: when `bufferLevel` is
set to a non-leaf level, `--convert-cinm-to-cnm` emits a smaller `cinm`
op referencing the level-tagged block arguments, instead of lowering
straight to `linalg.contract`/whatever it emits today.

**Changes:**
- Locate and branch the launch-body-construction code path (the
  `createCnmLaunchBlock` callback passed into `createLaunchOp`
  /`convertCinmToCnm`
  ([CinmToCnm.cpp:324-357](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L324-L357)),
  and the call sites building `linalg::ContractOp`
  ([CinmToCnm.cpp:930](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L930),
  [:1036](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L1036))): when
  `bufferLevel` was set (buffers are non-leaf), build a smaller
  `cinm.op.gemv`/`cinm.op.gemm`/`cinm.op.reduce` (memref mode, per
  M0-M3) over the launch block's memref arguments instead of the linalg
  op. When `bufferLevel` is unset (today's default path), keep emitting
  linalg exactly as today — **this is a genuinely new code path, not a
  modification of the existing one**, gated on the same option added in
  M1.
- The new inner op needs `cinm.tile_sizes`/the WRAM-tile-param-names
  attribute (M2) already present for M5 to pick up — but at this stage
  (`--convert-cinm-to-cnm` runtime) that attribute must have already
  been propagated by the *first* `--cinm-tiling` run (M2's fix) from the
  original pre-tiling op onto whatever `--convert-cinm-to-cnm` is now
  consuming. Verify the pipeline ordering in `buildPipeline()`
  (tiling → convert-cinm-to-cnm, already the existing order) makes this
  hold; this milestone shouldn't need pipeline reordering, just
  confirmation.

**Tests:**
- Extend `test/Conversion/CinmToCnm/cinm-to-cnm-mram-buffers.mlir`
  (from M1) with a case checking the launch body contains
  `cinm.op.gemv %arg0, %arg1 into %arg2 : memref<...>, memref<...> into
  memref<..., #upmem.mram>` rather than `linalg.contract`.
- A parallel case for `cinm.op.reduce` in the same file (depends on M3).
- Regression: the *default* (`cnm-buffer-level` unset) path in the same
  file still produces `linalg.contract`, unchanged from
  `cinm-to-cnm.mlir`'s existing checks.

### M5 — `upmem-tile-mram-buffers` cleanup pass

**Goal:** implement §A3's core: the UPMEM-provided pass that runs
`--cinm-tiling` a second time inside `cnm.launch` bodies and rewrites
leaf-tiled ops onto new private-WRAM buffers with `cnm.local_transfer`
around them.

**Changes:**
- New pass, registered in
  `include/cinm-mlir/Dialect/UPMEM/Transforms/Passes.td`
  (alongside `UPMEMDedupKernelsPass` etc.
  ([Passes.td:12-19](../include/cinm-mlir/Dialect/UPMEM/Transforms/Passes.td#L12-L19))):
  `UpmemTileMRAMBuffersPass`, pass name `upmem-tile-mram-buffers`, no
  options needed for v1 (tile sizes come from attributes already on the
  IR, per M2/M4). New file
  `lib/Dialect/UPMEM/Transforms/UpmemTileMRAMBuffers.cpp`.
- Implementation, two phases:
  1. For every `cnm::LaunchOp` in the module, run `--cinm-tiling`
     nested under it (`OpPassManager` scoped via `.nest<cnm::LaunchOp>()`
     — legal because `LaunchOp` is `IsolatedFromAbove`
     ([CnmOps.td:126](../include/cinm-mlir/Dialect/Cnm/IR/CnmOps.td#L126)));
     the inner `cinm` op's WRAM tile-size attribute (from M2/M4) drives
     this round exactly like the outer round does today.
  2. Walk the resulting IR for leaf-tiled `cinm` ops whose operand
     memrefs carry a non-leaf level (i.e. still `#upmem.mram`, since
     tiling only changed loop bounds/subview offsets, not the memory
     level — per §A3). For each such operand: allocate a private
     `memref.alloc` sized to the tile with memory space
     `#upmem.wram`, insert `cnm.local_transfer` from a
     `memref.subview` of the MRAM operand into it (inputs) or from it
     into a subview (outputs), and rewrite the op to use the new WRAM
     buffer. "Leaf-tiled" = the op's tile size now equals the leaf
     entry of `getWorkgroupMemoryLevels()`
     ([CnmInterfaces.td:27-33](../include/cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.td#L27-L33))
     — i.e. no `cinm.tile_sizes` attribute remains that would trigger
     another `--cinm-tiling` pass, which is the natural post-condition
     of phase 1 rather than something this pass needs to re-derive.
- Explicitly **not** in scope for M5 (§A3 "deferred optimization" +
  §C): broadcast-across-tasklets sharing of the new WRAM buffer via a
  static alloc + barrier. Phase 2 always allocates private WRAM.

**Tests:**
- `test/Transform/UPMEM/upmem-tile-mram-buffers.mlir` (new, matching
  where other post-pipeline UPMEM transform tests live —
  `test/Transform/UPMEM/simulate-python.mlir`): input is a `cnm.launch`
  containing an MRAM-level `cinm.op.gemv` already carrying
  `cinm.tile_sizes` (hand-written, standing in for what M2/M4 would
  produce) — output checks: nested `affine.for` loop, `memref.alloc`
  with `#upmem.wram` memory space at tile size, `cnm.local_transfer`
  before and after, smaller `cinm.op.gemv` on the WRAM buffers.
- A second case with an already-leaf-sized MRAM op (no `cinm.tile_sizes`)
  checking the pass still inserts the alloc+transfer+op+transfer
  rewrite even without a second tiling round needed (MRAM tile ==
  WRAM tile is a legal degenerate case, corresponds to
  `useMRAMTiling=false`'s "cinm 1.0 mode" per
  [UpmemInferAccelerator.cpp:427-440](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L427-L440)).

### M6 — `--convert-cnm-to-upmem` backend lowering

**Goal:** implement §A3's backend-realization bullets and §A4's backend
realization: MRAM block arguments lower to a subview of a single static
allocation (or the allocation itself, if broadcast), and
`cnm.local_transfer` lowers to `upmem.local_transfer`.

**Changes, in `CnmToUPMEM.cpp`:**
- Generalize the existing `mramHasTaskletDim` subview logic in
  `createTransfer`
  ([CnmToUPMEM.cpp:335-367](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L335-L367))
  so it applies to *every* use of an MRAM-level `cnm.launch` block
  argument that this pass rewrites, not just the transfers it already
  inserts today. Since `cnm.alloc` already unconditionally gets an
  `upmem::StaticAllocOp`
  ([CnmToUPMEM.cpp:522-535](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L522-L535)),
  the main change is making sure *all* remaining uses inside the launch
  body (not just the pass's own inserted transfers) get the subview
  treatment — after M4/M5, uses inside the body are exactly the
  `cnm.local_transfer` ops M5 inserted (M5 already builds them as
  memref-to-memref against subviews of the *logical* MRAM memref;
  this milestone's job is making sure the memref that ends up there,
  post-conversion, really is a subview of the single static alloc).
  `isMramBroadcastOverThreads`
  ([CnmToUPMEM.cpp:424](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L424))
  is reused as-is for the shared-vs-subview decision, per §A3.
- Add a conversion pattern for `cnm.local_transfer` →
  `upmem.local_transfer`, near-1:1 (both take `source`/`target` memrefs,
  no shape transformation needed — the operands are already
  correctly-shaped memrefs by construction from M5).
- Verify `--cnm-ensure-scatter-gather-contiguous`
  ([Passes.td:65-78](../include/cinm-mlir/Dialect/Cnm/Transforms/Passes.td#L65-L78))
  doesn't need to run on `cnm.local_transfer` too — it currently only
  targets `cnm.scatter`/`cnm.gather` (host↔device transfers);
  `cnm.local_transfer`'s operands are subviews of on-device buffers
  M5 already constructed to be the right shape, so contiguity should
  be a non-issue, but worth an explicit check/comment rather than an
  assumption.

**Tests:**
- `test/Conversion/CnmToUpmem/cnm-to-upmem-mram-level.mlir` (new,
  following `cnm-to-upmem.mlir`'s RUN-line style): feed IR with an
  MRAM-level `cnm.alloc`/`cnm.launch` (block arg `memref<...,
  #upmem.mram>`) containing `cnm.local_transfer` ops (hand-written,
  standing in for M5's output), checking: `upmem.static_alloc` with
  `DpuMemSpace::MRAM`, `memref.subview` of it inside the launch body,
  `upmem.local_transfer` where `cnm.local_transfer` was.
- A second case with a broadcast scatter map (tasklet-independent),
  checking no subview is inserted — the whole static alloc is used
  directly, matching `isMramBroadcastOverThreads`'s existing
  broadcast path.

### M7 — `UpmemInferAccelerator` plugin wiring

**Goal:** implement §D concretely, resolving §0.2-0.4.

**Changes:**
- `buildPipeline()`
  ([UpmemInferAccelerator.cpp:161-249](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L161-L249)):
  insert `--convert-cinm-to-cnm=cnm-buffer-level=mram` (replacing the
  option-less call at Step 2) and `--upmem-tile-mram-buffers`
  (M5) right after it, before `cnm::createCnmHoistWorkgroupsPass()` —
  matching §A3's pipeline sketch
  (`--convert-cinm-to-cnm=cnm-buffer-level=mram` → cleanup pass →
  further CNM optimizations → `--convert-cnm-to-upmem`).
- `handleGemv`/`handleReduce`: uncomment and fix the
  `kTileParamNamesAttr` stamping for `mramRow`/`mramCol` too (today only
  `wramRow`/`wramCol` are stamped, with a comment saying it's "used by
  applyTileSizes() in the non-MRAM pipeline path" — that comment is
  stale/backwards now that MRAM is the primary path; both tile-size
  pairs need stamping). Fix the `// fixme here` commented-out stamping
  in `handleReduce`
  ([UpmemInferAccelerator.cpp:524-527](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L524-L527)) —
  currently `handleReduce` stamps nothing at all, presumably because
  reduce never had a working non-bypass path; now it needs both MRAM-
  and WRAM-tile attributes like gemv (depends on M3 existing so there's
  a memref-mode reduce for it to apply to).
- Add `use-generic-pipeline` option (§0.4) to `UpmemInferAcceleratorPass`
  (`UPMEMTransformsPasses.td`), default `false`. In `evaluate()`, when
  true, call `runPipeline(pipeline.get(), ...)` (resurrecting the
  commented-out block at
  [UpmemInferAccelerator.cpp:356-368](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L356-L368))
  and simulate its output, instead of the `registerSimulator` bypass —
  per §0.3(a), this stays off by default; flipping it on is milestone
  M8 (below), not this one.
- `applyTileSizes()` needs to run (at least) twice per trial when the
  generic pipeline is enabled: once before `buildPipeline()`'s tiling
  step for the MRAM round, and the WRAM round's attribute is already
  riding along on the original op per M2's propagation fix, so no
  second explicit `applyTileSizes()` call should be needed — the
  *values* were already resolved into `cinm.tile_sizes`-adjacent form
  before `--cinm-tiling` ever ran the first time. Double check this
  holds once implemented; if the WRAM SpaceVar values can't be resolved
  until dpus/tasklets are known (which they are, at `evaluate()` time,
  before `applyTileSizes()` runs) this should be fine, but call out
  explicitly since it's exactly the kind of interaction §0.2 flagged.

**Tests:**
- Fix `test/Dialect/UPMEM/upmem-infer-accelerator.mlir`: it currently
  has **no `RUN:` line** at all (confirmed — it's dead as a lit test
  today). Give it one, e.g. `// RUN: cinm-opt
  --upmem-infer-accelerator="eval-solution=... use-generic-pipeline=false"
  %s | FileCheck %s` using the existing `eval-solution` option to force
  a single, deterministic configuration (bypasses BO search, per
  [Passes.td:129-131](../include/cinm-mlir/Dialect/UPMEM/Transforms/Passes.td#L129-L131)) —
  this is needed regardless of this effort, flagging as a pre-existing
  gap this work should also close since it's directly relevant.
- `test/Dialect/UPMEM/upmem-infer-accelerator-generic-pipeline.mlir`
  (new): same fixed-solution technique, `use-generic-pipeline=true`,
  small static-shape gemv, checking the annotated/committed output
  module contains real `upmem.dpu_program`/`upmem.local_transfer`
  structure (i.e. the real pipeline actually ran and produced UPMEM IR)
  rather than whatever the bypass path leaves behind.
- End-to-end structural test,
  `test/Transform/UPMEM/gemv-generic-mram-pipeline.mlir` (new): skip the
  inference plugin entirely and manually chain
  `--cinm-tiling{...tile-sizes already annotated by hand...}
  --convert-cinm-to-cnm=cnm-buffer-level=mram --upmem-tile-mram-buffers
  --cnm-ensure-scatter-gather-contiguous --convert-cnm-to-upmem` on a
  small, fully concrete (static-shape) `cinm.op.gemv`, FileCheck'ing the
  final UPMEM IR shape (DPU alloc, MRAM static alloc, WRAM private
  alloc, transfers, kernel body). This is the closest thing to a
  regression test against §F's migration goal ("does the generic
  pipeline produce something structurally equivalent to what
  `generateGemv` produces") that doesn't require BO/simulator
  infrastructure, and is valuable independent of M8.

### M8 (not scheduled — explicitly deferred per §0.3)

Switching DSE's inner-loop cost evaluation to always run the real
pipeline (§0.3 option (b)), and/or retiring the `generateGemv`/
`generateTailReduction` templates and their `SimulationTemplates.cpp`
machinery entirely. Blocked on M0-M7 landing and on validating that the
real pipeline's output cost roughly matches the templates' for at least
gemv (§F's suggested first target), and on a runtime-cost check that
running `buildPipeline()` per BO candidate is actually affordable.
Explicitly out of scope for this plan; listed so it doesn't get lost.

## 3. Test file summary

| Milestone | New/changed pass or option | Test file(s) |
|---|---|---|
| M0 | `#upmem.mram`, `#upmem.wram` attrs | `test/Dialect/UPMEM/upmem-level-attrs.mlir` (new); `test/Dialect/UPMEM/upmem-ops.mlir` (regression case) |
| M1 | `--convert-cinm-to-cnm=cnm-buffer-level=<level>` | `test/Conversion/CinmToCnm/cinm-to-cnm-mram-buffers.mlir` (new, incl. error case); existing `cinm-to-cnm.mlir`/`cinm-to-cnm-difficult.mlir` (regression) |
| M2 | `cnm.local_transfer` op; `--cinm-tiling` attr propagation | `test/Dialect/Cnm/cnm-ops.mlir` (extend); `test/Dialect/Cnm/verifier.mlir` (new or extend); `test/Dialect/Cinm/cinm-tiling-attr-propagation.mlir` (new) |
| M3 | `cinm.op.reduce` memref mode | `test/Dialect/Cinm/cinm-ops.mlir` (extend); `test/Dialect/Cinm/cinm-reduce-memref-tiling.mlir` (new); `test/Dialect/Cinm/cinm-gemv-memref-tiling.mlir` (new, pre-existing coverage gap) |
| M4 | `--convert-cinm-to-cnm` inner-cinm-op launch bodies | `test/Conversion/CinmToCnm/cinm-to-cnm-mram-buffers.mlir` (extend, gemv + reduce cases + default-path regression) |
| M5 | `--upmem-tile-mram-buffers` (new pass) | `test/Transform/UPMEM/upmem-tile-mram-buffers.mlir` (new, 2 cases) |
| M6 | `--convert-cnm-to-upmem` MRAM handling; `cnm.local_transfer` lowering | `test/Conversion/CnmToUpmem/cnm-to-upmem-mram-level.mlir` (new, 2 cases) |
| M7 | `--upmem-infer-accelerator=use-generic-pipeline=<bool>` | `test/Dialect/UPMEM/upmem-infer-accelerator.mlir` (fix — currently has no RUN line); `test/Dialect/UPMEM/upmem-infer-accelerator-generic-pipeline.mlir` (new); `test/Transform/UPMEM/gemv-generic-mram-pipeline.mlir` (new, end-to-end) |

---

<!-- Add your comments below or inline as blockquotes -->

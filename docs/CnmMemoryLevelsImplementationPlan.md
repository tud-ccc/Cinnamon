# Implementation plan: CNM memory-level awareness (§A)

Code-level plan for [CnmMemoryLevelsDesign.md](CnmMemoryLevelsDesign.md) §A.
Everything in §0 below is settled context to keep in mind while
implementing; the per-milestone sections (§3) carry the actual work.

For a one-page map of the whole effort — what changed, what was broken,
and what is deliberately left for later — see
[CnmRefactoringOverview.md](CnmRefactoringOverview.md).

## 0. Standing decisions

**Goal.** `cinm.op.gemv`/`cinm.op.reduce` should reach UPMEM through the
real pipeline (`--cinm-tiling` → `--convert-cinm-to-cnm` → cleanup →
`--convert-cnm-to-upmem`), producing code of the same quality as the
hand-written templates in `SimulationTemplates.cpp`.

1. **Levels are `#upmem.mram` / `#upmem.wram`** — the *existing*
   `upmem::DpuMemSpaceAttr`, which already serves as the memref memory
   space everywhere downstream. It gains a `CinmLevelAttrInterface`
   implementation. No new attributes.
2. **Level identity ≠ level capacity.** The level attribute identifies a
   level and doubles as the memref memory space; it carries no size
   (it can't — see §1.4). Capacity stays platform-side in
   `CinmLevelDefAttr`, reached via `platform.getLevel(name)`.
3. **`--convert-cinm-to-cnm` gets one pass-wide `cnm-buffer-level`
   option.** No per-operand granularity. UPMEM always passes `mram`.
   Default (empty) = today's behavior exactly.
4. **`cnm.launch` bodies receive level-tagged memrefs and contain real
   `cinm` ops.** The launch body is device code; on-device staging can't
   be hoisted outside it. Inner ops implement `CinmTilingInterface`.
5. **WRAM staging = a second `--cinm-tiling` round inside the launch
   body**, then a UPMEM-provided pass that allocates private WRAM and
   wraps the leaf op in `cnm.local_transfer`s. `--cinm-tiling` itself is
   reused unmodified.
6. **`cnm.local_transfer` operates on memrefs**, not `cnm.buffer` —
   tiling is expressed with ordinary `memref.subview`. Its backend
   realization is `--convert-cnm-to-upmem`'s business.
7. **Search parameters live in the plugin, not in IR attributes.**
   `UpmemInferencePlugin` keeps an `Operation* → SpaceVars` map and
   re-stamps freshly resolved values immediately before each pass that
   consumes them. IR attributes are never assumed to survive a pass.
8. **`evaluate()` runs the full lowering** — there is no separate,
   cheaper "evaluation-only" lowering that could drift from real codegen.
9. **The templates stay.** `SimulationTemplates.cpp` and the
   `registerSimulator` bypass are *not* deleted; they are the quality
   bar the generic pipeline must reach. `evaluate()` selects between the
   two paths so they can be compared directly.
10. **No cost/transfer metadata at the CNM level.** The cost model runs
    on UPMEM-dialect IR, after conversion.

### 0.1 Status

**M0–M4 are implemented** (branch `cost-model`). The two design-doc
corrections M0/M3 turned up — §A1 (`CinmLevelDefAttr` is a capacity
descriptor and does *not* implement the level interface) and §F (the
templates stay indefinitely, not just as a development oracle) — have
been folded back into [CnmMemoryLevelsDesign.md](CnmMemoryLevelsDesign.md).

**M0–M11 are implemented and M12 is settled as not-doing.** The generic
path lowers `cinm → linalg → cnm → upmem` end to end, driven entirely by
a search space derived from the linalg op's iteration space, and it runs
on real hardware. Three milestones were added after the original plan and
are outstanding: **M13** (sequential trips), **M14** (exact post-lowering
occupancy check) and **M15** (the host repack), all in §3 below.

Test baseline: **54/61 passing**, 7 failing. All seven are unrelated to
this work and were failing before it — `CimToMemristor` ×2,
`TorchToCinm`, `Transform/Cim` ×2, `Dialect/UPMEM/upmem-to-c.mlir`,
`Transform/UPMEM/simulate-python.mlir`.

The measured state, `gemv_64MB` on hardware, same configuration through
both lowerings: `atim_templateflow` 10.782 ms net, `atim_genericflow`
103.433 ms. **Device time already matches** (launch 0.390 vs 0.434 ms) —
the whole gap is host-side data movement, and 77.9 ms of it is M15.

Known gaps worth carrying forward:
- `--upmem-tile-mram-buffers` promotes with `useFullTileBuffers=false`, so
  a tile that does not divide its extent gets a full-size staging buffer
  and a partial view of it. Correct, but the buffer is larger than the
  tile; observed in the plugin test, where a 16-row WRAM buffer holds a
  2-row tile.
- M7's templates-vs-generic **cost comparison is still unmeasured**. Both
  paths are selectable (`lowering=templates|generic`) on one space, so the
  same configuration can be costed both ways, but the cheap `op-count`
  simulator models neither loads nor stores and does not report time. That
  comparison is the criterion for deleting the templates, and it needs a
  cycle-accurate simulator run.
- The remaining deferred optimizations (broadcast detection, constant
  scatter, transfer coalescing, WRAM sharing, device-side tree reduction)
  are catalogued in
  [CnmRefactoringOverview.md](CnmRefactoringOverview.md) §3 rather than
  here — they are follow-on work, not milestones of this plan.

Two decisions were added along the way that the plan did not anticipate.

**11. The per-leaf buffer budget follows the selected level.** It is
derived from `getWorkgroupMemoryLevels()` (which workgroup dimension owns
the level) and `getWorkgroupShape()` (how many leaves share it), rather
than from `bufferSizeOfLeaf()`. Otherwise `cnm-buffer-level=mram` sizes
MRAM buffers against WRAM and rejects every tile that needs MRAM, which
made the option almost inert. The derivation reproduces
`bufferSizeOfLeaf()` exactly for UPMEM's leaf level, so the no-flag path
is unchanged. See M4.

**12. Launch bodies stay linalg; the level tag alone drives staging.**
This *replaces* decision 4's second half. Selecting a level changes where
buffers live, not what the launch body is, and `--upmem-tile-mram-buffers`
recognizes any `LinalgOp` with buffer semantics whose operands are in a
non-leaf level. Reasons:

- It covers gemm. `ConvertCinmGemmToCnm` gives each leaf exactly one
  output element, so its per-leaf tile is always a dot product and no
  `cinm.op.gemm` could ever stand in its launch body. The same is true of
  gemv whenever the tile is a single row.
- Upstream already does the work: `linalg::promoteSubViews` is built for
  "allocate in faster memory, copy in, compute, copy out", with hooks for
  the memory space, the allocation, and the copy op — the last of which is
  where `cnm.local_transfer` goes.
- It generalizes to any linalg op rather than the two or three cinm ops
  that have a usable launch-body shape.
- `--convert-cinm-to-cnm` goes back to making no decision at all about the
  body, which is the point of the exercise.

The `cinm`-op-in-launch-body branch M4 originally added was reverted.
`CinmTilingInterface` remains what drives the *outer* `--cinm-tiling`
round on the host side; it is only the on-device staging that is linalg's
job. The memref (destination-passing) mode added to `cinm.op.reduce` in
M3b is consequently off this critical path, though it still fills a real
gap for memref-mode input programs.

## 1. What already exists

Reviewing the code (and running it) turned up considerably more working
infrastructure than the design doc assumed. Each of these shrinks or
redirects a milestone.

### 1.1 `#upmem.mram` / `#upmem.wram` already exist and are already used as memory spaces

`upmem::DpuMemSpaceAttr`
([UPMEMAttributes.td:25-38](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.td#L25-L38))
is an enum attr over `{MRAM, WRAM}` that prints as `#upmem.mram` /
`#upmem.wram`. It is already:

- the `memSpace` attribute of `upmem.static_alloc`
  ([UPMEMOps.td:153](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.td#L153));
- the **memref memory space** produced by `--convert-cnm-to-upmem`
  ([CnmToUPMEM.cpp:482-498](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L482-L498));
- what the C++ translator dispatches on
  (`isInMemspace`, [UPMEMTranslateToCpp.cpp:358-364](../lib/Target/UPMEMCpp/UPMEMTranslateToCpp.cpp#L358-L364))
  and what the Python simulator reads
  ([UpmemPythonSimulator.cpp:88-89](../lib/Dialect/UPMEM/Transforms/UpmemPythonSimulator.cpp#L88-L89)).

So M0 is not "define new attributes", it is "implement one interface on
an attribute that already exists and is already load-bearing".

### 1.2 `cnm.buffer`'s `level` field is already exercised end-to-end

Contrary to the design doc's inventory ("nothing ever sets it"),
`test/Conversion/CnmToUpmem/cnm-to-upmem.mlir:72-79` already writes
`!cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>` with matching
`memref<64xi32, #upmem.wram>` launch block arguments, and that input
lowers correctly through `--convert-cnm-to-upmem` today (verified by
running it). What is true is narrower: **`--convert-cinm-to-cnm` never
sets it** ([CinmToCnm.cpp:309-311](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L309-L311),
`// todo level is hardcoded`), and **`--convert-cnm-to-upmem` never
reads it** — `getLevel()` has no callers outside the type's own
printer and `LaunchOp::verify`. Today the field is decorative: writing
`#upmem.wram` or nothing produces identical output.

I also confirmed by construction that an MRAM-tagged buffer with
MRAM-tagged launch block arguments parses and verifies today — §A3's
type-level premise needs no new IR support.

### 1.3 BUG: `cnm.launch` does not round-trip level-carrying buffers

`printShorthandBufferType`
([CnmOps.cpp:74-79](../lib/Dialect/Cnm/IR/CnmOps.cpp#L74-L79)) prints
only shape and element type, while `parseShorthandBufferType`
([CnmOps.cpp:120-133](../lib/Dialect/Cnm/IR/CnmOps.cpp#L120-L133))
accepts an optional level. So printing then re-parsing a launch whose
operands carry a level fails:

```
error: use of value '%cnm_buf' expects different type than prior uses:
  '!cnm.buffer<64xi32 on ...>' vs '!cnm.buffer<64xi32 on ..., #upmem.mram>'
```

Latent today (nothing emits levels), but it becomes a hard blocker the
moment M1 lands, and it breaks any `cinm-opt | cinm-opt` test. Three-line
fix, scheduled first (M0).

### 1.4 Capacity cannot live on the level attribute

`DpuMemSpaceAttr` is a parameterless enum, and it must stay that way: it
is a memref memory space, so parameterizing it with a size would make
`memref<64xi32, #upmem.wram<size=57344>>` and `...<size=55296>`
*different types* on different platforms. But WRAM capacity genuinely
differs per platform — v1A 65536 vs v1B 63488, both less 8192
([UPMEMAttributes.cpp:145-163](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L145-L163)).

Hence decision 2: the interface carries identity only; size/alignment
stay in `CinmLevelDefAttr` on the platform. Every existing size consumer
already has the platform or accelerator in hand
(`handleGemv`'s `platform.getWramLevel().getSizeInElements()`,
`UpmemAcceleratorAttr::bufferSizeOfLeaf()`), so nothing needs rerouting.

A welcome consequence: because levels stay `CinmLevelDefAttr`,
`getWorkgroupMemoryLevels()` and `CinmLevelArrayAttr` need no change at
all — the `ArrayOfAttr`-can't-hold-an-interface problem the previous
draft worried about simply doesn't arise.

### 1.5 `getMemrefMemspace` already exists, is dead, and is wrong

`CinmPlatformAttrInterface::getMemrefMemspace(level)`
([CinmPlatformAttrInterface.td:45-52](../include/cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td#L45-L52))
is exactly the level→memspace mapping M1 needs. It has no callers, and
the UPMEM implementation returns `level.getName()` — a `StringAttr`
`"mram"` — where the rest of the compiler uses `#upmem.mram`. Fixing it
to return `DpuMemSpaceAttr` is a one-line change that hands M1 its
entire resolution path.

### 1.6 `--cinm-tiling` on memref operands works; on reductions it is broken

Verified by running both:

- **Memref-mode gemv tiles correctly.** `cinm.op.gemv %A, %x into %y`
  with `cinm.tile_sizes = array<i64: 16, 32>` produces the expected
  `affine.for` nest with an inner `cinm.op.gemv` accumulating in place
  into a `memref.subview` of `%y` across the K loop. Confirms decision 4
  needs no DPS-variant work for gemm-like ops. It also confirms
  decision 7's premise: the inner op comes out carrying **no**
  attributes from the outer one.
- **Reduce tiling silently drops all but the last reduction tile.** For
  `cinm.op.reduce add` over `tensor<64x128xi32>` with tile sizes
  `[16, 32]`, the generated inner loop is:

  ```mlir
  %0 = tensor.empty() : tensor<64xi32>              // uninitialised, not identity
  affine.for %i = 0 to 64 step 16 iter_args(%acc = %0) {
    affine.for %i_0 = 0 to 128 step 32 iter_args(%acc_1 = %acc) {
      %3 = cinm.op.reduce add(%slice) : tensor<16x32xi32> -> tensor<16xi32>
      %ins = tensor.insert_slice %3 into %acc_1[%i] [16] [1]   // OVERWRITES
  ```

  Each of the four K trips overwrites the accumulator slice with its own
  partial sum instead of combining. The combining path in
  `ReduceTilingModel` ([CinmTilingImplementations.cpp:334-347](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L334-L347))
  is only reached when the tile's result is *scalar*, i.e. when no
  non-reduced dimension remains; the shaped case
  ([:325-333](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L325-L333))
  always overwrites. Latent because nothing has yet tiled a reduction
  dimension into more than one trip — but `handleReduce` splits K across
  `dpuCols`/`mramCol`/`wramCol`, so this effort hits it immediately.

### 1.7 Test baseline is already red

`ninja check-cinm-mlir`: **32/42 passing, 9 failing, 1 unresolved.**
Failing tests this effort touches:

| Test | Status | Cause |
|---|---|---|
| `Conversion/CnmToUpmem/cnm-to-upmem.mlir` | FAIL | Stale CHECKs: pass now emits `upmem.broadcast` (not `upmem.scatter`) and `static_alloc ... noinit` |
| `Conversion/CnmToUpmem/cnm-to-upmem-broadcast.mlir` | FAIL | Stale CHECKs (second RUN line, `cinm1-codegen=true`) |
| `Dialect/UPMEM/upmem-to-c.mlir` | FAIL | Translation test, separate cause |
| `Transform/UPMEM/simulate-python.mlir` | FAIL | Simulator test, separate cause |
| `Dialect/UPMEM/upmem-infer-accelerator.mlir` | UNRESOLVED | No `RUN:` line at all |

(The other four failures — `CimToMemristor` ×2, `TorchToCinm`,
`Transform/Cim` ×2 — are unrelated to this work.)

Consequence for the plan: "confirm existing tests still pass" is not a
usable check for the CnmToUpmem tests. **M0 starts by refreshing those
CHECK lines** so later milestones have a trustworthy signal. Without
that, every subsequent milestone is working blind in exactly the area it
changes.

## 2. Milestone dependency graph

```
M0  round-trip fix + refresh red CnmToUpmem tests + level interface
 |
 ├─ M1  --convert-cinm-to-cnm  cnm-buffer-level option
 |
 ├─ M2  cnm.local_transfer op
 |
 └─ M3  cinm.op.reduce: fix tiled-reduction accumulation, add DPS mode
        |
M4  (withdrawn -- see decision 12; launch bodies stay linalg)
 |
M5  --upmem-tile-mram-buffers: tile + promote linalg on MRAM    (needs M1, M2)
 |
M5b hoist the output tile out of the reduction loop             (needs M5)
 |
M6  --convert-cnm-to-upmem: MRAM launch args, cnm.local_transfer (needs M5)
 |
M7  plugin wiring: search-param map, staged pipeline, path selector
 |
M8  --convert-linalg-to-cnm: block-size parameter, canonical order,
 |   no reduction split yet                            (needs M7, design §G1-G3)
 |
M9  reduction splitting: partial buffers, gather dim, host merge
 |                                                     (needs M8, design §G4-G6)
 |
M10 linalg-first pipeline: refresh --convert-cinm-to-linalg, add fusion
 |                                                     (needs M8, design §G8)
 |
M11 plugin: generic space builder from linalg indexing maps  (needs M9, M10)
 |
M11b one generic space for every op; named eval-solution params  (needs M11)
 |
M12 (not doing: --convert-cinm-to-cnm is the CINM 1.0 baseline's lowering)

--- added after the original plan, all outstanding ---

M13 sequential trips in --convert-linalg-to-cnm         (needs M11b, design §I)
M14 exact post-lowering occupancy check                 (needs M11b, design §H4)
M15 stop materializing the tiled layout on the host     (needs M11b)
```

M1, M2, M3 are mutually independent once M0 lands. M9 and M10 are
mutually independent once M8 lands. M13, M14 and M15 are mutually
independent.

## 3. Milestones

### M0 — Unblock: round-trip fix, red-test refresh, level interface — **DONE**

Landed as three commits. Deviations from the plan below, all discovered
while implementing:

- `getMemrefMemspace` returns `CinmLevelAttrInterface`, not `Attribute`
  — a stronger contract that costs nothing since UPMEM is the only
  implementer. Its inverse `getLevelOfMemspace` was added at the same
  time; M5 needs it to ask "which level is this memref in", and writing
  it now keeps the pair symmetric.
- `cnm-to-upmem-broadcast.mlir`'s second RUN line was not a stale CHECK.
  It asserted that `cinm1-codegen` disables the `upmem.broadcast`
  shortcut, which was never implemented: the shortcut is gated on
  `use-bc-xfer-codegen` alone. The two transfer forms deliver identical
  bytes to identical MRAM symbols, so `cinm1-codegen` has no reason to
  care — the code comment claiming the gating existed was simply wrong
  and has been corrected. The fallback case now uses the option that
  really controls it, and the `cinm1-codegen` case checks what it does
  control (private WRAM per tasklet, no `noinit`).
- The `CinmLevelAttrInterface` methods have no lit coverage yet; they
  have no IR-observable effect until M1 consumes them.

**Goal:** make the tree trustworthy and put the level interface in place.
Nothing here changes generated code.

**Changes:**

1. **Fix the round-trip bug (§1.3).** `printShorthandBufferType`
   ([CnmOps.cpp:74-79](../lib/Dialect/Cnm/IR/CnmOps.cpp#L74-L79)) must
   print `, <level>` when the buffer type has one, mirroring
   `parseShorthandBufferType`'s optional-comma form and
   `BufferType::print` ([CnmTypes.cpp:88-89](../lib/Dialect/Cnm/IR/CnmTypes.cpp#L88-L89)).
2. **Refresh the stale CHECK lines** in
   `test/Conversion/CnmToUpmem/cnm-to-upmem.mlir` and
   `cnm-to-upmem-broadcast.mlir` to match current output (`upmem.broadcast`
   for the tasklet-independent scatter, `noinit` on the static allocs).
   Leave `upmem-to-c.mlir` and `simulate-python.mlir` alone — unrelated
   causes, out of scope, but note them so nobody mistakes them for
   regressions from this work.
3. **Add `CinmLevelAttrInterface`** next to the existing platform/
   accelerator interfaces
   ([CinmPlatformAttrInterface.td](../include/cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td)).
   Identity only, per decision 2 — one method, `getLevelName() ->
   StringRef`, which is what lets generic code round-trip a memspace back
   to its capacity record via `platform.getLevel(name)`. No size, no
   alignment, no arity. This is deliberately minimal: it exists so §B's
   backend-specific transfer hooks have somewhere to live later, and so
   M5 can ask "which level is this memref in" without hardcoding UPMEM.
4. **Implement it on `DpuMemSpaceAttr`** — `getLevelName()` returns
   `stringifyDpuMemSpace(getValue())`, which already yields exactly
   `"mram"`/`"wram"`, matching the names `upmemLevels()` gives the
   corresponding `CinmLevelDefAttr`s. Add
   `DeclareAttrInterfaceMethods<CinmLevelAttrInterface>` to
   `UPMEM_DpuMemSpaceAttr`
   ([UPMEMAttributes.td:35-38](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.td#L35-L38)).
5. **Fix `UpmemPlatformAttr::getMemrefMemspace`** (§1.5) to return
   `DpuMemSpaceAttr` for the level whose name is `"mram"`/`"wram"`,
   instead of the level's name as a `StringAttr`
   ([UPMEMAttributes.cpp:225-228](../lib/Dialect/UPMEM/IR/UPMEMAttributes.cpp#L225-L228)).

Explicitly **not** done here: no changes to `CinmLevelDefAttr`,
`getLevels()`, `getWorkgroupMemoryLevels()`, or `CinmLevelArrayAttr`
(§1.4 — the ripple the earlier draft anticipated doesn't exist). The dead
`arity` parameter on `CinmLevelDefAttr` stays; removing it is unrelated
cleanup.

**Tests:**
- `test/Dialect/Cnm/cnm-launch-roundtrip.mlir` (new): a `cnm.launch` with
  `#upmem.mram`-level buffers, run as `cinm-opt %s | cinm-opt |
  FileCheck %s` plus a `--mlir-print-op-generic` variant (the existing
  double-round-trip idiom from `test/Dialect/Cnm/cnm-ops.mlir:1-2`).
  This test fails before the §1.3 fix and passes after — the direct
  regression test for it.
- Refreshed `cnm-to-upmem.mlir` / `cnm-to-upmem-broadcast.mlir` go green.

### M1 — `cnm-buffer-level` option on `--convert-cinm-to-cnm` — **DONE**

Landed as planned. `createLaunchOp` derives each block argument's memory
space from its own buffer's level rather than taking the level as a
separate parameter — `LaunchOp::verify` requires exactly that agreement,
so there is no second source of truth to get wrong.

Original plan follows.

**Goal:** decision 3. Buffers and launch block args get a level.

**Changes:**
- `CinmPasses.td`
  ([:10-15](../include/cinm-mlir/Conversion/CinmPasses.td#L10-L15)): add
  `Option<"bufferLevel", "cnm-buffer-level", "std::string", "\"\"", ...>`.
  Keyed by level *name*, resolved against the platform — the pass stays
  backend-agnostic.
- `CinmToCnm.h`: add a
  `createConvertTiledCinmToCnmPass(ConvertTiledCinmToCnmOptions)`
  overload, mirroring `CnmToUPMEM.h`'s existing options-taking factory.
- `CinmToCnm.cpp`: resolve once per pattern application —
  `platform.getLevel(bufferLevel)` for the capacity record, then
  `platform.getMemrefMemspace(record)` (M0.5) for the attribute to store.
  Empty option → null level (today's behavior, byte-identical output);
  unknown name → `emitOpError` and fail the pass. Thread the result into
  the three `cnm::BufferType::get` call sites
  ([:309-311](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L309-L311),
  [:886-893](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L886-L893)),
  resolving the `// todo level is hardcoded` comment.
- `createLaunchOp` ([:334-344](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L334-L344))
  currently builds block args as `MemRefType::get(shape, elt)` with no
  memory space. Add the level as the memory space. This is required for
  `LaunchOp::verify` to accept the result at all
  ([CnmOps.cpp:236-242](../lib/Dialect/Cnm/IR/CnmOps.cpp#L236-L242)),
  and is a latent inconsistency today independent of this option.

**Tests:**
- `test/Conversion/CinmToCnm/cinm-to-cnm-mram-buffers.mlir` (new,
  `--split-input-file`):
  1. `--convert-cinm-to-cnm=cnm-buffer-level=mram` on a small
     `cinm.op.gemv`: CHECK `!cnm.buffer<... , #upmem.mram>` on the allocs
     and `memref<..., #upmem.mram>` on the launch block args.
  2. `cnm-buffer-level=wram`: same shape, different level — proves the
     option is really consulted rather than hardcoded.
  3. `cnm-buffer-level=nosuchlevel` with `-verify-diagnostics`: clean
     error.
- Regression: `cinm-to-cnm.mlir` and `cinm-to-cnm-difficult.mlir` must
  stay green with no flag (these two are currently *passing*, so unlike
  the CnmToUpmem tests they are a trustworthy signal).

### M2 — `cnm.local_transfer` — **DONE**

Landed as planned.

Original plan follows.

**Goal:** decision 6.

**Changes:**
- Add to `CnmOps.td`, modeled 1:1 on `upmem.local_transfer`
  ([UPMEMOps.td:190-210](../include/cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.td#L190-L210)):
  `source`/`target` memref operands with `MemReadAt`/`MemWriteAt`
  effects, no results, verifier requiring equal shape and element type.
  Direction is implicit in the operands' memory spaces — no direction
  attribute, unlike `upmem.local_transfer`'s prose which mentions a
  "declared direction".
- No new dialect dependency (`cnm` already depends on `memref`).

**Tests:**
- `test/Dialect/Cnm/cnm-ops.mlir`: round-trip `cnm.local_transfer` both
  directions, including a `memref.subview`-of-MRAM source (the shape M5
  emits).
- `test/Dialect/Cnm/cnm-verifier.mlir` (new — `test/Dialect/Cnm/` has no
  verifier test today): shape mismatch and element-type mismatch, using
  `-verify-diagnostics` in the style of
  `test/Dialect/UPMEM/verifier.mlir`.

### M3 — `cinm.op.reduce`: fix tiled accumulation, then add DPS mode — **DONE**

Landed as two commits. Deviations from the plan below:

- The combine is a `linalg.map` over `arith::getReductionOp` on the
  element type, rather than a switch from `ReduceMethod` to a linalg
  named op. One code path covers every method, including the
  `maxnumf`/`maximumf` distinction a named-op switch would blur.
- The 3a regression test is not a new file. `Transform/Cinm/
  cinm-tiling-reduce.mlir` already existed and had locked in the broken
  output for its `@min` and `@mul` cases (`tensor.empty` seed +
  overwriting `insert_slice`), so the fix belongs there; a new
  `Dialect/Cinm/cinm-reduce-tiling.mlir` would have duplicated it.
- 3b rejects a *tensor* `into` operand rather than accepting it as a
  bufferization hint the way the gemm-like ops do. Nothing needs that
  form for reduce, and rejecting it keeps the memref mode's accumulate
  semantics unambiguous.
- Making the result optional removes the implicit `ReduceOp -> Value`
  conversion (two call sites in `SoftmaxToCinmPass`), and the reduce
  patterns in `CinmToLinalg` and `CinmToCnm` had to learn to bail out on
  memref mode — they would otherwise have dereferenced the null result.
  Both now mirror the guard the gemm-like patterns beside them already
  had.

Original plan follows.

**3a — fix the accumulation bug (§1.6).** In `ReduceTilingModel::
convertToTiledOps`
([CinmTilingImplementations.cpp:270-354](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L270-L354)):
- Seed the shaped accumulator with the reduction's identity instead of
  `tensor.empty()`. The identity is already computed at
  [:287-289](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L287-L289)
  and currently used only by the scalar branch.
- Make the shaped branch
  ([:325-333](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L325-L333))
  extract the accumulator slice, combine it with the tile result via
  `arith::getReductionOp`, and re-insert — the same shape the scalar
  branch already has, applied per-slice.

**3b — add memref/DPS mode.** `Cinm_ReduceOp` has no `out` operand and is
`Pure` ([CinmOps.td:191-223](../include/cinm-mlir/Dialect/Cinm/IR/CinmOps.td#L191-L223)),
so it cannot appear in a `cnm.launch` body over memrefs the way
gemm-like ops can (§1.6). Add `Optional<AnyShaped>:$out`, swap `Pure` for
`DeclareOpInterfaceMethods<MemoryEffectsOpInterface>` (matching
`Cinm_GemmlikeOp`), add an `isTensorVariant()` accessor, and extend the
tiling model to accumulate into `out` when it is a memref — mirroring
`GemmLikeTilingModel`'s `outBuf` branch
([:177-183](../lib/Dialect/Cinm/Transforms/CinmTilingImplementations.cpp#L177-L183)).
Scope: only the last-dimension reduction case `handleReduce` already
restricts itself to
([UpmemInferAccelerator.cpp:497-500](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L497-L500)).

**Tests:**
- `test/Dialect/Cinm/cinm-reduce-tiling.mlir` (new, 3a): tensor-mode
  reduce with the reduction dim split into 4 trips (the §1.6 repro).
  CHECK that the loop body extracts, combines and re-inserts, and that
  the accumulator is seeded with the identity constant. Fails before 3a.
- `test/Dialect/Cinm/cinm-reduce-memref-tiling.mlir` (new, 3b): memref
  mode accumulating into `%out`.
- `test/Dialect/Cinm/cinm-gemv-memref-tiling.mlir` (new): locks in the
  memref-mode gemv behavior verified in §1.6. Pre-existing coverage gap —
  `test/Dialect/Cinm/` has no `--cinm-tiling` test at all today.
- `test/Dialect/Cinm/cinm-ops.mlir`: round-trip the new
  `cinm.op.reduce ... into %out` syntax.

### M4 — `cinm` ops inside `cnm.launch` bodies — **WITHDRAWN**

Implemented, then reverted in favour of decision 12: the launch body
stays linalg and `--upmem-tile-mram-buffers` keys off the memref memory
space instead. What that revert kept, because both are real bugs
independent of it:

- the launch body's reduction dimension (see below), and
- the reduce scatter init, which was zero for every method and therefore
  made `mul` reductions always return zero.

The reasoning that led here is recorded below, since it is what motivated
decision 12.

Deviations from the original plan:

- **The buffer budget had to move first** (decision 11 above). Without it
  a multi-row per-leaf tile never fits, so the launch body is always a
  dot product and M4 has nothing to convert.
- **Gemm is not converted.** `ConvertCinmGemmToCnm` gives each leaf
  exactly one output element, so its per-leaf tile is always a dot
  product; there is no shape in which `cinm.op.gemm` could stand in its
  launch body. Gemv falls back to `linalg.contract` for the same reason
  when the tile happens to be a single row (lhs rank 1). This is a
  legality condition, not a strategy choice, but it meant M5 would only
  ever see gemv and reduce launch bodies. **This is what made the whole
  approach look wrong**, and led to decision 12.
- **The launch body's reduction dimension was hardcoded to 0** and had to
  be fixed to the buffer's last dimension. `convertInputIntoAlloc`
  prepends the parallel work that did not fit on the workgroup, so
  dimension 0 is only right when the buffer holds nothing but the
  reduction. Previously unreachable — a multi-row tile never fit the WRAM
  budget — and immediately reachable once the budget follows MRAM, where
  `linalg.reduce` rejected the op outright.
- The output-init question the plan flagged ("worth confirming this holds
  for the reduce path too") turned up a bug and was fixed here. The reduce
  pattern scattered a *zero* `outputInit` regardless of method, which is
  the identity only for `add`; a `mul` reduction was seeded with zero and
  therefore always returned zero. It now scatters
  `arith::getIdentityValueAttr` for the method. This was equally broken on
  the pre-existing linalg path, but the memref-mode `cinm.op.reduce` M4
  emits accumulates into `out` by definition, so the identity seed is now
  load-bearing rather than incidental.

Original plan follows.

**Goal:** decision 4, first half.

**Changes:**
- In `ConvertCinmGemmToCnm` / `ConvertCinmGemvToCnm` / the reduce
  pattern, the launch body is built by the callback passed to
  `createLaunchOp`, which today emits `linalg::ContractOp`
  ([CinmToCnm.cpp:927-933](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L927-L933),
  [:1036](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L1036)). When
  `bufferLevel` is set, emit the corresponding memref-mode `cinm` op over
  the block arguments instead. When unset, keep emitting linalg exactly
  as today — a new branch, not a rewrite of the existing one.
- The output buffer is already zero/bias-initialized by the existing
  scatter of `getOutputInitForGemmLike`
  ([:914-919](../lib/Conversion/CinmToCnm/CinmToCnm.cpp#L914-L919)),
  which is what memref-mode `cinm.op.gemv`'s accumulate-into-`out`
  semantics require. Worth confirming this holds for the reduce path too
  rather than assuming it.

**Tests:**
- Extend `cinm-to-cnm-mram-buffers.mlir`: a gemv case and a reduce case
  checking the launch body holds `cinm.op.gemv`/`cinm.op.reduce` over
  `memref<..., #upmem.mram>`; plus a no-flag case still producing
  `linalg.contract`.

### M5 — `--upmem-tile-mram-buffers` — **DONE**

**Goal:** decision 5, now via linalg rather than cinm ops (decision 12).
New UPMEM-dialect pass, new file
`lib/Dialect/UPMEM/Transforms/UpmemTileMRAMBuffers.cpp`, registered in
[UPMEM Passes.td](../include/cinm-mlir/Dialect/UPMEM/Transforms/Passes.td)
alongside `UPMEMDedupKernelsPass`.

**Changes.** For each `LinalgOp` inside a `cnm::LaunchOp` that has pure
buffer semantics and whose operands live in a non-leaf level:

1. Tile it with `scf::tileUsingSCF`, at the WRAM tile sizes M7 supplies.
   Tiling a buffer-semantics `LinalgOp` through `TilingInterface` is
   supported — the guards in
   [TilingInterfaceImpl.cpp](../third-party/llvm/mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp)
   are on `generateScalarImplementation` (wants buffers) and
   `LinalgOpPartialReductionInterface` (wants tensors, and we don't need
   partial reduction: memref `outs` accumulates in place).
2. Promote the resulting subviews with `linalg::promoteSubViews`
   ([Transforms.h:961](../third-party/llvm/mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h#L961)),
   configured through `LinalgPromotionOptions`:
   - `setMemorySpace` — the leaf level's memref memory space, obtained
     via `getMemrefMemspace` (M0.5), not hardcoded;
   - `setAllocationDeallocationFns` — **required**, see the note below;
   - `setCopyInOutFns` — emit `cnm.local_transfer` (M2) instead of the
     default `linalg.copy`.

   `promoteSubviewsPrecondition` wants exactly what step 1 produces:
   pure buffer semantics and operands defined by `memref.subview`.

"Non-leaf" is decided by mapping the memref's memory space back through
`getLevelOfMemspace` (M0.5) and comparing against
`getWorkgroupMemoryLevels()`, so no `mram`/`wram` literals appear in the
pass.

**Verified before writing any of this**: tiling then promoting a
`linalg.contract` over `#upmem.mram` memrefs, driven by the transform
interpreter on stock upstream, already produces M5's intended shape —
an `scf.for` nest, `#upmem.wram` allocations, copy-in, the contract on
WRAM operands, copy-out.

**The custom allocation callback is not optional.** The default scheme
allocates a flat `memref<8192xi8, #upmem.wram>` and takes a
`memref.view` of it, yielding *dynamically shaped* promoted buffers
(`memref<?x?xi32>`). UPMEM's `static_alloc` needs static sizes, so the
callback must emit a statically shaped `memref.alloc` directly.

Out of scope (§A3's deferred optimization): sharing the WRAM buffer
across tasklets via a static alloc plus barrier when the scatter map is
tasklet-independent. Always private WRAM here.

Also out of scope, deliberately: hoisting the output tile out of the
reduction loop. See M5b — land M5 correct first, then measure.

**Tests:**
- `test/Transform/UPMEM/upmem-tile-mram-buffers.mlir` (new): input is a
  `cnm.launch` holding an MRAM-level `linalg.contract` (the shape
  `cinm-to-cnm-launch-body.mlir` pins). CHECK the loop nest, the
  `#upmem.wram` allocs with *static* shapes, `cnm.local_transfer` before
  and after, and the inner op on WRAM operands.
- A `linalg.reduce` case, to demonstrate the pass is not
  contract-specific.
- Second contract case: MRAM tile already equals the WRAM tile (no
  second tiling round needed) — the degenerate shape `useMRAMTiling=false`
  produces ([UpmemInferAccelerator.cpp:427-440](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L427-L440)) —
  still gets alloc + transfers.

### M5b — hoist the output tile out of the reduction loop — **DONE**

Option 1 (existing loop-invariant passes) was tried and does not work:
`--loop-invariant-code-motion` will not move ops with memory effects, and
`--loop-invariant-subset-hoisting` threads subsets through loop
`iter_args`, which an `scf.for` over memrefs does not have. Both hoist
the `memref.subview` and leave the transfer and the allocation behind.
Option 2 (two-stage promotion) is what landed.

**Goal:** close the transfer-volume gap with the templates.

`promoteSubViews` places copy-in and copy-out immediately around the op,
so with the reduction loop innermost the output tile is read from and
written back to MRAM on *every* trip. The hand-written templates keep the
output tile in WRAM across the whole reduction and write it back once.
On a K-split gemv that is the difference between one output transfer and
`K/tile` of them.

**Approach, in order of preference:**
1. Try MLIR's existing loop-invariant machinery first —
   `--loop-invariant-code-motion` and
   `--loop-invariant-subset-hoisting`. The copy-in/copy-out pair is a
   loop-invariant *subset* (same subview, same buffer, every trip), which
   is what `loop-invariant-subset-hoisting` targets, though it is
   primarily exercised on tensor subset ops rather than on memref copies
   guarded by aliasing. Worth an experiment before writing anything: if
   it works, M5b is a pipeline entry, not a pass.
2. Failing that, promote in two stages within M5's pass: promote the
   output operand at the outer (parallel) loop level and the inputs at
   the inner (reduction) level, using `setOperandsToPromote` twice. This
   is purpose-built and certain to work, at the cost of M5 knowing which
   loops are reductions — available from
   `LinalgOp::getIteratorTypesArray()`.

**Tests:**
- Extend `upmem-tile-mram-buffers.mlir` with a case whose reduction
  dimension is split into several trips, CHECKing that the output
  `cnm.local_transfer` pair sits outside the reduction loop while the
  input transfers stay inside.
- The templates-vs-generic cost comparison in M7 is what says whether
  this actually closed the gap; M5b should not be declared done on IR
  shape alone.

### M6 — `--convert-cnm-to-upmem` for MRAM-level launches — **DONE**

**Goal:** decisions 4 and 6, backend side.

Today this pass ignores `getLevel()` and unconditionally gives every
`cnm.alloc` an MRAM static alloc *plus* a WRAM alloc with transfers
around the launch
([CnmToUPMEM.cpp:471-602](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L471-L602)).
That is exactly right for WRAM-level buffers and exactly wrong for
MRAM-level ones, whose staging M5 has already made explicit.

**Changes:**
- Branch on the buffer's level. WRAM-level (or absent, today's default):
  unchanged. MRAM-level: create the static MRAM alloc and bind the launch
  block argument directly to it — a `memref.subview` indexed by tasklet,
  or the whole allocation when `isMramBroadcastOverThreads`
  ([:424](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L424)) holds. The
  subview logic already exists inside `createTransfer`
  ([:335-367](../lib/Conversion/CnmToUPMEM/CnmToUPMEM.cpp#L335-L367));
  this generalizes it from "the transfers this pass inserts" to "every
  use of the block argument".
- Lower `cnm.local_transfer` → `upmem.local_transfer` (near-1:1).
- Confirm (don't assume) that `--cnm-ensure-scatter-gather-contiguous`
  needs no `cnm.local_transfer` handling — it targets host↔device
  scatter/gather, and M5 constructs these operands already correctly
  shaped.

**Tests:**
- `test/Conversion/CnmToUpmem/cnm-to-upmem-mram-level.mlir` (new): input
  with MRAM-level buffers and `cnm.local_transfer` inside the launch
  (M5-shaped, hand-written). CHECK a single `upmem.static_alloc
  ...(mram)`, a tasklet-indexed `memref.subview` of it, no spurious WRAM
  staging around the launch, and `upmem.local_transfer` where
  `cnm.local_transfer` was.
- Second case: tasklet-independent scatter map → the static alloc is used
  whole, no subview.

### M7 — Plugin wiring — **DONE except the cost comparison**

**Goal:** decisions 7, 8, 9.

**Changes:**
- **Search-param map (decision 7).** Add
  `DenseMap<Operation *, SmallVector<std::pair<std::string, SpaceVar>>>`
  to `UpmemInferencePlugin`. `handleGemv`/`handleReduce` populate it
  during `initializeSpace` with *every* SpaceVar the op's lowering will
  need (`mramRow`, `mramCol`, `wramRow`, `wramCol`, `dpuCols`,
  `taskletCols`), replacing the `kTileParamNamesAttr` stamping
  ([UpmemInferAccelerator.cpp:420-423](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L420-L423))
  and filling the `// fixme here` gap where `handleReduce` records
  nothing at all ([:524-527](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L524-L527)).
- **Trial correspondence.** `makeTrialInfo` deep-clones the reference
  module with no retained `IRMapping`
  ([AcceleratorInference.cpp:600-609](../lib/Dialect/Cinm/AcceleratorInference/AcceleratorInference.cpp#L600-L609)),
  so trial ops are different pointers. The clone is structurally
  identical, so a lockstep pre-order walk of `refClone` and
  `trial.computeBlock` recovers the correspondence. Resolve
  `SpaceVar`s against `trial.conf()` into a trial-local
  `DenseMap<Operation *, ...>` once, at the top of `evaluate()`.
  (I looked for an existing utility for this in `AcceleratorInference.cpp`
  and didn't find one — worth a second look before writing a new helper,
  in case it lives somewhere I missed.)
- **Staged pipeline (decision 8).** `buildPipeline()`
  ([:161-249](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L161-L249))
  becomes a sequence of sub-pipelines with stamping steps between them:
  stamp MRAM tile sizes → `--cinm-tiling` →
  `--convert-cinm-to-cnm=cnm-buffer-level=mram` → stamp WRAM tile sizes on
  the ops now inside the launch bodies → `--upmem-tile-mram-buffers` →
  the existing bufferize/affine/`--convert-cnm-to-upmem` tail. Each stamp
  writes a fresh `cinm.tile_sizes` from the trial-local map immediately
  before the pass that reads it; nothing relies on an attribute surviving
  a pass.
- **Path selection (decision 9).** `evaluate()` gains an explicit choice
  between the template path (today's `registerSimulator` loop, unchanged
  and *not* deleted) and the generic pipeline above. Suggest an enum
  option on `UpmemInferAcceleratorPass` — `lowering=templates|generic`,
  defaulting to `templates` while the generic path is under development —
  following the existing `simulatorClValues` enum-option idiom
  ([Passes.td:21-29](../include/cinm-mlir/Dialect/UPMEM/Transforms/Passes.td#L21-L29))
  rather than another bool. Deliberately *not* `useMRAMTiling`: that
  option means something else (whether `mramRow`/`mramCol` may differ
  from the WRAM tile, [:427-440](../lib/Dialect/UPMEM/Transforms/UpmemInferAccelerator.cpp#L427-L440))
  and stays independently selectable — "single-level tiling through the
  generic pipeline" is a combination worth being able to ask for.

**Tests:**
- `test/Transform/UPMEM/gemv-generic-mram-pipeline.mlir` (new): the whole
  chain driven by pass flags only, no plugin, no BO — `--cinm-tiling`
  (hand-annotated sizes) → `--convert-cinm-to-cnm=cnm-buffer-level=mram`
  → `--upmem-tile-mram-buffers` → `--cnm-ensure-scatter-gather-contiguous`
  → `--convert-cnm-to-upmem`, on a static-shape `cinm.op.gemv`.
  CHECK the final UPMEM structure: DPU alloc, MRAM static allocs, WRAM
  private allocs, transfers, kernel body. **Land this before the plugin
  work** — it is the milestone's real correctness signal, and it isolates
  pipeline bugs from plugin bugs.
- `test/Dialect/UPMEM/upmem-infer-accelerator.mlir`: give it a `RUN:`
  line (§1.7 — it is currently UNRESOLVED). Use `eval-solution` to pin a
  single configuration and skip search
  ([Passes.td:129-131](../include/cinm-mlir/Dialect/UPMEM/Transforms/Passes.td#L129-L131)).
- `test/Dialect/UPMEM/upmem-infer-accelerator-generic.mlir` (new): same
  fixed-solution technique with `lowering=generic`, checking the committed
  module really contains `upmem.dpu_program` / `upmem.local_transfer`.
- **Comparison harness (decision 9's actual purpose).** The point of
  keeping both paths is measuring the gap. Worth adding a small script or
  lit test that runs the same op under `lowering=templates` and
  `lowering=generic` with `eval-solution` pinned to the same
  configuration and reports both simulated costs. This is the milestone's
  success criterion, not a nice-to-have — without it "as good as the
  templates" is unmeasurable.

### M8 — `--convert-linalg-to-cnm`: mechanical distribution — **DONE**

Design §G1-G3. A new conversion that distributes a `linalg` op on
tensors onto a `cnm.workgroup`, driven entirely by a block-size vector.
Reduction splitting is *not* in this milestone: reject `f_d > 1` with a
clear diagnostic, so this is a like-for-like replacement of today's
behaviour on a principled footing.

- **Parameter.** `cnm.tile_sizes = array<i64: b_0, ..., b_{n-1}>`, one
  block size per iteration dimension of the op (§G2). Verify
  `∏(E_i / b_i) == |WG|` and report the mismatch with both numbers —
  that diagnostic is what M7's failure needed.
- **Order.** Parallel dims outer, reduction dims inner, op dim order
  within each group (§G3). A fixed rule, no option, no attribute. Write
  the rule down in a comment at the point of use; it is load-bearing
  and non-obvious.
- **Scatter map.** `indexingMap ∘ (workgroup → tile)`. No reshape, no
  transpose, no case analysis — contiguity belongs to
  `--cnm-ensure-scatter-gather-contiguous`.
- **Per-leaf buffer shape.** The `b_i` of the dims appearing in that
  operand's indexing map.
- **Launch body.** The same linalg op on the leaf-shaped memrefs, which
  is exactly the shape M5's `--upmem-tile-mram-buffers` already
  consumes (pinned by `cinm-to-cnm-launch-body.mlir`).

**Tests:** `test/Conversion/LinalgToCnm/linalg-to-cnm.mlir`,
`linalg-to-cnm-invalid.mlir`, and
`test/Transform/UPMEM/gemv-linalg-generic-pipeline.mlir` — the last one
runs the whole flags-only chain from a `linalg.contract` to a DPU
program, confirming `--upmem-tile-mram-buffers` and
`--convert-cnm-to-upmem` consume this pass's output unchanged.

**Landed as** `a99fc64`. One restriction worth remembering: indexing
maps must be projected permutations. Dropping a dimension is fine (that
is how broadcast operands work); repeating one is rejected.

### M9 — Reduction splitting — **DONE**

Design §G4-G6. The structural addition: allow `f_d > 1`.

- Output buffer holds a partial of the parallel tile shape; the gather
  destination gains a dimension of size `f_d`; a host-side
  `linalg.reduce` over that dim merges partials with the op's combiner.
- All leaves seeded with the reduction identity; the original `out`
  folded in exactly once at the merge (§G5).
- Legality gate: associative + commutative combiner required; floats
  behind an explicit opt-in flag, not silent (§G6).
**Open question 8 resolved:** `linalg::splitReduction` does the job.
It turns the reduction tile index into an extra *parallel* dimension and
leaves all indexing maps projected permutations, so M8's distribution
then applies unchanged — no `PartialReductionOpInterface` needed. It
also covers §G5 for free (neutral-element fill; merge accumulates into
the original `outs`).

**Tests:** `linalg-to-cnm-split-reduction.mlir` (new; K-split gemv,
`max` reduce with its own neutral element, float under the opt-in);
`linalg-to-cnm-invalid.mlir` (float without the opt-in, non-associative
`arith.subi` combiner).

**Landed as** `4289c03`.

**Known cost, deliberately not fixed here** (`0d9413d`): the identity
seed is a constant-zero buffer that gets scattered to every leaf on
every launch. `cnm.set_zero` is the right answer but only
`--convert-cnm-to-gpu` lowers it; `--convert-cnm-to-upmem` has no
pattern, and emitting it makes the backend conversion fail *silently*.
Fixing this needs device-side zeroing of an MRAM buffer — worth its own
milestone, and worth also giving `--convert-cnm-to-upmem` a diagnostic
for ops it cannot legalise.

### M10 — Linalg-first pipeline — **DONE (fusion deferred)**

Design §G8. The generic branch now runs `--convert-cinm-ops-to-linalg`
→ `--convert-linalg-to-cnm` in place of `--cinm-tiling` →
`--convert-cinm-to-cnm`. Landed as `1d1d094`, `5217f22`.

Two things fell out that the plan did not anticipate:

- **The host-side tiling round disappears.** `cnm.tile_sizes` is a block
  size and the workgroup takes the whole tile space at once, so there is
  no sequential outer loop left for `--cinm-tiling` to emit.
- **`stampLeafTileSizes` disappears with it.** Both levels are stamped
  up front on the same op and ride the conversions
  (`--convert-cinm-ops-to-linalg` forwards discardable attributes,
  `--convert-linalg-to-cnm` clones them into the launch body). Nothing
  has to re-find the op mid-pipeline, so the positional
  launch-to-op correspondence — the thing §G8 said could not survive
  fusion — is gone already.

**Fusion is deliberately not added.** The benchmarks (gemv, reduce) are
single-op compute blocks, so fusion would be a no-op for them, while
enabling it requires moving search-space construction after the
conversion (§G8: fusion changes the op count, the walk indices and the
iteration space). That refactor should be paid for by a benchmark that
needs it. The pipeline is arranged so fusion slots in right after the
linalg conversion when that day comes.

**Attribute plumbing was the real work here** (`1d1d094`, `190678e`).
Two silent-loss bugs had to be fixed for decisions to survive the
pipeline at all: `--convert-cinm-ops-to-linalg` dropped discardable
attributes, and `linalg::splitReduction` builds a fresh op so
`upmem.leaf_tile_sizes` vanished exactly when a reduction split —
producing slower code with no diagnostic, only for the configurations
that split.

### M11 — Plugin: projected search space — **DONE**

Design §G2, §G8. Landed as `5217f22`. The plan called for replacing the
per-op handlers with one generic handler; that turned out to be the
wrong move for now, and the *projection* is what was actually needed.

`mramRow`/`mramCol` are per-DPU and a DPU's tasklets subdivide that
tile, while a CNM leaf *is* a tasklet, so a leaf's share is

    b_m = mramRow * taskletCols / tasklets
    b_k = mramCol / taskletCols

Both divisions are exact given the constraints already in `handleGemv`,
and the generic path's own requirement — tile counts filling the
workgroup exactly — *follows* from those constraints rather than adding
one. This is the M7 defect fixed at its root: it fed `mramRow` straight
through, which is the per-DPU tile, not the per-leaf one.

**No new `SpaceVar`, by design.** `eval_solution` serialised
positionally, so inserting a variable would silently reinterpret every
params dict in `experiments/`. Parameter names and their order were left
unchanged; recording a *derived expression* rather than a variable is
what `SpaceValue` is for.

**Superseded by M11b below**, which removed that constraint at its root.

**Tests:** `Dialect/UPMEM/upmem-infer-accelerator-generic-split.mlir`
(new) pins the autotuner's gemv_64MB configuration end to end — the one
that used to fail with `numParallelElts (64) % numWgItems (16384)`. The
per-DPU MRAM footprint it produces (`8192 + 128 + 64` elements) is
exactly the template's own constraint `mramRow*mramCol + mramCol +
mramRow`.

### M11b — One generic space for every op — **DONE**

Design §H5, and the resolution of its "open decision — how it coexists".
M11 kept the per-op template handlers owning the space and *projected*
their variables onto the generic path's parameters. That was the right
first move but the wrong end state: the templates are meant to go, and
preserving a positional ABI for benchmarks that can simply be rerun is
not worth the coupling. Landed as `944135a`, `8481f33`, `fa2e72c`,
`ffaa9a9`, `73e6629`.

The direction is reversed. **The generic space is the only space**, built
by walking the linalg ops of the converted compute block — one block size
per iteration dimension per level, `∏(E_i/b_i) == dpus*tasklets`, leaf
block dividing the block. The *templates* now read those numbers back
through the §H5 projection, which is the inverse of what M11 did.

Three supporting changes made that possible:

- **`eval-solution` takes names** (`944135a`). `dpus=2048,tasklets=8,
  gemv.M0=8,…` instead of a positional list; unknown and missing names are
  both reported. This is design §H5's option 3, which the design doc
  called out as having independent merit — the positional coupling had
  already been a hazard twice.
- **Names are `<op>.<dim><level>`** (`73e6629`). `op0.block0` said where a
  parameter sat in a walk, not what it meant. Dimension names come from
  the originating `cinm` op — M/K for gemv, M/N/K for gemm, B/M/N/K for
  batch_gemm, leading parallel dims plus K for a trailing reduction —
  falling back to `D0, D1, …`. Op kinds are counted first, so a block with
  two gemvs gets `gemv0`/`gemv1` while the ordinary single-op case stays
  unadorned.
- **`cinm.lowered_from`** (`8481f33`) marks which `cinm` op a linalg op
  came from, which is what supplies those dimension names. It also fixed a
  crash: without it the walk handed `linalg.fill`'s scalar operand to
  `cast<ShapedType>`. Only computation-carrying ops get parameters.

**Capacity stays a cheap necessary condition** — the maximal-sharing
bound of §H4, which can never over-estimate. The exact check is M14.

**Benchmarks moved with it** (`ffaa9a9`): `experiments/` configs are
written in the generic names, and the `atim` optimum is spelled
`gemv.M0=8 gemv.K0=128 gemv.M1=8 gemv.K1=64` on both paths.

**Tests:** `upmem-infer-accelerator-generic-split.mlir` re-expressed in
the named space; the two `dodo.py` `atim` configs differ only in
`lowering=`, which is what makes them a same-configuration comparison.

### M12 — Delete the distribution heuristics — **NOT DOING**

The plan said to decide this with `grep`, not in advance. The grep says
keep them: `--convert-cinm-to-cnm` is the backbone of the **CINM 1.0
baseline flow** ([cinm1.py:72](../experiments/cinm_experiments/cinm1.py#L72),
transcribed from `testbench/Makefile`), which is a comparison point in
the experiments, and it is also the only cinm→cnm route for the GPU
backend. Deleting `computeShapeOfTensors`' case analysis would break a
published baseline to tidy a path nothing on the UPMEM side uses any
more.

The simplification §G9 promised is therefore *realised* rather than
*collected*: `--convert-linalg-to-cnm` has none of that machinery, and
the UPMEM generic path no longer runs any of it. The old code stays
where it is, serving the flow it was written for.

**Revisit when** the CINM 1.0 baseline is retired, or if the GPU backend
moves to `--convert-linalg-to-cnm`. Neither is on the critical path.

### M13 — Sequential trips — **TODO**

Design §I. §G2 requires the tile counts to fill the workgroup exactly and
parked sequential passes over a larger problem in `--cinm-tiling`, which
M10 then removed from the generic pipeline. Nothing has expressed trips
since, and the `cinm2` gemv_64MB baselines need them: with `dpus=256
tasklets=4` (1024 leaves) they cover 4096x4096 in 4 passes over M and 4
over K. Those two configs are **commented out** in
`experiments/gemv_microbenchmark/dodo.py` until this lands, deliberately
still spelling their old parameter names so they fail loudly rather than
being silently reinterpreted.

**Decided** (design §I, option 2 of three): derive the trips from the tile
counts rather than parameterize them. The space keeps exactly the
parameters §H5 gives it and the requirement relaxes from
`∏(E_i/b_i) == leaves` to `∏(E_i/b_i) == leaves * trips`, with the tile
space linearized in the §G3 order — low-order digits index the workgroup,
high-order digits index the trip loop.

**Changes:**
- Relax the count check in `--convert-linalg-to-cnm` and derive `trips`.
- **A trip boundary can fall inside a dimension** (§I1). The `cinm2`
  configuration's §G3-ordered counts are `[4, 4096, 1]` against 1024
  leaves, and no suffix of that product is 1024 — `m`'s 4096 tiles split
  4 outer × 1024 inner. So the tiled layout's tile dimension is
  `tensor.expand_shape`d into `[tripPart, leafPart]`, the trip parts are
  permuted to the front, and each trip `tensor.extract_slice`s its own
  chunk before scattering. The tidy "trips consume whole outermost
  dimensions" rule does not survive contact with a real configuration.
- **Trips over a reduction dimension accumulate across launches** (§I2).
  4 of those 16 trips come from `k_outer`. Each trip's leaves are seeded
  with the identity and merged by the gather (§G5), but that merge has to
  accumulate into the *running* total: the host loop carries the output as
  an `scf.for` iter_arg and folds the original `outs` in exactly once at
  the end — §G5's rule, one level up. An all-parallel trip loop needs none
  of this and is the easy case; it is not the case the benchmarks need.

**Tests:** a `linalg-to-cnm-trips.mlir` with (a) an all-parallel trip
loop, (b) the §I1 shape where the boundary falls inside a dimension, and
(c) the §I2 shape where a reduction dimension trips. Re-enabling `cinm2`
and `cinm2_partial_reduction` in `dodo.py` is the end-to-end signal.

### M14 — Exact post-lowering occupancy check — **TODO**

Design §H4. Occupancy is a property of the lowered program (§H3), so the
a-priori constraint keeps only what can never over-estimate — assume
maximal sharing — and the exact check runs on the lowered IR.

Standing decision 8 already has `evaluate()` run the real lowering, so the
UPMEM-dialect IR is in hand: sum the `upmem.static_alloc` sizes per memory
space and compare against the level's capacity. Shaped as a transform
returning `DiagnosedSilenceableFailure::silenceableFailure` so an
infeasible trial is rejected rather than crashing the search.

The cost is that infeasible configurations stay in the space and the
optimizer spends trials on them; the maximal-sharing filter keeps that
bounded. The same measurement also hands the cost model *real* occupancy
instead of modelled occupancy.

**Tests:** a configuration that fits and one that does not, both pinned
with `eval-solution`, checking the second is reported as silenceable
rather than hard-failing.

### M15 — Stop materializing the tiled layout on the host — **TODO**

`cnm.scatter` requires the host value's shape to *end with* the buffer
shape ([CnmOps.cpp:292-375](../lib/Dialect/Cnm/IR/CnmOps.cpp#L292-L375)),
so `--convert-linalg-to-cnm`'s `toTiledLayout` emits a real permuted copy
for any operand tiled in ≥2 dimensions. On gemv_64MB that is
`memref.alloc() : memref<512x32x1x8x1x128xi32>` — 16 M elements, a full
64 MB copy of `A` filled by a 4-deep `scf.for` nest — and it is **77.9 ms
of the generic path's 103.4 ms**. It is the single largest item between
the generic path and the templates, and therefore the gate on §F's
"equivalent code quality".

**Two candidate fixes, not yet chosen:**
1. **Give `cnm.scatter` a general affine map into the host buffer**
   instead of the shape-suffix requirement, so a strided DMA falls out of
   the map rather than out of a copy. The real answer, and the larger
   change — it touches the op's verifier, `--convert-cnm-to-upmem` and
   `--cnm-ensure-scatter-gather-contiguous`.
2. **Emit the permuted view as a strided `memref`** (no copy) and let
   `--cnm-ensure-scatter-gather-contiguous` decide where packing is
   genuinely required. Cheap to try and it measures how much of the repack
   is actually necessary, which informs (1).

Worth measuring (2) first; expect (1) to be what lands.

**Tests:** the existing `Transform/UPMEM/gemv-linalg-generic-pipeline.mlir`
gains a CHECK-NOT for a host-side `memref.alloc` of the operand's full
size; the hardware number in `experiments/gemv_microbenchmark` is the real
signal.

## 4. Test summary

| Milestone | Subject | Tests |
|---|---|---|
| M0 ✅ | launch round-trip fix; `CinmLevelAttrInterface` on `DpuMemSpaceAttr` | `Dialect/Cnm/cnm-launch-roundtrip.mlir` (new); `Conversion/CnmToUpmem/cnm-to-upmem.mlir` + `-broadcast.mlir` fixed, now green |
| M1 ✅ | `--convert-cinm-to-cnm=cnm-buffer-level=<name>` | `Conversion/CinmToCnm/cinm-to-cnm-buffer-level.mlir` (new; mram/wram/no-flag prefixes); `cinm-to-cnm-buffer-level-invalid.mlir` (new; unknown level); `cinm-to-cnm.mlir` + `-difficult.mlir` stay green |
| M2 ✅ | `cnm.local_transfer` | `Dialect/Cnm/cnm-local-transfer.mlir` (new, round-trip); `Dialect/Cnm/cnm-verifier.mlir` (new) |
| M3 ✅ | reduce accumulation fix; reduce DPS mode | `Transform/Cinm/cinm-tiling-reduce.mlir` (fixed: had locked in the bug); `Dialect/Cinm/cinm-reduce-memref-tiling.mlir` (new); `cinm-reduce-verifier.mlir` (new); `cinm-gemv-memref-tiling.mlir` (new, existing gap); `cinm-parse-memrefs.mlir` (extended) |
| M4 ⊘ | withdrawn; launch bodies stay linalg | `Conversion/CinmToCnm/cinm-to-cnm-launch-body.mlir` (new; pins the linalg-on-levelled-memrefs shape M5 consumes); `cinm-to-cnm-buffer-level.mlir` (extended: reduce dimension, `mul` identity) |
| M5 ✅ | `--upmem-tile-mram-buffers` (tile + promote) | `Transform/UPMEM/upmem-tile-mram-buffers.mlir` (new; contract, reduce, degenerate-tile cases) |
| M5b ✅ | output-tile hoisting | `upmem-tile-mram-buffers.mlir` (extend: split reduction, transfers outside the loop); M7's cost comparison is the real signal |
| M6 ✅ | MRAM launch args; `cnm.local_transfer` lowering | `Conversion/CnmToUpmem/cnm-to-upmem-mram-level.mlir` (new, 2 cases) |
| M7 ◐ | plugin map, staged pipeline, `lowering=` selector (cost comparison outstanding) | `Transform/UPMEM/gemv-generic-mram-pipeline.mlir` (new, flags-only end-to-end); `Dialect/UPMEM/upmem-infer-accelerator.mlir` (add RUN line); `upmem-infer-accelerator-generic.mlir` (new); templates-vs-generic cost comparison |
| M8 ✅ | `--convert-linalg-to-cnm`, block sizes + canonical order | `Conversion/LinalgToCnm/linalg-to-cnm.mlir` (new); `linalg-to-cnm-invalid.mlir` (new); `Transform/UPMEM/gemv-linalg-generic-pipeline.mlir` (new, flags-only end-to-end) |
| M9 ✅ | reduction splitting via `linalg::splitReduction` | `linalg-to-cnm-split-reduction.mlir` (new: K-split gemv, `max` neutral element, float opt-in); `linalg-to-cnm-invalid.mlir` (extended: float without opt-in, non-associative combiner) |
| M10 ✅ | linalg-first pipeline (fusion deferred) | `Transform/UPMEM/gemv-linalg-generic-pipeline.mlir` (new); `gemv-split-k-grouping.mlir` (new) |
| M11 ✅ | projected search space for the generic path | `Dialect/UPMEM/upmem-infer-accelerator-generic-split.mlir` (new, the autotuner's gemv_64MB configuration) |
| M11b ✅ | one generic space for every op; named `eval-solution` parameters | `upmem-infer-accelerator-generic-split.mlir` (re-expressed in `gemv.M0`/`gemv.K0`/…); `experiments/` configs moved to the same names |
| M12 ⊘ | not doing: `--convert-cinm-to-cnm` is the CINM 1.0 baseline's lowering | — |
| M13 ☐ | sequential trips (§I) | `linalg-to-cnm-trips.mlir` (all-parallel, boundary inside a dim, reduction trip); `cinm2` + `cinm2_partial_reduction` re-enabled in `dodo.py` |
| M14 ☐ | exact post-lowering occupancy check (§H4) | fitting vs. non-fitting configuration under `eval-solution`, the second reported as silenceable |
| M15 ☐ | stop materializing the tiled layout on the host | `gemv-linalg-generic-pipeline.mlir` CHECK-NOT on the full-size host alloc; the hardware number is the real signal |

---

<!-- Add your comments below or inline as blockquotes -->

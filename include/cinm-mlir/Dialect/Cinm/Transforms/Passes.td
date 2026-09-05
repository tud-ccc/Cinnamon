//===- Passes.td - Cinm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_TRANSFORM_PASSES
#define CINM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def CinmTilingPass: Pass<"cinm-tiling"> {
  let summary = "Applys tiling to all operations implementing the cinm tiling interface";
  let description = [{}];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::affine::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def CinmIsolateComputePass: Pass<"cinm-isolate-compute-blocks"> {
  let summary = "Turn compute ops into compute ops (isolated from above)";
  let description = [{}];

  let dependentDialects = [
  ];
}

def CinmDeisolateComputeBlocks: Pass<"cinm-deisolate-compute-blocks"> {
  let summary = "Turn compute ops into compute ops (not isolated from above)";
  let description = [{}];

  let dependentDialects = [
  ];
}


def CinmUnwrapComputeBlocksPass: Pass<"cinm-unwrap-compute-blocks"> {
  let summary = "Unwrap all compute and compute ops";
  let description = [{}];

  let dependentDialects = [
  ];
}


def SoftmaxToCinmPass: Pass<"convert-linalg-softmax-to-cinm"> {
  let summary = "converts the linalg::softmax op to cinm";
  let description = [{}];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::linalg::LinalgDialect",
  ];
}

def CinmGemvMinWritePass : Pass<"cinm-gemv-min-write"> {
  let summary = "Reschedule cinm GEMV tiles to minimize writes back to the first operand";
  let description = [{}];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def CinmCleanupLinalgPass : Pass<"cinm-cleanup-linalg", "mlir::func::FuncOp"> {
  let summary = "Cleanup Linalg tensor ops prior to CINM lowering";
  let description = [{
    Replaces tensor results of linalg ops with their output operands when they
    are immediately yielded, ensuring loop-carried tensors remain equivalent.
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def CinmDecomposeAccumulationPass : Pass<"cinm-decompose-accum"> {
  let summary = "Decompose accumulating GEMV/GEMM into pure compute + explicit add";
  let description = [{
    Rewrites:
      - `cinm.gemv_memref(A, x, out)`  (semantics: out += A * x)
      - `cinm.gemm_memref(A, B, out)` (semantics: out += A * B)
    into:
      tmp = alloc like(out)
      linalg.fill tmp = 0
      cinm.gem{v,m}_memref(..., tmp)   // compute partial into tmp
      cinm.add_memref(out, tmp, out)   // explicit accumulation
      dealloc tmp

    This pass is independent of tiling/bufferization and can run after tiling
    and before Cinm→Cim conversion.
  }];

  let dependentDialects = [
    "mlir::cinm::CinmDialect",
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}


def CinmRelowerPass : Pass<"cinm-relower", "mlir::func::FuncOp"> {
  let summary = "Lower CINM tensor ops outside compute blocks back to linalg";
  let description = [{
    Finds CINM tensor ops (currently cinm.op.add / cinm.op.sub) that are not
    nested inside a `cinm.compute_block` and rewrites them to equivalent
    `linalg.generic` elementwise operations. This is useful for bufferization
    pipelines that expect raw linalg.
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}


def CinmGemmToLoopedGemvPass : Pass<"cinm-gemm-to-gemv"> {
  let summary = "Rewrite tensor cinm.op.gemm into a loop of cinm.op.gemv over a chosen free dimension (i or j)";
  let description = [{
    For C = A[M,K] x B[K,N], choose which free (non-reduction) dimension to peel
    into an outer loop with --split-dim:

      * --split-dim=1  (peel i / M):
          for i in [0..M):
            a_i : tensor<K>          // row of A
            y   = B^T[N,K] * a_i[K]  // GEMV → tensor<N>
            C[i, :] = y

      * --split-dim=2  (peel j / N):
          for j in [0..N):
            x   : tensor<K>          // column of B
            y   = A[M,K] * x[K]      // GEMV → tensor<M>
            C[:, j] = y

    Choosing the reduction dimension k is invalid and rejected.

    If an enclosing `cinm.compute_block` carries GEMM tileSizes [M,N,K], the pass
    retargets them to GEMV-style sizes matching the split:
      - split-dim=1 → [N, K]
      - split-dim=2 → [M, K]
  }];

  let options = [
    Option<"splitDimOpt", "split-dim", "unsigned", "2",
           "Which free dimension to peel: 1 -> i (M), 2 -> j (N).">
  ];

  let dependentDialects = [
    "mlir::cinm::CinmDialect",
    "mlir::affine::AffineDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}

def InsertCinmQuantization
  : Pass<"cinm-insert-quantization", operation="func::FuncOp"> {
  let summary = "Insert Quantize -> CINM op -> Dequantize around selected CINM ops";
  let description = [{
    Inserts per-tensor affine Quantize/Dequantize around selected CINM tensor ops.
    Supported now: cinm.op.gemm, cinm.op.gemv (tensor forms). Memref forms are
    accepted in --ops but skipped with a warning for now.
  }];

  let dependentDialects = [
    "cinm::CinmDialect",
    "arith::ArithDialect",
    "tensor::TensorDialect",
    "memref::MemRefDialect",
    "func::FuncDialect"
  ];

  let options = [
    Option<"ops", "ops", "std::string", "\"gemm,gemv\"",
      "Comma-separated CINM ops to quantize (choices: gemm, gemv, gemm_memref, gemv_memref)">,

    Option<"qElementTypeStr", "qtype", "std::string", "\"i8\"",
      "Element type for quantized tensors/memrefs (e.g. i8, i16, int8)">,

    Option<"scale", "scale", "double", "1.0",
      "Per-tensor quantization scale (must be > 0)">,
    Option<"zeroPoint", "zp", "int64_t", "0",
      "Per-tensor zero-point (fits in qtype)">,

    Option<"roundingStr", "rounding", "std::string", "\"nearest\"",
      "Rounding mode: nearest | towards_zero">,
    Option<"narrowRange", "narrow-range", "bool", "false",
      "Use (qmin+1 .. qmax) instead of (qmin .. qmax)">
  ];
}


def CinmMemoryCleanupPass
  : Pass<"cinm-memory-cleanup", "mlir::ModuleOp"> {
  let summary = "Fuse tmp+activate+copy and lower dynamic memref.copy to loops";
  let description = [{
    Post-bufferization cleanups for Cinm:

      (A) Fuse tmp + `cinm.activate_memref` + `memref.copy(tmp -> dst)` into an
          in-place activate that writes directly to `dst` (drops tmp allocs).

      (B) Lower any `memref.copy` that involves a dynamic shape (any rank) to an
          explicit `scf.for` loop nest. This avoids asserts inside MemRef
          canonicalizations that require static shapes.

    Run it where you currently hit dynamic-shape `memref.copy` problems.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::affine::AffineDialect"
  ];

  let options = [
    Option<"fuseActivateTmpCopy", "fuse-activate-tmp-copy", "bool", "true",
           "Fuse tmp + activate + copy into in-place activate">,

    Option<"lowerDynRank1Copy", "lower-dyn-rank1-copy", "bool", "true",
           "Lower dynamic-shape memref.copy to loops (works for any rank)">,

    Option<"lowerAnyDynamicCopy", "lower-any-dynamic-copy", "bool", "false",
           "Alias for enabling dynamic-shape memref.copy lowering">
  ];
}

def CinmAssignPlatformsPass : Pass<"cinm-assign-platforms", "mlir::func::FuncOp"> {
  let summary = "Wrap cinm.op.* operations in cinm.compute based on platform interest";
  let description = [{
    Walks a func.func that carries cinm.available_platforms and, for each
    cinm.op.* operation, queries every listed platform via isOffloadingTarget.
    If at least one platform is interested, the operation is wrapped in a
    cinm.compute op annotated with the interested platforms.
  }];

  let dependentDialects = [
    "mlir::cinm::CinmDialect",
    "mlir::func::FuncDialect"
  ];
}

def CinmExpandComputeScopePass : Pass<"cinm-expand-compute-scope"> {
  let summary = "Grow cinm.compute_block regions by absorbing the ops that initialize their operands";
  let description = [{
    Widens the scope of `cinm.compute_block` operations so that the
    initialization of their inputs happens inside the offloaded region, where
    it can become on-device initialization instead of a host computation plus
    a transfer: the op producing a block operand is rematerialized inside the
    block, and the block takes what that op needs instead. This is done for
    the pure ops that only initialize data, namely `tensor.splat`,
    `linalg.fill`, constants, and the `linalg.generic` ops that are a fill,
    a broadcast or a transpose. A `tensor.empty` stays outside and becomes a
    block operand (the absorbed fill then writes the operand buffer in
    place): absorbed, it would bufferize to an allocation inside the
    isolated region, which `--buffer-hoisting` breaks out of.
    Producers are pulled in transitively, so e.g. a `linalg.fill` of a
    `tensor.empty` moves in as a whole and only the fill value is passed in.
    A producer that captures values in its region (a `linalg.generic` in
    fill form holds the fill value in its body) has those moved in too.

    A `tensor.extract` that unpacks a single-element block result also moves
    into the block, which then yields the scalar directly: the one-element
    tensor is an implementation detail of linalg's reduction form, and
    yielding it would materialize a one-element buffer outside the block.

    Slicing ops (`tensor.extract_slice`) and other result consumers
    (`tensor.insert_slice`) deliberately stay outside the block. After loop
    unrolling their offsets are per-iteration constants, and absorbing them
    bakes those constants into otherwise identical block bodies, which makes
    the graph scheduler's signature-based deduplication see one class per
    iteration instead of one class overall. Staticness is not lost by leaving
    a slice outside: `isStaticValue` follows static-offset slices to the
    annotated weight behind them.

    Only `cinm.compute_block` is handled: the operands of an isolated block are
    explicit, so the data crossing the offloading boundary is unambiguous. Run
    `--cinm-isolate-compute-blocks` first to turn `cinm.compute` ops (as
    produced by `--cinm-assign-platforms`) into `cinm.compute_block` ops.
  }];

  let dependentDialects = [
    "mlir::cinm::CinmDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def CinmAbsorbResultDestinationsPass
    : Pass<"cinm-absorb-result-destinations"> {
  let summary = "Destination-pass compute block results by absorbing their consumers";
  let description = [{
    Moves the destination-style consumer of a `cinm.compute_block` result --
    a `tensor.insert_slice` inserting it somewhere, or a
    `bufferization.materialize_in_destination` pinning where it lands -- into
    the block. The destination becomes a block operand and the block yields
    the full destination tensor, so the block is destination-passed: a global
    bufferization can then arrange for the block to write its result straight
    into the destination buffer, with no copy at the block boundary.

    This deliberately does NOT run before the accelerator search: absorbing a
    consumer bakes its constant offsets into the block body, which makes
    structurally identical blocks look distinct to the graph scheduler's
    signature-based deduplication (see --cinm-expand-compute-scope). It is a
    finalization rewrite: run it after configurations are stamped and before
    --upmem-lower-stamped, which also runs it itself.
  }];

  let dependentDialects = [
    "mlir::cinm::CinmDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def CinmSimplifyComputeSignaturesPass
    : Pass<"cinm-simplify-compute-signatures"> {
  let summary = "Drop unused arguments and passthrough results of compute ops";
  let description = [{
    Narrows the boundary of `cinm.compute` and `cinm.compute_block` ops:

      * a result whose yielded value is defined outside the region (for
        `cinm.compute`) or is a passthrough of a block argument (for
        `cinm.compute_block`) is dropped, and its uses are redirected to the
        outer value;

      * a `cinm.compute_block` argument with no uses inside the region is
        dropped together with its operand. A forwarded workgroup
        (WorkgroupTypeInterface) is kept however unused: it is a residency
        declaration that only the lowering consumes.

    These rewrites used to be canonicalization patterns, but they change the
    op's signature, and the accelerator search relies on a block keeping its
    boundary through the trial pipelines (which run the canonicalizer many
    times): the winning trial's body is spliced back into the original op by
    zipping operands and results positionally. Keeping them out of the
    canonicalizer and in this pass, which the search pipelines never run,
    is what makes that splice sound.
  }];

  let dependentDialects = [
    "mlir::cinm::CinmDialect"
  ];
}

def CinmCompleteComputeGraphPass : Pass<"cinm-complete-compute-graph"> {
  let summary = "Wrap the host code between compute blocks into host compute blocks, connecting the compute graph";
  let description = [{
    For every function that already contains `cinm.compute` ops, wraps the
    remaining operations of the function body into new `cinm.compute` ops
    pinned to `#cinm.host_platform`, so that every operation belongs to some
    compute block and the dataflow between blocks forms a single connected
    graph -- the input the graph scheduling algorithm needs to see the whole
    application, including the parts that run on the host CPU.

    Maximal runs of consecutive non-compute ops become one host block each.
    Constants and `tensor.empty` are left outside: they are free, and
    yielding an undefined tensor out of an isolated block would leave
    bufferization no sane place for the allocation.

    A control flow op that contains no compute op is absorbed whole into a
    host block. One that does contain compute ops cannot be wrapped: by
    default the pass emits a warning and leaves it in place, which leaves the
    graph disconnected around it. With `demote-nested-compute`, the nested
    compute ops are dissolved into plain host code instead, so the whole
    control flow op becomes a single host node. A call whose callee contains
    compute ops is likewise left in place (with a warning); run `--inline`
    first to avoid those.

    Functions that contain no compute op at all are left untouched.
  }];

  let options = [
    Option<"demoteNestedCompute", "demote-nested-compute", "bool", "false",
           "Dissolve compute ops nested inside control flow ops so the whole "
           "control flow op can be wrapped as a single host block">
  ];

  let dependentDialects = [
    "mlir::cinm::CinmDialect"
  ];
}

def CinmInferTileSizesPass : Pass<"cinm-infer-tile-sizes"> {
  let summary = "Infer and annotate cinm.tile_sizes on tileable ops using the enclosing accelerator";
  let description = [{
    For each cinm op that implements CinmTilingInterface and does not already
    have a cinm.tile_sizes attribute, queries the enclosing accelerator via
    CinmAcceleratorAttrInterface::computeTilingFactors and writes the result
    as cinm.tile_sizes on the op. Ops without an enclosing accelerator or for
    which the accelerator cannot determine tile sizes are left unchanged.
  }];
}



#endif // CINM_TRANSFORM_PASSES

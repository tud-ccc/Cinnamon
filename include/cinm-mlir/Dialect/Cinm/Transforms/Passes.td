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



def CinmAnnotateTileSizesPass : Pass<"cinm-annotate-tiles"> {
  let summary = "Attach tileSizes selectively";
  let description = [{
    Attach tiling metadata only to selected ops (compute, gemm, gemv, activate).
    Use --ops=... to choose, and --tile-sizes to provide sizes:
      - activate: single size (e.g. 256)
      - gemv: MxK
      - gemm: MxNxK
      - compute: forwarded as-is
    Separators may be 'x', 'X', or ','.
  }];

  let options = [
    Option<"tileSizesOpt", "tile-sizes", "std::string", "",
           "Tile sizes (activate: S | gemv: MxK | gemm: MxNxK; commas also ok)">,
    Option<"opsOpt", "ops", "std::string", "",
           "Comma-separated list of kinds to annotate: compute,gemm,gemv,activate.">
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

def RunCostModelPass: Pass<"run-cost-model"> {
  let summary = "";
  let description = [{}];

  let options = [
    Option<"costModelPath", "costModelPath", "std::string", "", "path to the costmodel">
  ];

  let dependentDialects = [
    "cinm::CinmDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
  ];
}

def CostModelFinalizePass: Pass<"cost-model-finalize"> {
  let summary = "";
  let description = [{}];
}


#endif // CINM_TRANSFORM_PASSES

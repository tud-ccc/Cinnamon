//===- Passes.td - Cinm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_TRANSFORM_PASSES
#define CINM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def CinmTileGemmOpPass: Pass<"cinm-tile-gemm-op"> {
  let summary = "Converts a cinm.gemm op to tiled affine.for loops.";
  let description = [{}];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::affine::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
  ];
}

#endif // CINM_TRANSFORM_PASSES

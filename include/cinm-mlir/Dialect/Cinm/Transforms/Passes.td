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

  let options = [
    Option<"reductionTileSize", "reduction-tile-size", "uint64_t", /*default=*/"32",
           "Set size of batches for reduction loop (default: 32)">
  ];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::affine::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // CINM_TRANSFORM_PASSES

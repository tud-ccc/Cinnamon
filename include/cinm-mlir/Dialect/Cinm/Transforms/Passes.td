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

def SoftmaxToCinmPass: Pass<"softmax-to-cinm"> {
  let summary = "converts the linalg::softmax op to cinm";
  let description = [{}];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::linalg::LinalgDialect",
  ];
}

#endif // CINM_TRANSFORM_PASSES

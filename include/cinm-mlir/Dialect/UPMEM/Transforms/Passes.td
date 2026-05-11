//===- Passes.td - UPMEM dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_TRANSFORM_PASSES
#define UPMEM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def UPMEMDedupKernelsPass: Pass<"upmem-dedup-kernels", "ModuleOp"> {
  let summary = "Deduplicate the DPU kernels within an upmem module";
  let description = [{
  }];
  let dependentDialects = [
    "::mlir::upmem::UPMEMDialect"
  ];
}

def UpmemInferAcceleratorPass : Pass<"upmem-infer-accelerator", "ModuleOp"> {
  let summary = "Infer UPMEM accelerator parameters for cinm.compute ops";
  let description = [{
    For each cinm.compute op that declares a UPMEM platform via
    cinm.available_platforms, runs Bayesian optimization to find the best
    combination of ranks, dpus_per_rank, tasklets_per_dpu, and tiling factors.
    Annotates the op with the resulting accelerator attribute and sets
    cinm.tile_sizes on each tileable op in the compute body.
  }];
  let dependentDialects = [
    "::mlir::cinm::CinmDialect",
    "::mlir::cnm::CnmDialect",
    "::mlir::upmem::UPMEMDialect",
    "::mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::affine::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"maxEvals", "max-evals", "int", "50",
           "Maximum number of Bayesian optimization evaluations">
  ];
}

#endif // UPMEM_TRANSFORM_PASSES

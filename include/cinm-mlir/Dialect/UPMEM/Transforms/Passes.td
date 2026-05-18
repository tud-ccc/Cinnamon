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
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect"
  ];
  let options = [
    // InferenceOptions fields
    Option<"maxEvals", "max-evals", "int", "50",
           "Total number of valid BO evaluations (LHS init + surrogate-guided)">,
    Option<"nInit", "n-init", "int", "10",
           "Number of LHS evaluations before the surrogate model takes over">,
    Option<"rngSeed", "rng-seed", "int", "42",
           "RNG seed for reproducibility">,
    Option<"kappa", "kappa", "double", "2.0",
           "UCB exploration weight for the BANANAS surrogate">,
    Option<"epochs", "epochs", "int", "200",
           "Training epochs per MLP ensemble member">,
    Option<"nEnsemble", "n-ensemble", "int", "5",
           "Number of MLP ensemble members">,
    Option<"hidden", "hidden", "int", "64",
           "Hidden layer width of each ensemble MLP">,
    Option<"depth", "depth", "int", "2",
           "Number of hidden layers in each ensemble MLP">
  ];
}

#endif // UPMEM_TRANSFORM_PASSES

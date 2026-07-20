//===- Passes.td - Cnm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_CONVERSION_PASSES
#define CNM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

#ifdef CINM_GPU_SUPPORT
def ConvertCnmToGPUPass : Pass<"convert-cnm-to-gpu"> {
    let summary = "";
    let constructor = "mlir::cnm::createConvertCnmToGPUPass()";
    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::bufferization::BufferizationDialect",
        "mlir::gpu::GPUDialect",
        "mlir::memref::MemRefDialect",
        "mlir::tensor::TensorDialect"
    ];
}
#endif

def ConvertCnmToUPMEMPass : Pass<"convert-cnm-to-upmem"> {
    let summary = "";
    let constructor = "mlir::cnm::createConvertCnmToUPMEMPass()";
    let options = [
      Option<"kernelModuleName", "kernel-module-name", "std::string",
             [{"dpu_kernels"}],
             "Name of the submodule to create DPU kernels in">,
      Option<"cinm1Codegen", "cinm1-codegen", "bool", "false",
             "Whether to match CINM 1 code generation (no barriers, tasklets don't share buffers)">
    ];
    let dependentDialects = [
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::memref::MemRefDialect",
        "mlir::tensor::TensorDialect",
        "mlir::upmem::UPMEMDialect"
    ];
}

#endif // CNM_CONVERSION_PASSES

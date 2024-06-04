//===- Passes.td - Cnm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_CONVERSION_PASSES
#define CNM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCnmToGPUPass : Pass<"convert-cnm-to-gpu"> {
    let summary = "";
    let constructor = "mlir::cnm::createConvertCnmToGPUPass()";
    let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::gpu::GPUDialect"];
}

def ConvertCnmToUPMEMPass : Pass<"convert-cnm-to-upmem"> {
    let summary = "";
    let constructor = "mlir::cnm::createConvertCnmToUPMEMPass()";
    let dependentDialects = ["mlir::upmem::UPMEMDialect"];
}

#endif // CNM_CONVERSION_PASSES

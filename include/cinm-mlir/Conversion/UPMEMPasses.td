//===- Passes.td - UPMEM dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_CONVERSION_PASSES
#define UPMEM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertUPMEMToLLVMPass : Pass<"convert-upmem-to-llvm", "ModuleOp"> {
    let summary = "";
    let constructor = "mlir::upmem::createConvertUPMEMToLLVMPass()";
    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::memref::MemRefDialect",
        "mlir::tensor::TensorDialect",
        "mlir::upmem::UPMEMDialect"
    ];
}


#endif // UPMEM_CONVERSION_PASSES//===- Passes.td - UPMEM dialect passes ---------------------*- tablegen -*-===//
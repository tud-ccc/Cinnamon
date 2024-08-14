//===- Passes.td - Hbmpim dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Hbmpim dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef HBMPIM_CONVERSION_PASSES
#define HBMPIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


// def HbmpimRewriteDeviceCallsPass : Pass<"hbmpim-rewrite-device-calls", "ModuleOp"> {
//     let summary = "";
//     let constructor = "mlir::upmem::createHbmpimRewriteDeviceCallsPass()";
//     let dependentDialects = [
//         // "mlir::LLVM::LLVMDialect",
//         // "mlir::affine::AffineDialect",
//         // "mlir::arith::ArithDialect",
//         // "mlir::memref::MemRefDialect",
//         // "mlir::tensor::TensorDialect",
//         "mlir::hbmpim::HbmpimDialect"
//     ];
// }


#endif // HBMPIM_CONVERSION_PASSES//===- Passes.td - Hbmpim dialect passes ---------------------*- tablegen -*-===//
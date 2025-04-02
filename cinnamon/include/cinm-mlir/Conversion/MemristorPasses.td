//===- Passes.td - Memristor dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Memristor dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef MEMRISTOR_CONVERSION_PASSES
#define MEMRISTOR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemristorToFunc : Pass<"convert-memristor-to-func", operation="func::FuncOp"> {
    let summary = "Convert memristor dialect to func dialect";
    let constructor = "mlir::memristor::createConvertMemristorToFuncPass()";
    let dependentDialects = ["memristor::MemristorDialect", "func::FuncDialect", 
        "memref::MemRefDialect", "tensor::TensorDialect"];
}

#endif // MEMRISTOR_CONVERSION_PASSES

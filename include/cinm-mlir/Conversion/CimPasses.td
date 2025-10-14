//===- Passes.td - Cim dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cim dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_CONVERSION_PASSES
#define CIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCimToMemristorPass : Pass<"convert-cim-to-memristor", operation="func::FuncOp"> {
    let summary = "Convert cim dialect to memristor dialect";
    let constructor = "mlir::cim::createConvertCimToMemristorPass()";
    let dependentDialects = ["cim::CimDialect", "memristor::MemristorDialect",
        "bufferization::BufferizationDialect", "memref::MemRefDialect", "tensor::TensorDialect"];
}

#endif // CIM_CONVERSION_PASSES

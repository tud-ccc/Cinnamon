//===- Passes.td - Cinm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_CONVERSION_PASSES
#define CINM_CONVERSION_PASSES


include "mlir/Pass/PassBase.td"

def ConvertTiledCinmToCnm : Pass<"convert-cinm-to-cnm", operation="func::FuncOp"> {
    let summary = "Convert CINM dialect to CNM dialect (may require --cinm-tiling before)";
    let constructor = "mlir::cinm::createConvertTiledCinmToCnmPass()";
    let dependentDialects = ["cnm::CnmDialect", "func::FuncDialect", 
        "arith::ArithDialect", "tensor::TensorDialect"];
}

#endif // CINM_CONVERSION_PASSES
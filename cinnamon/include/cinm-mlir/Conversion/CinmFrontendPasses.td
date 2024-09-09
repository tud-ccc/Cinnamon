//===- Passes.td - CinmFrontend passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm frontend conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_FRONTEND_CONVERSION_PASSES
#define CINM_FRONTEND_CONVERSION_PASSES


include "mlir/Pass/PassBase.td"

def ConvertTorchToCinm : Pass<"convert-torch-to-cinm", operation="func::FuncOp"> {
    let summary = "Convert torch dialect to cinm dialect";
    let constructor = "mlir::cinm_frontend::createConvertTorchToCinmPass()";
    let dependentDialects = ["torch::Torch::TorchDialect", "torch::TorchConversion::TorchConversionDialect", 
        "cinm::CinmDialect", "func::FuncDialect", "tensor::TensorDialect"];
}

#endif // CINM_FRONTEND_CONVERSION_PASSES
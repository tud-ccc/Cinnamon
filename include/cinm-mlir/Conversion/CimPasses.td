//===- Passes.td - Cim dialect passes ---------------------*- tablegen -*-===//

#ifndef CIM_CONVERSION_PASSES
#define CIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCimToMemristorPass : Pass<"convert-cim-to-memristor", operation="func::FuncOp"> {
    let summary = "Convert cim dialect to memristor dialect";
    let constructor = "mlir::cim::createConvertCimToMemristorPass()";
    let dependentDialects = ["cim::CimDialect", "memristor::MemristorDialect",
        "bufferization::BufferizationDialect", "memref::MemRefDialect", "tensor::TensorDialect"];
}

def ConvertCimToAlpinePass
    : Pass<"convert-cim-to-alpine", "func::FuncOp"> {
  let summary = "Replace eligible cim.op.gemv with Alpine streaming ops";
  let description = [{
    Rewrites cim.op.gemv (without `relower = true`) to the Alpine sequence:
      alpine.alloc_tile (once per crossbar)
      alpine.write_weights
      alpine.enqueue_vec
      alpine.process { count = 1 }
      alpine.dequeue_vec
    Other CIM ops are left untouched.
  }];

  let dependentDialects = [
    "mlir::alpine::AlpineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
    "mlir::cim::CimDialect"
  ];
  let constructor = "mlir::cim::createConvertCimToAlpinePass()";

}
#endif // CIM_CONVERSION_PASSES

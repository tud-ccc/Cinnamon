//===- Passes.td - Alpine dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Alpine dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef ALPINE_CONVERSION_PASSES
#define ALPINE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertAlpineToFunc
  : Pass<"convert-alpine-to-func", operation = "ModuleOp"> {
  let summary = "Convert alpine dialect to func dialect";
  let constructor = "mlir::alpine::createConvertAlpineToFuncPass()";
  let dependentDialects = [
    "mlir::alpine::AlpineDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}
#endif // ALPINE_CONVERSION_PASSES

//===----------------------------------------------------------------------===//
//===- Passes.td - ALPINE dialect passes -------------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef ALPINE_TRANSFORM_PASSES
#define ALPINE_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def AlpineHoistWriteWeightsPass
    : Pass<"alpine-hoist-write-weights", "mlir::func::FuncOp"> {
  let summary = "Hoist alpine.write_weights out of column loops when operands are invariant";
  let description = [{
    Detects alpine.write_weights ops nested under GEMV-style loop nests and
    hoists the weight upload together with the surrounding allocation / cast /
    quantize chain out of inner column loops whenever the operands do not
    depend on those loop induction variables. This reduces redundant crossbar
    writes when the left-hand tile is reused across multiple GEMV columns.
  }];

  let dependentDialects = [
    "mlir::alpine::AlpineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // ALPINE_TRANSFORM_PASSES

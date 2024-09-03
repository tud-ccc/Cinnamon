//===- Passes.td - Cim dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cim dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_TRANSFORM_PASSES
#define CIM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def CimScheduleAsapPass : Pass<"cim-schedule-asap", "func::FuncOp"> {
  let summary = "Insert barriers to enforce as soon as possible scheduling.";
  let description = [{}];
  let dependentDialects = [];
}

def CimScheduleAlapPass : Pass<"cim-schedule-alap", "func::FuncOp"> {
  let summary = "Insert barriers to enforce as late as possible scheduling.";
  let description = [{}];
  let dependentDialects = [];
}

#endif // CIM_TRANSFORM_PASSES

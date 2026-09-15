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



def CimMarkRelowerPass : Pass<"cim-mark-relower"> {
  let summary = "Mark selected CIM ops with `relower = true` for re-lowering";
  let description = [{
    Sets a BoolAttr named `relower` to true on CIM ops whose names match the
    list provided via --ops. Accepts short names (e.g. "add,gemm,gemv") or
    fully-qualified names (e.g. "cim.op.add").
  }];

  let dependentDialects = [
    "mlir::cim::CimDialect"
  ];

  let options = [
    Option<"opsOpt", "ops", "std::string",
           "", // default = empty (no marking)
           "Comma-separated CIM op names to mark (e.g. add,gemm,gemv or cim.op.add)">
  ];
}

def CimCleanupUnsupportedPass
  : Pass<"cim-cleanup-unsupported", "func::FuncOp"> {
  let summary = "Lower unsupported CIM ops (e.g., cim.op.add) and erase admin ops";
  let description = [{
    Rewrites `cim.op.add` (materialized via barrier+copy) back to a bufferized
    `linalg.add`, and removes CIM admin ops:
      • Pattern: memref.copy (cim.barrier (cim.op.add ...)), %dst
          → linalg.add ins(%lhs, %rhs) outs(%dst)
      • Erases: cim.release_device / cim.release_crossbar (always)
      • Erases: cim.acquire_* if dead
      • Erases: cim.barrier if dead
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}



#endif // CIM_TRANSFORM_PASSES

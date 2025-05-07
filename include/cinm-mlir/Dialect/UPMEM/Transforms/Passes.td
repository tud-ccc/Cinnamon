//===- Passes.td - UPMEM dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_TRANSFORM_PASSES
#define UPMEM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def UPMEMDedupKernelsPass: Pass<"upmem-dedup-kernels", "ModuleOp"> {
  let summary = "Deduplicate the DPU kernels within an upmem module";
  let description = [{
  }];
  let dependentDialects = [
    "::mlir::upmem::UPMEMDialect"
  ];
}

#endif // UPMEM_TRANSFORM_PASSES

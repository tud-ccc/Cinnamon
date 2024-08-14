//===- Passes.td - Hbmpim dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Hbmpim dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef HBMPIM_TRANSFORM_PASSES
#define HBMPIM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def HbmpimOutlineKernelPass: Pass<"hbmpim-outline-kernel", "ModuleOp"> {
  let summary = "Outlines a kernel to a function and replaces it with a function call";
  let description = [{
  }];
}

#endif // HBMPIM_TRANSFORM_PASSES

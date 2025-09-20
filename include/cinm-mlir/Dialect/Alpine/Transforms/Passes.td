//===----------------------------------------------------------------------===//
//===- Passes.td - ALPINE dialect passes -------------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef ALPINE_TRANSFORM_PASSES
#define ALPINE_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

// def AlpineLowerMVMToStreamingPass : Pass<"alpine-lower-mvm-to-streaming"> {
//   let summary = "Lower alpine.mvm into enqueue_vec → process(count=1) → dequeue_vec";
//   let description = [{
//     Rewrites each `alpine.mvm` op to the equivalent streaming sequence:
//     - `alpine.enqueue_vec %tile, %x`
//     - `alpine.process %tile { count = 1 }`
//     - `alpine.dequeue_vec %tile, %y`

//     This pass does not insert quantization or type conversions; it assumes the
//     device contract is already satisfied (i8 buffers).
//   }];

//   let dependentDialects = [
//     "mlir::alpine::AlpineDialect"
//   ];
// }

#endif // ALPINE_TRANSFORM_PASSES

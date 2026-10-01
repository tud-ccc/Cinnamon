// RUN: rm -rf %t && mkdir -p %t
// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast fixed-tasklets=4 graph-allocation=1 allocation-in=%S/Inputs/graph-loop-result allocation-report=%t" -o /dev/null
// RUN: FileCheck %s < %t/infer_looped.json

// A block after a loop consumes what the loop's body yields, not only the
// loop's initial value: its node depends on the in-loop node.

// CHECK:      "nodes": [
// CHECK:          "class": 0,
// CHECK:          "loops": [
// CHECK:          "predecessors": []
// CHECK:          "class": 1,
// CHECK:          "loops": [],
// CHECK-NEXT:     "member": 0,
// CHECK-NEXT:     "predecessors": [
// CHECK-NEXT:       0
// CHECK-NEXT:     ]

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

func.func @looped(%A: tensor<128x128xi32>, %B: tensor<64x128xi32>, %x: tensor<128xi32>) -> tensor<64xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %y = scf.for %i = %c0 to %c2 step %c1 iter_args(%v = %x) -> (tensor<128xi32>) {
    %r = cinm.compute -> tensor<128xi32> {
      %g = cinm.op.gemv %A, %v : tensor<128x128xi32>, tensor<128xi32> -> tensor<128xi32>
      cinm.yield %g : tensor<128xi32>
    }
    scf.yield %r : tensor<128xi32>
  }
  %out = cinm.compute -> tensor<64xi32> {
    %g = cinm.op.gemv %B, %y : tensor<64x128xi32>, tensor<128xi32> -> tensor<64xi32>
    cinm.yield %g : tensor<64xi32>
  }
  return %out : tensor<64xi32>
}

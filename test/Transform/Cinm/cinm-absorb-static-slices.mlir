// RUN: cinm-opt %s --split-input-file --cinm-absorb-static-slices --cinm-isolate-compute-blocks | FileCheck %s

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// The layer loop: the weight read inside the compute op is a slice of the
// static stack at the loop index. Absorbed, the isolated block takes the
// whole stack and the index, and slices inside.

// CHECK-LABEL: func.func @layers
//       CHECK:   scf.for %[[L:.*]] = %c0 to %c4
//       CHECK:     cinm.compute_block (%[[W:.*]] = %arg0 : tensor<4x256x256xi8>, %[[I:.*]] = %[[L]] : index, %[[X:.*]] = %{{.*}} : tensor<256xi8>)
//       CHECK:       %[[S:.*]] = tensor.extract_slice %[[W]][%[[I]], 0, 0] [1, 256, 256] [1, 1, 1]
//       CHECK:       cinm.op.gemv %[[S]], %[[X]]
func.func @layers(%W: tensor<4x256x256xi8> {cinm.static}, %x0: tensor<256xi8>) -> tensor<256xi8>
    attributes {cinm.available_platforms = [#upmem]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %l = %c0 to %c4 step %c1 iter_args(%x = %x0) -> (tensor<256xi8>) {
    %w = tensor.extract_slice %W[%l, 0, 0] [1, 256, 256] [1, 1, 1] : tensor<4x256x256xi8> to tensor<256x256xi8>
    %y = cinm.compute -> tensor<256xi32> {
      %g = cinm.op.gemv %w, %x : tensor<256x256xi8>, tensor<256xi8> -> tensor<256xi32>
      cinm.yield %g : tensor<256xi32>
    }
    %init = tensor.empty() : tensor<256xi8>
    %n = linalg.map ins(%y : tensor<256xi32>) outs(%init : tensor<256xi8>)
      (%v : i32, %o : i8) {
        %t = arith.trunci %v : i32 to i8
        linalg.yield %t : i8
      }
    scf.yield %n : tensor<256xi8>
  }
  return %r : tensor<256xi8>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// A slice at a constant offset is the unrolled case and stays outside, so
// that identical blocks keep identical signatures.

// CHECK-LABEL: func.func @unrolled
//       CHECK:   %[[S:.*]] = tensor.extract_slice %arg0[1, 0, 0]
//       CHECK:   cinm.compute_block (%{{.*}} = %[[S]] : tensor<256x256xi8>
func.func @unrolled(%W: tensor<4x256x256xi8> {cinm.static}, %x: tensor<256xi8>) -> tensor<256xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %w = tensor.extract_slice %W[1, 0, 0] [1, 256, 256] [1, 1, 1] : tensor<4x256x256xi8> to tensor<256x256xi8>
  %y = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %w, %x : tensor<256x256xi8>, tensor<256xi8> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %y : tensor<256xi32>
}

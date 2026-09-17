// RUN: cinm-opt %s --cinm-absorb-static-slices --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=4 n-init=4 fixed-tasklets=4 graph-allocation=1 allocation-granularity=4 latency-objective=1" | FileCheck %s

// A layer loop left rolled: one compute block, whose weight is the slice of
// the static stack at the loop index. The block is searched and allocated
// once, and its lowering keeps every layer's weight resident: the repack
// stages the slices side by side, each iteration scatters its slice into its
// own slot and broadcasts the slot index for the program to select it.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: func.func @layers
//       CHECK:   scf.for %[[L:.*]] = %c0 to %c4
//       CHECK:     cinm.compute_block on accelerator
//       CHECK:       upmem.broadcast %{{.*}} onto @slot of
//       CHECK:       %[[STACK:.*]] = memref.get_global @__cnm_repack_{{.*}} : memref<4x{{.*}}> {cinm.static}
//       CHECK:       %[[SLOT:.*]] = memref.subview %[[STACK]][%[[I:.*]], 0, 0, 0, 0, 0, 0] [1,
//       CHECK:       cnm.compact_buffer %{{.*}} into %[[SLOT]][#{{.*}}] {cinm.static}
//       CHECK:       upmem.scatter_on_array %[[SLOT]][{{.*}}] onto @[[BUF:buf_[0-9]+]] slot %[[I]] of
//       CHECK:   upmem.dpu_program
//       CHECK:     upmem.static_alloc @[[BUF]](mram) noinit slots 4 : memref<4x
//       CHECK:     upmem.static_alloc @slot(wram) noinit : memref<2xi32, #upmem.wram>
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

// RUN: cinm-opt %s --split-input-file --cinm-absorb-static-slices --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=4 n-init=4 fixed-tasklets=4 graph-allocation=1 screen-menu=0 allow-host-placement=0 allocation-granularity=4 latency-objective=1" | FileCheck %s
// Both halves of the placement decision are off here (screen-menu=0,
// allow-host-placement=0): these toy blocks are far too small to beat the
// host on any DPU count, and what this test pins is what the search, the
// allocation and the commit do with a block that is offloaded, not whether
// one should be.

// A layer loop left rolled: one compute block, whose weight is the slice of
// the static stack at the loop index. The block is searched and allocated
// once, and its lowering keeps every layer's weight resident: the repack
// stages the slices side by side, each iteration scatters its slice into its
// own slot, and the program selects the slot from a count of its launches --
// the loop is the only thing around the block, so launch n is layer n mod 4.
// Nothing about the slot crosses to the device per launch.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: func.func @layers
//       CHECK:   scf.for %[[L:.*]] = %c0 to %c4
//       CHECK:     cinm.compute_block on accelerator
//  CHECK-SAME:       sequence = {loops = [{lb = 0 : i64, operands = array<i64: 1>, step = 1 : i64, stride = 1 : i64, trip = 4 : i64}]}
//   CHECK-NOT:       onto @slot
//       CHECK:       %[[STACK:.*]] = memref.get_global @__cnm_repack_{{.*}} : memref<4x{{.*}}> {cinm.static}
//       CHECK:       %[[SLOT:.*]] = memref.subview %[[STACK]][%[[I:.*]], 0, 0, 0, 0, 0, 0] [1,
//       CHECK:       cnm.compact_buffer %{{.*}} into %[[SLOT]][#{{.*}}] {cinm.static}
//       CHECK:       upmem.scatter_on_array %[[SLOT]][{{.*}}] onto @[[BUF:buf_[0-9]+]] slot %[[I]] of
//       CHECK:   upmem.dpu_program
//       CHECK:     upmem.static_alloc @[[BUF]](mram) noinit slots 4 : memref<4x
//       CHECK:     %[[COUNT:.*]] = upmem.static_alloc @launch_count(wram) zeroinit : memref<4xi32, #upmem.wram>
//       CHECK:     %[[T:.*]] = upmem.tasklet_dim()
//       CHECK:     %[[N:.*]] = memref.load %[[COUNT]][%[[T]]]
//       CHECK:     %[[N1:.*]] = arith.addi %[[N]], %c1_i32
//       CHECK:     memref.store %[[N1]], %[[COUNT]][%[[T]]]
//       CHECK:     %[[NI:.*]] = arith.index_cast %[[N]] : i32 to index
//       CHECK:     arith.remui %[[NI]], %c4
//   CHECK-NOT:     @slot(wram)
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

// Two members per layer sharing one set (latency objective). A pass over the
// loop body launches both, in order, so launch n is member n mod 2 of layer
// n div 2, and each member's slot is its position times the 4 layers plus
// the layer: the host lands the second member's slice in slot 4 + l, and the
// program derives the same from its count.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: func.func @two_per_layer
//       CHECK:   scf.for %[[L:.*]] = %c0 to %c4
//       CHECK:     cinm.compute_block on accelerator
//  CHECK-SAME:       slot = 0 : i64, slots = 2 : i64
//   CHECK-NOT:       onto @slot
//       CHECK:       upmem.scatter_on_array %{{.*}} onto @[[BUF:buf_[0-9]+]] slot %[[I:[a-z0-9_]+]] of
//       CHECK:     cinm.compute_block on accelerator
//  CHECK-SAME:       slot = 1 : i64, slots = 2 : i64
//   CHECK-NOT:       onto @slot
//       CHECK:       %[[S1:.*]] = arith.addi %[[I]], %c4
//       CHECK:       upmem.scatter_on_array %{{.*}} onto @[[BUF]] slot %[[S1]] of
//       CHECK:   upmem.dpu_program
//       CHECK:     upmem.static_alloc @[[BUF]](mram) noinit slots 8 : memref<8x
//       CHECK:     upmem.static_alloc @launch_count(wram) zeroinit : memref<4xi32, #upmem.wram>
//       CHECK:     %[[NI:.*]] = arith.index_cast %{{.*}} : i32 to index
//   CHECK-DAG:     %[[P:.*]] = arith.remui %[[NI]], %c2
//   CHECK-DAG:     %[[Q:.*]] = arith.divui %[[NI]], %c2
//   CHECK-DAG:     %[[BASE:.*]] = arith.muli %[[P]], %c4
//   CHECK-DAG:     %[[LAYER:.*]] = arith.remui %[[Q]], %c4
//       CHECK:     arith.addi %[[BASE]], %[[LAYER]]
func.func @two_per_layer(%Wq: tensor<4x256x256xi8> {cinm.static}, %Wk: tensor<4x256x256xi8> {cinm.static}, %x0: tensor<256xi8>) -> tensor<256xi8>
    attributes {cinm.available_platforms = [#upmem]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %l = %c0 to %c4 step %c1 iter_args(%x = %x0) -> (tensor<256xi8>) {
    %wq = tensor.extract_slice %Wq[%l, 0, 0] [1, 256, 256] [1, 1, 1] : tensor<4x256x256xi8> to tensor<256x256xi8>
    %wk = tensor.extract_slice %Wk[%l, 0, 0] [1, 256, 256] [1, 1, 1] : tensor<4x256x256xi8> to tensor<256x256xi8>
    %q = cinm.compute -> tensor<256xi32> {
      %g = cinm.op.gemv %wq, %x : tensor<256x256xi8>, tensor<256xi8> -> tensor<256xi32>
      cinm.yield %g : tensor<256xi32>
    }
    %k = cinm.compute -> tensor<256xi32> {
      %g = cinm.op.gemv %wk, %x : tensor<256x256xi8>, tensor<256xi8> -> tensor<256xi32>
      cinm.yield %g : tensor<256xi32>
    }
    %init = tensor.empty() : tensor<256xi8>
    %n = linalg.map ins(%q, %k : tensor<256xi32>, tensor<256xi32>) outs(%init : tensor<256xi8>)
      (%a : i32, %b : i32, %o : i8) {
        %s = arith.addi %a, %b : i32
        %t = arith.trunci %s : i32 to i8
        linalg.yield %t : i8
      }
    scf.yield %n : tensor<256xi8>
  }
  return %r : tensor<256xi8>
}

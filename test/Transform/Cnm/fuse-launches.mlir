// RUN: cinm-opt %s --split-input-file --cnm-fuse-launches | FileCheck %s
// RUN: cinm-opt %s --split-input-file --cnm-fuse-launches=merge-launch-bodies=false | FileCheck %s --check-prefix=NOMERGE

// A gather immediately followed by a scatter with the same map into a buffer
// of the same type sends every leaf the block it just handed back. The pass
// points the consumer at the producer's buffer and drops both transfers.
//
// The shapes here are what --convert-linalg-to-cnm produces for `gemv` then
// elementwise on the 4MB prim_gemv case, with the gemv's K not split: M blocked
// 64 over a 16-leaf workgroup, so both ops tile the 1024-element result the
// same way. See docs/LaunchFusionDesign.md §A.

#map = affine_map<(d0, d1, d2, d3) -> (d1 * 64 + d3)>
#bcast = affine_map<(d0, d1, d2) -> ()>
#mat = affine_map<(d0, d1) -> (d0, d1)>
#vec = affine_map<(d0, d1) -> (d1)>
#res = affine_map<(d0, d1) -> (d0)>
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

#pf = #upmem.platform<type = v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// CHECK-LABEL: func.func @round_trip
// One workgroup survives, and one release of it.
// CHECK:       = cnm.workgroup
// CHECK-NOT:   = cnm.workgroup
// A single launch, with both bodies in it and the intermediate staying in MRAM.
// CHECK:       cnm.launch
// CHECK:         linalg.contract
// CHECK:         linalg.generic
// CHECK-NOT:   cnm.launch
// Only the final result is brought back, and nothing is sent in again.
// CHECK:       cnm.gather
// CHECK-NOT:   cnm.scatter
// CHECK:       cnm.free_workgroup
// CHECK-NOT:   cnm.free_workgroup

// Without the merge, the transfers still go but the two kernels stay.
// NOMERGE-LABEL: func.func @round_trip
// NOMERGE:       cnm.launch
// NOMERGE:         linalg.contract
// NOMERGE:       cnm.launch
// NOMERGE:         linalg.generic
func.func @round_trip(%A: tensor<1024x1024xi32>, %x: tensor<1024xi32>, %c: i32) -> tensor<1024xi32> {
  %wg0 = cnm.workgroup : !cnm.workgroup<#acc>
  %prod = cnm.declare_buffer() for %wg0 : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %vec = cnm.declare_buffer() for %wg0 : !cnm.buffer<1024xi32 on #acc, #upmem.mram>
  %mat = cnm.declare_buffer() for %wg0 : !cnm.buffer<64x1024xi32 on #acc, #upmem.mram>
  %wg1 = cnm.workgroup : !cnm.workgroup<#acc>
  %out = cnm.declare_buffer() for %wg1 : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %cst = cnm.declare_buffer() for %wg1 : !cnm.buffer<i32 on #acc, #upmem.mram>
  %in = cnm.declare_buffer() for %wg1 : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    cnm.scatter %A into %mat[#map] of %wg0 : tensor<1024x1024xi32> into !cnm.buffer<64x1024xi32 on #acc, #upmem.mram>
    cnm.scatter %x into %vec[#bcast] of %wg0 : tensor<1024xi32> into !cnm.buffer<1024xi32 on #acc, #upmem.mram>
    cnm.launch %wg0 ins(%a = %mat : <64x1024xi32, #upmem.mram>, %b = %vec : <1024xi32, #upmem.mram>) outs(%p = %prod : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
      linalg.contract indexing_maps = [#mat, #vec, #res] ins(%a, %b : memref<64x1024xi32, #upmem.mram>, memref<1024xi32, #upmem.mram>) outs(%p : memref<64xi32, #upmem.mram>)
    }
    %e0 = tensor.empty() : tensor<1024xi32>
    %g = cnm.gather %prod[#map] of %wg0 into %e0 : !cnm.buffer<64xi32 on #acc, #upmem.mram> into tensor<1024xi32>
    %s = tensor.from_elements %c : tensor<i32>
    cnm.scatter %g into %in[#map] of %wg1 : tensor<1024xi32> into !cnm.buffer<64xi32 on #acc, #upmem.mram>
    cnm.scatter %s into %cst[#bcast] of %wg1 : tensor<i32> into !cnm.buffer<i32 on #acc, #upmem.mram>
    cnm.launch %wg1 ins(%a = %in : <64xi32, #upmem.mram>, %b = %cst : <i32, #upmem.mram>) outs(%o = %out : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
      linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel"]} ins(%a, %b : memref<64xi32, #upmem.mram>, memref<i32, #upmem.mram>) outs(%o : memref<64xi32, #upmem.mram>) {
      ^bb0(%in_0: i32, %in_1: i32, %o_0: i32):
        %m = arith.muli %in_0, %in_1 : i32
        linalg.yield %m : i32
      }
    }
    %e1 = tensor.empty() : tensor<1024xi32>
    %res = cnm.gather %out[#map] of %wg1 into %e1 : !cnm.buffer<64xi32 on #acc, #upmem.mram> into tensor<1024xi32>
    cinm.yield %res : tensor<1024xi32>
  }
  cnm.free_workgroup %wg1 : !cnm.workgroup<#acc>
  cnm.free_workgroup %wg0 : !cnm.workgroup<#acc>
  func.return %r : tensor<1024xi32>
}

// -----

// The producer split its reduction across the workgroup, so it gathers partial
// results into a taller host tensor and merges them there. Neither the map nor
// the buffer shape matches what the consumer scatters, and the merge sits
// between the two -- nothing fires. This is the case the fusion level exists to
// choose *against*.

#partial = affine_map<(d0, d1, d2, d3, d4) -> (d1 floordiv 4, d1 mod 4 * 256 + d4)>
#whole = affine_map<(d0, d1, d2, d3) -> (d1 * 64 + d3)>
#bcast = affine_map<(d0, d1, d2) -> ()>
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
#merge_in = affine_map<(d0, d1) -> (d0, d1)>
#merge_out = affine_map<(d0, d1) -> (d1)>

#pf = #upmem.platform<type = v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// CHECK-LABEL: func.func @split_reduction_is_left_alone
// CHECK:       = cnm.workgroup
// CHECK:       = cnm.workgroup
// CHECK:       cnm.gather
// CHECK:       linalg.generic
// CHECK:       cnm.scatter
// CHECK:       cnm.launch
func.func @split_reduction_is_left_alone(%partials: tensor<4x1024xi32>, %init: tensor<1024xi32>, %c: i32) -> tensor<1024xi32> {
  %wg0 = cnm.workgroup : !cnm.workgroup<#acc>
  %prod = cnm.declare_buffer() for %wg0 : !cnm.buffer<1x256xi32 on #acc, #upmem.mram>
  %wg1 = cnm.workgroup : !cnm.workgroup<#acc>
  %out = cnm.declare_buffer() for %wg1 : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %cst = cnm.declare_buffer() for %wg1 : !cnm.buffer<i32 on #acc, #upmem.mram>
  %in = cnm.declare_buffer() for %wg1 : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %e0 = tensor.empty() : tensor<4x1024xi32>
    %g = cnm.gather %prod[#partial] of %wg0 into %e0 : !cnm.buffer<1x256xi32 on #acc, #upmem.mram> into tensor<4x1024xi32>
    // The host merge of the partials, which is what makes the gathered value
    // something no leaf holds.
    %merged = linalg.generic {indexing_maps = [#merge_in, #merge_out], iterator_types = ["reduction", "parallel"]} ins(%g : tensor<4x1024xi32>) outs(%init : tensor<1024xi32>) {
    ^bb0(%p: i32, %acc: i32):
      %a = arith.addi %p, %acc : i32
      linalg.yield %a : i32
    } -> tensor<1024xi32>
    %s = tensor.from_elements %c : tensor<i32>
    cnm.scatter %merged into %in[#whole] of %wg1 : tensor<1024xi32> into !cnm.buffer<64xi32 on #acc, #upmem.mram>
    cnm.scatter %s into %cst[#bcast] of %wg1 : tensor<i32> into !cnm.buffer<i32 on #acc, #upmem.mram>
    cnm.launch %wg1 ins(%a = %in : <64xi32, #upmem.mram>, %b = %cst : <i32, #upmem.mram>) outs(%o = %out : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
      linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel"]} ins(%a, %b : memref<64xi32, #upmem.mram>, memref<i32, #upmem.mram>) outs(%o : memref<64xi32, #upmem.mram>) {
      ^bb0(%in_0: i32, %in_1: i32, %o_0: i32):
        %m = arith.muli %in_0, %in_1 : i32
        linalg.yield %m : i32
      }
    }
    %e1 = tensor.empty() : tensor<1024xi32>
    %res = cnm.gather %out[#whole] of %wg1 into %e1 : !cnm.buffer<64xi32 on #acc, #upmem.mram> into tensor<1024xi32>
    cinm.yield %res : tensor<1024xi32>
  }
  cnm.free_workgroup %wg1 : !cnm.workgroup<#acc>
  cnm.free_workgroup %wg0 : !cnm.workgroup<#acc>
  func.return %r : tensor<1024xi32>
}

// -----

// Same tiling on both sides, but the two ops disagree about which leaf gets
// which block -- the consumer's map is the producer's reversed. Fusing would
// hand each leaf somebody else's data.

#fwd = affine_map<(d0, d1, d2, d3) -> (d1 * 64 + d3)>
#rev = affine_map<(d0, d1, d2, d3) -> (960 - d1 * 64 + d3)>
#id = affine_map<(d0) -> (d0)>

#pf = #upmem.platform<type = v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// CHECK-LABEL: func.func @map_mismatch
// CHECK:       cnm.gather
// CHECK:       cnm.scatter
// CHECK:       cnm.launch
func.func @map_mismatch(%init: tensor<1024xi32>) -> tensor<1024xi32> {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %prod = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %out = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %g = cnm.gather %prod[#fwd] of %wg into %init : !cnm.buffer<64xi32 on #acc, #upmem.mram> into tensor<1024xi32>
  cnm.scatter %g into %in[#rev] of %wg : tensor<1024xi32> into !cnm.buffer<64xi32 on #acc, #upmem.mram>
  cnm.launch %wg ins(%a = %in : <64xi32, #upmem.mram>) outs(%o = %out : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : memref<64xi32, #upmem.mram>) outs(%o : memref<64xi32, #upmem.mram>) {
    ^bb0(%in_0: i32, %o_0: i32):
      linalg.yield %in_0 : i32
    }
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  func.return %g : tensor<1024xi32>
}

// -----

// The gathered value is wanted on the host as well, so the gather stays -- only
// the transfer back in was certainly redundant.

#map = affine_map<(d0, d1, d2, d3) -> (d1 * 64 + d3)>
#id = affine_map<(d0) -> (d0)>

#pf = #upmem.platform<type = v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// CHECK-LABEL: func.func @gather_still_used
// CHECK:       %[[G:.*]] = cnm.gather
// CHECK-NOT:   cnm.scatter
// CHECK:       cnm.launch
// CHECK:       return %[[G]]
func.func @gather_still_used(%init: tensor<1024xi32>) -> tensor<1024xi32> {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %prod = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %out = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  %g = cnm.gather %prod[#map] of %wg into %init : !cnm.buffer<64xi32 on #acc, #upmem.mram> into tensor<1024xi32>
  cnm.scatter %g into %in[#map] of %wg : tensor<1024xi32> into !cnm.buffer<64xi32 on #acc, #upmem.mram>
  cnm.launch %wg ins(%a = %in : <64xi32, #upmem.mram>) outs(%o = %out : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : memref<64xi32, #upmem.mram>) outs(%o : memref<64xi32, #upmem.mram>) {
    ^bb0(%in_0: i32, %o_0: i32):
      linalg.yield %in_0 : i32
    }
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  func.return %g : tensor<1024xi32>
}

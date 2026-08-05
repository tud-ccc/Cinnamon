// RUN: cinm-opt %s --upmem-tile-mram-buffers --canonicalize --cse --split-input-file | FileCheck %s

#acc = #upmem.array<1x2048x8, <type = v1A, dimensions = 32x64x24>>

// A launch parameter that a linalg.fill sets to a constant -- what
// --cnm-scatter-optimizations leaves behind for a uniform scatter. Staging it
// would load MRAM that the fill is about to overwrite, so the staging buffer
// is filled directly instead and the MRAM fill goes away. Only the write-back
// remains, since the buffer is still gathered.

// CHECK-LABEL: func.func @fold_device_init
// CHECK:       %[[WY:.*]] = memref.alloca() : memref<1x8xi32, #upmem.wram>
// CHECK:       linalg.fill ins(%{{.*}} : i32) outs(%[[WY]] : memref<1x8xi32, #upmem.wram>)
// CHECK-NOT:   cnm.local_transfer %{{.*}} into %[[WY]]
// CHECK:       affine.for
// CHECK:         linalg.generic {{.*}} outs(%[[WY]]
// CHECK:       cnm.local_transfer %[[WY]] into %{{.*}} : memref<1x8xi32, #upmem.wram> to memref<1x8xi32, #upmem.mram>
// CHECK-NOT:   linalg.fill {{.*}}#upmem.mram
func.func @fold_device_init(%A: memref<4096x32x128xi32>, %x: memref<32x128xi32>,
                            %out: memref<32x4096xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %ba = cnm.declare_buffer() for %wg : !cnm.buffer<8x1x128xi32 on #acc, #upmem.mram>
  cnm.scatter %A into %ba[affine_map<(d0, d1, d2, d3, d4, d5) -> (d1 * 64 + d2 * 8 + d3 - (d1 floordiv 64) * 4096, d1 floordiv 64, d5)>] of %wg : memref<4096x32x128xi32> into !cnm.buffer<8x1x128xi32 on #acc, #upmem.mram>
  %bx = cnm.declare_buffer() for %wg : !cnm.buffer<1x128xi32 on #acc, #upmem.mram>
  cnm.scatter %x into %bx[affine_map<(d0, d1, d2, d3, d4) -> (d1 floordiv 64, d4)>] of %wg : memref<32x128xi32> into !cnm.buffer<1x128xi32 on #acc, #upmem.mram>
  %by = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #acc, #upmem.mram>
  cnm.launch %wg ins(%a = %ba : <8x1x128xi32, #upmem.mram>, %xx = %bx : <1x128xi32, #upmem.mram>) outs(%y = %by : <1x8xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    %z = arith.constant 0 : i32
    linalg.fill ins(%z : i32) outs(%y : memref<1x8xi32, #upmem.mram>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %xx : memref<8x1x128xi32, #upmem.mram>, memref<1x128xi32, #upmem.mram>) outs(%y : memref<1x8xi32, #upmem.mram>) attrs = {upmem.leaf_tile_sizes = array<i64: 1, 8, 64>} {
    ^bb0(%in: i32, %in_1: i32, %o: i32):
      %m = arith.muli %in, %in_1 : i32
      %s = arith.addi %o, %m : i32
      linalg.yield %s : i32
    }
  }
  cnm.gather %by[affine_map<(d0, d1, d2, d3, d4) -> (d1 floordiv 64, d1 * 64 + d2 * 8 + d4 - (d1 floordiv 64) * 4096)>] of %wg into %out : !cnm.buffer<1x8xi32 on #acc, #upmem.mram> into memref<32x4096xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}

// -----

#acc = #upmem.array<1x2048x8, <type = v1A, dimensions = 32x64x24>>

// Two ops consume the filled parameter, so the second one reads what the first
// wrote back rather than the constant. Folding its staging copy into a fill
// would hand it the constant instead, so nothing is folded: the fill keeps its
// own staging buffer and copies in like any other op. Slower than the fold,
// but the same program.

// CHECK-LABEL: func.func @two_consumers
// CHECK:       %[[WF:.*]] = memref.alloca() : memref<1x8xi32, #upmem.wram>
// CHECK:       cnm.local_transfer %{{.*}} into %[[WF]] : memref<1x8xi32, #upmem.mram> to memref<1x8xi32, #upmem.wram>
// CHECK:       linalg.fill ins(%{{.*}} : i32) outs(%[[WF]] : memref<1x8xi32, #upmem.wram>)
func.func @two_consumers(%in: memref<32x8xi32>, %out: memref<32x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %bi = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #acc, #upmem.mram>
  cnm.scatter %in into %bi[affine_map<(d0, d1, d2, d3, d4) -> (d1 floordiv 64, d4)>] of %wg : memref<32x8xi32> into !cnm.buffer<1x8xi32 on #acc, #upmem.mram>
  %by = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #acc, #upmem.mram>
  cnm.launch %wg ins(%a = %bi : <1x8xi32, #upmem.mram>) outs(%y = %by : <1x8xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    %z = arith.constant 0 : i32
    linalg.fill ins(%z : i32) outs(%y : memref<1x8xi32, #upmem.mram>)
    linalg.add ins(%a, %y : memref<1x8xi32, #upmem.mram>, memref<1x8xi32, #upmem.mram>) outs(%y : memref<1x8xi32, #upmem.mram>)
    linalg.add ins(%a, %y : memref<1x8xi32, #upmem.mram>, memref<1x8xi32, #upmem.mram>) outs(%y : memref<1x8xi32, #upmem.mram>)
  }
  cnm.gather %by[affine_map<(d0, d1, d2, d3, d4) -> (d1 floordiv 64, d4)>] of %wg into %out : !cnm.buffer<1x8xi32 on #acc, #upmem.mram> into memref<32x8xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}

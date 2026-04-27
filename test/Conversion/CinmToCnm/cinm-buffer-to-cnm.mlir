// RUN: cinm-opt --cinm-isolate-compute-blocks --convert-cinm-to-cnm %s | cinm-opt | FileCheck %s

// Affine maps emitted for scatter/gather index computations.
// CHECK-DAG: #[[MAP_ROW_MOD8:[^ ]*]] = affine_map<(d0, d1, d2) -> (d1 mod 8)>
// CHECK-DAG: #[[MAP_ROW_SUM:[^ ]*]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-DAG: #[[MAP_ROW_COL:[^ ]*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP_ID:[^ ]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[MAP_SCALAR:[^ ]*]] = affine_map<(d0) -> ()>
// CHECK-DAG: #[[MAP_FLAT:[^ ]*]] = affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2)>
// CHECK-DAG: #[[MAP_BCAST:[^ ]*]] = affine_map<(d0, d1, d2) -> ()>

// CHECK-LABEL: func.func @mm_dimm8_nopt
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8x128xi32>
// CHECK: linalg.fill ins(%[[C0]] : i32) outs(%[[ALLOC]] : memref<8x128xi32>)
// CHECK: %[[WG:.*]] = cnm.workgroup : !cnm.workgroup<8x128x1>
// CHECK: %[[ALLOC_T:.*]] = memref.alloc() : memref<128x1024xi32>
// CHECK: linalg.transpose ins(%arg1 : memref<1024x128xi32>) outs(%[[ALLOC_T]] : memref<128x1024xi32>) permutation = [1, 0]
// CHECK: %[[BUF_A:.*]] = cnm.alloc() for %[[WG]] : !cnm.buffer<1024xi32 on 8x128x1>
// CHECK: %[[BUF_B:.*]] = cnm.alloc() for %[[WG]] : !cnm.buffer<1024xi32 on 8x128x1>
// CHECK: %[[BUF_C:.*]] = cnm.alloc() for %[[WG]] : !cnm.buffer<i32 on 8x128x1>
// CHECK: cnm.scatter %arg0 into %[[BUF_A]][#[[MAP_ROW_MOD8]]] of %[[WG]] : memref<8x1024xi32> into !cnm.buffer<1024xi32 on 8x128x1>
// CHECK: cnm.scatter %[[ALLOC_T]] into %[[BUF_B]][#[[MAP_ROW_SUM]]] of %[[WG]] : memref<128x1024xi32> into !cnm.buffer<1024xi32 on 8x128x1>
// CHECK: cnm.scatter %[[ALLOC]] into %[[BUF_C]][#[[MAP_ROW_COL]]] of %[[WG]] : memref<8x128xi32> into !cnm.buffer<i32 on 8x128x1>
// CHECK: cnm.launch %[[WG]] in(%[[BUF_A]], %[[BUF_B]] : !cnm.buffer<1024xi32 on 8x128x1>, !cnm.buffer<1024xi32 on 8x128x1>) out(%[[BUF_C]] : !cnm.buffer<i32 on 8x128x1>) on !cnm.workgroup<8x128x1> {
// CHECK: ^bb0(%[[K_A:.*]]: memref<1024xi32>, %[[K_B:.*]]: memref<1024xi32>, %[[K_C:.*]]: memref<i32>):
// CHECK: linalg.contract indexing_maps = [#[[MAP_ID]], #[[MAP_ID]], #[[MAP_SCALAR]]] ins(%[[K_A]], %[[K_B]] : memref<1024xi32>, memref<1024xi32>) outs(%[[K_C]] : memref<i32>)
// CHECK: }
// CHECK: cnm.gather %[[BUF_C]][#[[MAP_ROW_COL]]] of %[[WG]] into %[[ALLOC]] : !cnm.buffer<i32 on 8x128x1> into memref<8x128xi32>
// CHECK: cnm.free_workgroup %[[WG]] : !cnm.workgroup<8x128x1>
// CHECK: return %[[ALLOC]] : memref<8x128xi32>

func.func @mm_dimm8_nopt(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>) -> memref<8x128xi32> {
  %0 = cinm.compute_ -> memref<8x128xi32> attributes {workgroupShape = array<i64: 8, 128, 1>} {
    %alloc = memref.alloc() : memref<8x128xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x128xi32>)
    cinm.op.gemm %arg0, %arg1 into %alloc : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
    cinm.yield %alloc : memref<8x128xi32>
  }
  return %0 : memref<8x128xi32>
}


// CHECK-LABEL: func.func @gemv
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xi32>
// CHECK: linalg.fill ins(%[[C0]] : i32) outs(%[[ALLOC]] : memref<8xi32>)
// CHECK: %[[WG:.*]] = cnm.workgroup : !cnm.workgroup<2x4x1>
// CHECK: %[[BUF_A:.*]] = cnm.alloc() for %[[WG]] : !cnm.buffer<1024xi32 on 2x4x1>
// CHECK: cnm.scatter %arg0 into %[[BUF_A]][#[[MAP_FLAT]]] of %[[WG]] : memref<8x1024xi32> into !cnm.buffer<1024xi32 on 2x4x1>
// CHECK: %[[BUF_B:.*]] = cnm.alloc() for %[[WG]] : !cnm.buffer<1024xi32 on 2x4x1>
// CHECK: cnm.scatter %arg1 into %[[BUF_B]][#[[MAP_BCAST]]] of %[[WG]] : memref<1024xi32> into !cnm.buffer<1024xi32 on 2x4x1>
// CHECK: %[[BUF_C:.*]] = cnm.alloc() for %[[WG]] : !cnm.buffer<i32 on 2x4x1>
// CHECK: cnm.scatter %[[ALLOC]] into %[[BUF_C]][#[[MAP_FLAT]]] of %[[WG]] : memref<8xi32> into !cnm.buffer<i32 on 2x4x1>
// CHECK: cnm.launch %[[WG]] in(%[[BUF_A]], %[[BUF_B]] : !cnm.buffer<1024xi32 on 2x4x1>, !cnm.buffer<1024xi32 on 2x4x1>) out(%[[BUF_C]] : !cnm.buffer<i32 on 2x4x1>) on !cnm.workgroup<2x4x1> {
// CHECK: ^bb0(%[[K_A:.*]]: memref<1024xi32>, %[[K_B:.*]]: memref<1024xi32>, %[[K_C:.*]]: memref<i32>):
// CHECK: linalg.contract indexing_maps = [#[[MAP_ID]], #[[MAP_ID]], #[[MAP_SCALAR]]] ins(%[[K_A]], %[[K_B]] : memref<1024xi32>, memref<1024xi32>) outs(%[[K_C]] : memref<i32>)
// CHECK: }
// CHECK: cnm.gather %[[BUF_C]][#[[MAP_FLAT]]] of %[[WG]] into %[[ALLOC]] : !cnm.buffer<i32 on 2x4x1> into memref<8xi32>
// CHECK: cnm.free_workgroup %[[WG]] : !cnm.workgroup<2x4x1>
// CHECK: return %[[ALLOC]] : memref<8xi32>

func.func @gemv(%arg0: memref<8x1024xi32>, %arg1: memref<1024xi32>) -> memref<8xi32> {
  %0 = cinm.compute_ -> memref<8xi32> attributes {tileSizes = array<i64: 1, 8>, workgroupShape = array<i64: 2, 4, 1>} {
    %alloc = memref.alloc() : memref<8xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8xi32>)
    cinm.op.gemv %arg0, %arg1 into %alloc : memref<8x1024xi32>, memref<1024xi32> into memref<8xi32>
    cinm.yield %alloc : memref<8xi32>
  }
  return %0 : memref<8xi32>
}


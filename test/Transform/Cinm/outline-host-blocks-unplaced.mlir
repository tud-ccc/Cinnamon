// RUN: cinm-opt %s --cinm-outline-host-blocks="unplaced=true manifest-file=%t.json" | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json

// On a lowered module, every compute block without an accelerator is host
// code, whatever platforms it was offered. Its boundary is made C-callable:
// the row view at a dynamic offset is passed as the view at offset 0 of its
// buffer plus the offset, and the result, written in place into %out, is
// dropped in favour of %out.

// CHECK-LABEL: func.func @row_sum
// CHECK:         cinm.compute_block (%[[ROW:.*]] = %{{.*}} : memref<8xi32, strided<[1], offset: ?>>, %[[OUT:.*]] = %{{.*}} : memref<8xi32>)
// CHECK:           %[[BASE:.*]], %[[OFF:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[ROW]]
// CHECK:           %[[AT0:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [0], sizes: [8], strides: [1]
// CHECK:           func.call @row_sum_host0(%[[AT0]], %[[OFF]], %[[OUT]]) : (memref<8xi32, strided<[1]>>, index, memref<8xi32>) -> ()
// CHECK:           cinm.yield %[[OUT]]
// A row updated in place through its dynamic-offset view: the result is
// the caller's view itself.

// CHECK-LABEL: func.func @row_scale
// CHECK:         %[[ROW:.*]] = memref.subview
// CHECK:         %[[R:.*]] = cinm.compute_block
// CHECK:           func.call @row_scale_host0(%{{.*}}, %{{.*}}) : (memref<8xi32, strided<[1]>>, index) -> ()
// CHECK:           cinm.yield %{{.*}} : memref<8xi32, strided<[1], offset: ?>>
// CHECK:         return %[[R]]

// CHECK:       module @outlined
// CHECK:         func.func @row_sum_host0(%[[A:.*]]: memref<8xi32, strided<[1]>>, %[[O:.*]]: index, %[[B:.*]]: memref<8xi32>) {
// CHECK:           %[[V:.*]] = memref.reinterpret_cast %[[A]] to offset: [%[[O]]], sizes: [8], strides: [1]
// CHECK:           linalg.add ins(%[[V]], %[[B]] : {{.*}}) outs(%[[B]] : memref<8xi32>)
// CHECK-NEXT:      return
// CHECK:       func.func private @row_sum_host0(memref<8xi32, strided<[1]>>, index, memref<8xi32>)

// MANIFEST:      "class": 3
// MANIFEST:      "name": "row_sum_host0"
// MANIFEST:      "params": [
// MANIFEST:          "kind": "memref"
// MANIFEST:          "offset": "next"
// MANIFEST:          "kind": "offset"
// MANIFEST:          "kind": "memref"
// MANIFEST:          "offset": 0
// MANIFEST:      "results": [
// MANIFEST:          "in_place": 1
// MANIFEST:          "kind": "memref"
// MANIFEST:      "name": "row_scale_host0"
// MANIFEST:      "results": [
// MANIFEST:          "in_place": 0

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @row_sum(%m: memref<4x8xi32>, %i: index, %out: memref<8xi32>) -> memref<8xi32> {
  %row = memref.subview %m[%i, 0] [1, 8] [1, 1] : memref<4x8xi32> to memref<8xi32, strided<[1], offset: ?>>
  %r = cinm.compute_block (%a = %row : memref<8xi32, strided<[1], offset: ?>>, %b = %out : memref<8xi32>) -> memref<8xi32> attributes {cinm.available_platforms = [#upmem], cinm.graph_alloc = {class = 3 : i64, graph = "infer_row_sum", member = 0 : i64, placement = "host"}} {
    linalg.add ins(%a, %b : memref<8xi32, strided<[1], offset: ?>>, memref<8xi32>) outs(%b : memref<8xi32>)
    cinm.yield %b : memref<8xi32>
  }
  return %r : memref<8xi32>
}

func.func @row_scale(%m: memref<4x8xi32>, %i: index) -> memref<8xi32, strided<[1], offset: ?>> {
  %row = memref.subview %m[%i, 0] [1, 8] [1, 1] : memref<4x8xi32> to memref<8xi32, strided<[1], offset: ?>>
  %r = cinm.compute_block (%a = %row : memref<8xi32, strided<[1], offset: ?>>) -> memref<8xi32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
    linalg.add ins(%a, %a : memref<8xi32, strided<[1], offset: ?>>, memref<8xi32, strided<[1], offset: ?>>) outs(%a : memref<8xi32, strided<[1], offset: ?>>)
    cinm.yield %a : memref<8xi32, strided<[1], offset: ?>>
  }
  return %r : memref<8xi32, strided<[1], offset: ?>>
}

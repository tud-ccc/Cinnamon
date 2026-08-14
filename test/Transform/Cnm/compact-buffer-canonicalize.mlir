// RUN: cinm-opt %s --canonicalize | FileCheck %s

// Two repacks of the same data are one. Each is decided by a pass that cannot
// see the other -- one packs a strided host layout, the next reorders for the
// leaves -- so the chain is assembled rather than written.
//
// The merged map names, for each element of the final buffer, the element of
// the original it comes from. The `cinm.static` tag comes from the first: the
// merged op reads its source, and the second could not have carried the tag
// anyway, its source having been a fresh allocation.

// CHECK-DAG: #[[MERGED:.*]] = affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)>
// CHECK-LABEL: func.func @chain
// CHECK-NOT:   memref.alloc
// CHECK:       cnm.compact_buffer %arg0 into %arg1[#[[MERGED]]] {cinm.static} : memref<4x8xi32, strided<[16, 1]>> into memref<2x2x8xi32>
// CHECK-NOT:   cnm.compact_buffer
func.func @chain(%a: memref<4x8xi32, strided<[16, 1]>>, %out: memref<2x2x8xi32>) {
  %mid = memref.alloc() : memref<4x8xi32>
  cnm.compact_buffer %a into %mid [affine_map<(i, j) -> (i, j)>] {cinm.static}
      : memref<4x8xi32, strided<[16, 1]>> into memref<4x8xi32>
  cnm.compact_buffer %mid into %out [affine_map<(w, i, j) -> (w * 2 + i, j)>]
      : memref<4x8xi32> into memref<2x2x8xi32>
  return
}

// A cast is absorbed so the repack keeps the most static type: the backend
// sizes the copy from it.
// CHECK-LABEL: func.func @absorb_cast
// CHECK-NOT:   memref.cast
// CHECK:       cnm.compact_buffer %arg0 into %arg1[#{{.*}}] : memref<4x8xi32> into memref<4x8xi32>
func.func @absorb_cast(%a: memref<4x8xi32>, %out: memref<4x8xi32>) {
  %c = memref.cast %a : memref<4x8xi32> to memref<?x?xi32>
  cnm.compact_buffer %c into %out [affine_map<(i, j) -> (i, j)>]
      : memref<?x?xi32> into memref<4x8xi32>
  return
}

// The intermediate is read by something else, so the write to it has to stay.
// CHECK-LABEL: func.func @observed_intermediate
// CHECK:       cnm.compact_buffer %arg0 into %[[MID:.*]][#{{.*}}]
// CHECK:       cnm.compact_buffer %[[MID]] into %arg1[#{{.*}}]
func.func @observed_intermediate(%a: memref<4x8xi32, strided<[16, 1]>>,
                                 %out: memref<2x2x8xi32>) -> memref<4x8xi32> {
  %mid = memref.alloc() : memref<4x8xi32>
  cnm.compact_buffer %a into %mid [affine_map<(i, j) -> (i, j)>]
      : memref<4x8xi32, strided<[16, 1]>> into memref<4x8xi32>
  cnm.compact_buffer %mid into %out [affine_map<(w, i, j) -> (w * 2 + i, j)>]
      : memref<4x8xi32> into memref<2x2x8xi32>
  return %mid : memref<4x8xi32>
}

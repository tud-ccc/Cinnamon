// RUN: cinm-opt %s --split-input-file --cnm-promote-host-allocs | FileCheck %s

// A scratch buffer written and read in place, through a view, becomes one
// global per site; its dealloc goes with it.
// CHECK:       memref.global "private" @__cnm_scratch_0 : memref<8x16xi32> {alignment = 64 : i64}
// CHECK-LABEL: func.func @scratch
//       CHECK:   %[[G:.*]] = memref.get_global @__cnm_scratch_0 : memref<8x16xi32>
//       CHECK:   memref.expand_shape %[[G]]
//   CHECK-NOT:   memref.alloc
//   CHECK-NOT:   memref.dealloc
func.func @scratch(%out: memref<16xi32>) {
  %c0 = arith.constant 0 : index
  %buf = memref.alloc() {alignment = 64 : i64} : memref<8x16xi32>
  %v = memref.expand_shape %buf [[0], [1, 2]] output_shape [8, 4, 4] : memref<8x16xi32> into memref<8x4x4xi32>
  %x = memref.load %v[%c0, %c0, %c0] : memref<8x4x4xi32>
  memref.store %x, %out[%c0] : memref<16xi32>
  memref.dealloc %buf : memref<8x16xi32>
  return
}

// -----

// Returned, directly or as a view: it outlives the call.
// CHECK-LABEL: func.func @returned
//       CHECK:   memref.alloc
//       CHECK:   memref.alloc
//   CHECK-NOT:   memref.get_global
func.func @returned() -> (memref<16xi32>, memref<4x4xi32>) {
  %a = memref.alloc() : memref<16xi32>
  %b = memref.alloc() : memref<16xi32>
  %v = memref.expand_shape %b [[0, 1]] output_shape [4, 4] : memref<16xi32> into memref<4x4xi32>
  return %a, %v : memref<16xi32>, memref<4x4xi32>
}

// -----

// Passed to a call, which could keep it.
// CHECK-LABEL: func.func @called
//       CHECK:   memref.alloc
func.func private @keep(memref<16xi32>)
func.func @called() {
  %a = memref.alloc() : memref<16xi32>
  func.call @keep(%a) : (memref<16xi32>) -> ()
  return
}

// -----

// Inside a loop, or of dynamic shape: left alone.
// CHECK-LABEL: func.func @nested_or_dynamic
//       CHECK:   memref.alloc(%{{.*}}) : memref<?xi32>
//       CHECK:   scf.for
//       CHECK:     memref.alloc() : memref<16xi32>
//   CHECK-NOT:   memref.get_global
func.func @nested_or_dynamic(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %i0 = arith.constant 0 : i32
  %d = memref.alloc(%n) : memref<?xi32>
  memref.store %i0, %d[%c0] : memref<?xi32>
  scf.for %i = %c0 to %c4 step %c1 {
    %a = memref.alloc() : memref<16xi32>
    memref.store %i0, %a[%c0] : memref<16xi32>
  }
  return
}

// RUN: cinm-opt %s --cinm-tiling | FileCheck %s

// Tiling a gemm-like op whose operands are already memrefs (destination
// passing style). The output subview is hoisted out of the reduction loop and
// accumulated into in place, so no accumulator tensor and no insert_slice are
// needed. This is the shape --convert-cinm-to-cnm relies on when it puts cinm
// ops inside a cnm.launch body over the launch's memref block arguments.

// CHECK-LABEL: @gemv_memref
func.func @gemv_memref(%A: memref<64x128xi32>, %x: memref<128xi32>, %y: memref<64xi32>) {
  // CHECK: affine.for %[[I:.*]] = 0 to 64 step 16 {
  // The output tile lives outside the reduction loop and is written in place.
  // CHECK: %[[YT:.*]] = memref.subview %{{.*}}[%[[I]]] [16] [1] : memref<64xi32>
  // CHECK: affine.for %[[K:.*]] = 0 to 128 step 32 {
  // CHECK: %[[AT:.*]] = memref.subview %{{.*}}[%[[I]], %[[K]]] [16, 32] [1, 1]
  // CHECK: %[[XT:.*]] = memref.subview %{{.*}}[%[[K]]] [32] [1]
  // CHECK: cinm.op.gemv %[[AT]], %[[XT]] into %[[YT]]
  // CHECK-NOT: tensor.
  cinm.op.gemv %A, %x into %y { cinm.tile_sizes = array<i64: 16, 32> }
    : memref<64x128xi32>, memref<128xi32> into memref<64xi32>
  return
}

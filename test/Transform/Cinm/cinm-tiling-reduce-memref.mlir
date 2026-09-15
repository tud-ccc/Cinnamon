// RUN: cinm-opt %s --cinm-tiling | FileCheck %s

// In memref (destination-passing) mode cinm.op.reduce accumulates into %out,
// so tiling needs no accumulator carried through the loop nest: each tile
// accumulates into the matching tile of %out. This is the form the op takes
// inside a cnm.launch body, over the launch's memref block arguments.

// CHECK-LABEL: @reduce_last_dim
func.func @reduce_last_dim(%a: memref<64x128xi32>, %o: memref<64xi32>) {
  // CHECK-NOT: iter_args
  // CHECK: affine.for %[[I:.*]] = 0 to 64 step 16 {
  // CHECK: affine.for %[[K:.*]] = 0 to 128 step 32 {
  // CHECK: %[[AT:.*]] = memref.subview %{{.*}}[%[[I]], %[[K]]] [16, 32] [1, 1]
  // CHECK: %[[OT:.*]] = memref.subview %{{.*}}[%[[I]]] [16] [1] : memref<64xi32>
  // CHECK: cinm.op.reduce add(%[[AT]]) into %[[OT]]
  cinm.op.reduce add (%a) into %o { cinm.tile_sizes = array<i64: 16, 32> }
    : memref<64x128xi32> into memref<64xi32>
  return
}

// CHECK-LABEL: @reduce_dim0
func.func @reduce_dim0(%a: memref<64x128xi32>, %o: memref<128xi32>) {
  // The output tile is indexed by the *parallel* loop, which for dim-0
  // reduction is the inner one.
  // CHECK: affine.for %{{.*}} = 0 to 64 step 16 {
  // CHECK: affine.for %[[J:.*]] = 0 to 128 step 32 {
  // CHECK: %[[OT:.*]] = memref.subview %{{.*}}[%[[J]]] [32] [1] : memref<128xi32>
  // CHECK: cinm.op.reduce add(%{{.*}}) dim 0 into %[[OT]]
  cinm.op.reduce add (%a) dim 0 into %o { cinm.tile_sizes = array<i64: 16, 32> }
    : memref<64x128xi32> into memref<128xi32>
  return
}

// CHECK-LABEL: @reduce_to_rank0
func.func @reduce_to_rank0(%a: memref<128xi32>, %o: memref<i32>) {
  // Nothing is left to slice in a fully-reduced output: the destination is
  // used whole.
  // CHECK: affine.for %[[I:.*]] = 0 to 128 step 32 {
  // CHECK: %[[AT:.*]] = memref.subview %{{.*}}[%[[I]]] [32] [1]
  // CHECK: cinm.op.reduce add(%[[AT]]) into %[[O:.*]] : memref<32xi32{{.*}}> into memref<i32>
  cinm.op.reduce add (%a) into %o { cinm.tile_sizes = array<i64: 32> }
    : memref<128xi32> into memref<i32>
  return
}

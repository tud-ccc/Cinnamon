// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @batch_gemv_memref
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[x:.*]]: memref<{{.*}}>) ->
func.func @batch_gemv_memref(%arg0: memref<4x8x1024xi32>, %arg1: memref<4x1024xi32>) -> memref<4x8xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: linalg.fill ins({{.*}}) outs(%[[out]] :
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2
  // CHECK-NOT: iter_args
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128
  // CHECK: %[[sliceA:.*]] = memref.subview %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceX:.*]] = memref.subview %[[x]][%[[b]], %[[k]]] [2, 128] [1, 1] :
  // CHECK: cinm.op.batch_gemv %[[sliceA]], %[[sliceX]] into %[[sliceOut]] :
  %alloc = memref.alloc() : memref<4x8xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<4x8xi32>)
  cinm.op.batch_gemv %arg0, %arg1 into %alloc {cinm.tile_sizes = array<i64: 2, 8, 128>}
      : memref<4x8x1024xi32>, memref<4x1024xi32> into memref<4x8xi32>
  return %alloc : memref<4x8xi32>
}

// -----
// CHECK-LABEL: @batch_gemv_memref_bias
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[x:.*]]: memref<{{.*}}>, %[[bias:.*]]: memref<{{.*}}>) ->
func.func @batch_gemv_memref_bias(%arg0: memref<4x8x1024xi32>, %arg1: memref<4x1024xi32>, %bias: memref<4x8xi32>) -> memref<4x8xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: %[[sliceBias:.*]] = memref.subview %[[bias]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: linalg.add ins(%[[sliceBias]], %[[sliceOut]] : {{.*}}) outs(%[[sliceOut]] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128
  // CHECK: %[[sliceA:.*]] = memref.subview %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceX:.*]] = memref.subview %[[x]][%[[b]], %[[k]]] [2, 128] [1, 1] :
  // CHECK: cinm.op.batch_gemv %[[sliceA]], %[[sliceX]] into %[[sliceOut]] :
  %alloc = memref.alloc() : memref<4x8xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<4x8xi32>)
  cinm.op.batch_gemv %arg0, %arg1 plus %bias into %alloc {cinm.tile_sizes = array<i64: 2, 8, 128>}
      : memref<4x8x1024xi32>, memref<4x1024xi32> plus memref<4x8xi32> into memref<4x8xi32>
  return %alloc : memref<4x8xi32>
}

// -----
// CHECK-LABEL: @batch_gemv_tensor
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[x:.*]]: tensor<{{.*}}>) ->
func.func @batch_gemv_tensor(%A: tensor<4x8x1024xi32>, %x: tensor<4x1024xi32>) -> tensor<4x8xi32> {
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2 iter_args(%[[acc0:.*]] =
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%[[acc1:.*]] =
  // CHECK: %[[cst:.*]] = arith.constant dense<0> : tensor<2x8xi32>
  // CHECK: %[[init:.*]] = tensor.insert_slice %[[cst]] into %[[acc1]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(%[[acc2:.*]] = %[[init]])
  // CHECK: %[[sliceA:.*]] = tensor.extract_slice %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceX:.*]] = tensor.extract_slice %[[x]][%[[b]], %[[k]]] [2, 128] [1, 1] :
  // CHECK: %[[sliceAcc:.*]] = tensor.extract_slice %[[acc2]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: %[[r:.*]] = cinm.op.batch_gemv %[[sliceA]], %[[sliceX]] plus %[[sliceAcc]] into %[[sliceAcc]] :
  // CHECK: tensor.insert_slice %[[r]] into %[[acc2]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  %r = cinm.op.batch_gemv %A, %x {cinm.tile_sizes = array<i64: 2, 8, 128>}
      : tensor<4x8x1024xi32>, tensor<4x1024xi32> -> tensor<4x8xi32>
  func.return %r : tensor<4x8xi32>
}

// -----
// CHECK-LABEL: @batch_gemv_tensor_bias
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[x:.*]]: tensor<{{.*}}>, %[[bias:.*]]: tensor<{{.*}}>) ->
func.func @batch_gemv_tensor_bias(%A: tensor<4x8x1024xi32>, %x: tensor<4x1024xi32>, %bias: tensor<4x8xi32>) -> tensor<4x8xi32> {
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2 iter_args(%[[acc0:.*]] =
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%[[acc1:.*]] =
  // CHECK: %[[biasSlice:.*]] = tensor.extract_slice %[[bias]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: %[[init:.*]] = tensor.insert_slice %[[biasSlice]] into %[[acc1]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(%[[acc2:.*]] = %[[init]])
  // CHECK: %[[sliceA:.*]] = tensor.extract_slice %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceX:.*]] = tensor.extract_slice %[[x]][%[[b]], %[[k]]] [2, 128] [1, 1] :
  // CHECK: %[[sliceAcc:.*]] = tensor.extract_slice %[[acc2]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  // CHECK: %[[r:.*]] = cinm.op.batch_gemv %[[sliceA]], %[[sliceX]] plus %[[sliceAcc]] into %[[sliceAcc]] :
  // CHECK: tensor.insert_slice %[[r]] into %[[acc2]][%[[b]], %[[i]]] [2, 8] [1, 1] :
  %r = cinm.op.batch_gemv %A, %x plus %bias {cinm.tile_sizes = array<i64: 2, 8, 128>}
      : tensor<4x8x1024xi32>, tensor<4x1024xi32> plus tensor<4x8xi32> -> tensor<4x8xi32>
  func.return %r : tensor<4x8xi32>
}

// -----
// CHECK: #[[map:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @batch_gemv_tensor_dynamic_batch
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[x:.*]]: tensor<{{.*}}>) ->
func.func @batch_gemv_tensor_dynamic_batch(%A: tensor<?x8x1024xi32>, %x: tensor<?x1024xi32>) -> tensor<?x8xi32> {
  // Batch is dynamic, M and K are static.
  // CHECK: %[[Batch:.*]] = tensor.dim %[[A]], {{.*}}
  // CHECK: %[[init:.*]] = tensor.empty(%[[Batch]]) : tensor<?x8xi32>
  // CHECK: affine.for %[[b:.*]] = 0 to #[[map]](%[[Batch]]) step 2 iter_args(%[[acc0:.*]] = %[[init]])
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(
  // CHECK: %[[cst:.*]] = arith.constant dense<0> : tensor<2x8xi32>
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(
  // CHECK: tensor.extract_slice %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: tensor.extract_slice %[[x]][%[[b]], %[[k]]] [2, 128] [1, 1] :
  // CHECK: cinm.op.batch_gemv
  %r = cinm.op.batch_gemv %A, %x {cinm.tile_sizes = array<i64: 2, 8, 128>}
      : tensor<?x8x1024xi32>, tensor<?x1024xi32> -> tensor<?x8xi32>
  func.return %r : tensor<?x8xi32>
}

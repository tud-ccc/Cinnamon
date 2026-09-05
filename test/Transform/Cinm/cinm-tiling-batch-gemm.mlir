// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @batch_gemm_memref
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[B:.*]]: memref<{{.*}}>) ->
func.func @batch_gemm_memref(%arg0: memref<4x8x1024xi32>, %arg1: memref<4x1024x128xi32>) -> memref<4x8x128xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: linalg.fill ins({{.*}}) outs(%[[out]] :
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2
  // CHECK-NOT: iter_args
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 32
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128
  // CHECK: %[[sliceA:.*]] = memref.subview %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceB:.*]] = memref.subview %[[B]][%[[b]], %[[k]], %[[j]]] [2, 128, 32] [1, 1, 1] :
  // CHECK: cinm.op.batch_gemm %[[sliceA]], %[[sliceB]] into %[[sliceOut]] :
  %alloc = memref.alloc() : memref<4x8x128xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<4x8x128xi32>)
  cinm.op.batch_gemm %arg0, %arg1 into %alloc {cinm.tile_sizes = array<i64: 2, 8, 32, 128>}
      : memref<4x8x1024xi32>, memref<4x1024x128xi32> into memref<4x8x128xi32>
  return %alloc : memref<4x8x128xi32>
}

// -----
// CHECK-LABEL: @batch_gemm_memref_bias
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[B:.*]]: memref<{{.*}}>, %[[bias:.*]]: memref<{{.*}}>) ->
func.func @batch_gemm_memref_bias(%arg0: memref<4x8x1024xi32>, %arg1: memref<4x1024x128xi32>, %bias: memref<4x8x128xi32>) -> memref<4x8x128xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 32
  // CHECK: %[[sliceBias:.*]] = memref.subview %[[bias]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: linalg.add ins(%[[sliceBias]], %[[sliceOut]] : {{.*}}) outs(%[[sliceOut]] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128
  // CHECK: %[[sliceA:.*]] = memref.subview %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceB:.*]] = memref.subview %[[B]][%[[b]], %[[k]], %[[j]]] [2, 128, 32] [1, 1, 1] :
  // CHECK: cinm.op.batch_gemm %[[sliceA]], %[[sliceB]] into %[[sliceOut]] :
  %alloc = memref.alloc() : memref<4x8x128xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<4x8x128xi32>)
  cinm.op.batch_gemm %arg0, %arg1 plus %bias into %alloc {cinm.tile_sizes = array<i64: 2, 8, 32, 128>}
      : memref<4x8x1024xi32>, memref<4x1024x128xi32> plus memref<4x8x128xi32> into memref<4x8x128xi32>
  return %alloc : memref<4x8x128xi32>
}

// -----
// CHECK-LABEL: @batch_gemm_tensor
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[B:.*]]: tensor<{{.*}}>) ->
func.func @batch_gemm_tensor(%A: tensor<4x8x1024xi32>, %B: tensor<4x1024x128xi32>) -> tensor<4x8x128xi32> {
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2 iter_args(%[[acc0:.*]] =
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%[[acc1:.*]] =
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 32 iter_args(%[[acc2:.*]] =
  // CHECK: %[[cst:.*]] = arith.constant dense<0> : tensor<2x8x32xi32>
  // CHECK: %[[init:.*]] = tensor.insert_slice %[[cst]] into %[[acc2]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(%[[acc3:.*]] = %[[init]])
  // CHECK: %[[sliceA:.*]] = tensor.extract_slice %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceB:.*]] = tensor.extract_slice %[[B]][%[[b]], %[[k]], %[[j]]] [2, 128, 32] [1, 1, 1] :
  // CHECK: %[[sliceAcc:.*]] = tensor.extract_slice %[[acc3]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: %[[r:.*]] = cinm.op.batch_gemm %[[sliceA]], %[[sliceB]] plus %[[sliceAcc]] into %[[sliceAcc]] :
  // CHECK: tensor.insert_slice %[[r]] into %[[acc3]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  %r = cinm.op.batch_gemm %A, %B {cinm.tile_sizes = array<i64: 2, 8, 32, 128>}
      : tensor<4x8x1024xi32>, tensor<4x1024x128xi32> -> tensor<4x8x128xi32>
  func.return %r : tensor<4x8x128xi32>
}

// -----
// CHECK-LABEL: @batch_gemm_tensor_bias
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[B:.*]]: tensor<{{.*}}>, %[[bias:.*]]: tensor<{{.*}}>) ->
func.func @batch_gemm_tensor_bias(%A: tensor<4x8x1024xi32>, %B: tensor<4x1024x128xi32>, %bias: tensor<4x8x128xi32>) -> tensor<4x8x128xi32> {
  // CHECK: affine.for %[[b:.*]] = 0 to 4 step 2 iter_args(%[[acc0:.*]] =
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%[[acc1:.*]] =
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 32 iter_args(%[[acc2:.*]] =
  // CHECK: %[[biasSlice:.*]] = tensor.extract_slice %[[bias]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: %[[init:.*]] = tensor.insert_slice %[[biasSlice]] into %[[acc2]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(%[[acc3:.*]] = %[[init]])
  // CHECK: %[[sliceA:.*]] = tensor.extract_slice %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: %[[sliceB:.*]] = tensor.extract_slice %[[B]][%[[b]], %[[k]], %[[j]]] [2, 128, 32] [1, 1, 1] :
  // CHECK: %[[sliceAcc:.*]] = tensor.extract_slice %[[acc3]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  // CHECK: %[[r:.*]] = cinm.op.batch_gemm %[[sliceA]], %[[sliceB]] plus %[[sliceAcc]] into %[[sliceAcc]] :
  // CHECK: tensor.insert_slice %[[r]] into %[[acc3]][%[[b]], %[[i]], %[[j]]] [2, 8, 32] [1, 1, 1] :
  %r = cinm.op.batch_gemm %A, %B plus %bias {cinm.tile_sizes = array<i64: 2, 8, 32, 128>}
      : tensor<4x8x1024xi32>, tensor<4x1024x128xi32> plus tensor<4x8x128xi32> -> tensor<4x8x128xi32>
  func.return %r : tensor<4x8x128xi32>
}

// -----
// CHECK: #[[map:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @batch_gemm_tensor_dynamic_batch
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[B:.*]]: tensor<{{.*}}>) ->
func.func @batch_gemm_tensor_dynamic_batch(%A: tensor<?x8x1024xi32>, %B: tensor<?x1024x128xi32>) -> tensor<?x8x128xi32> {
  // Batch is dynamic, M, N, K are static.
  // CHECK: %[[Batch:.*]] = tensor.dim %[[A]], {{.*}}
  // CHECK: %[[init:.*]] = tensor.empty(%[[Batch]]) : tensor<?x8x128xi32>
  // CHECK: affine.for %[[b:.*]] = 0 to #[[map:.*]](%[[Batch]]) step 2 iter_args(%[[acc0:.*]] = %[[init]])
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 32 iter_args(
  // CHECK: %[[cst:.*]] = arith.constant dense<0> : tensor<2x8x32xi32>
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(
  // CHECK: tensor.extract_slice %[[A]][%[[b]], %[[i]], %[[k]]] [2, 8, 128] [1, 1, 1] :
  // CHECK: tensor.extract_slice %[[B]][%[[b]], %[[k]], %[[j]]] [2, 128, 32] [1, 1, 1] :
  // CHECK: cinm.op.batch_gemm
  %r = cinm.op.batch_gemm %A, %B {cinm.tile_sizes = array<i64: 2, 8, 32, 128>}
      : tensor<?x8x1024xi32>, tensor<?x1024x128xi32> -> tensor<?x8x128xi32>
  func.return %r : tensor<?x8x128xi32>
}

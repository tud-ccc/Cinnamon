// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @gemm_memref
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[B:.*]]: memref<{{.*}}>) ->
func.func @gemm_memref(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>) -> memref<8x128xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: linalg.fill ins({{.*}}) outs(%[[out]] :
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 32
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[i]], %[[j]]] [8, 32] [1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128
  // CHECK: %[[sliceA:.*]] = memref.subview %[[A]][%[[i]], %[[k]]] [8, 128] [1, 1] :
  // CHECK: %[[sliceB:.*]] = memref.subview %[[B]][%[[k]], %[[j]]] [128, 32] [1, 1] :
  // CHECK: cinm.op.gemm %[[sliceA]], %[[sliceB]] into %[[sliceOut]] :
  %alloc = memref.alloc() : memref<8x128xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x128xi32>)
  cinm.op.gemm %arg0, %arg1 into %alloc {cinm.tile_sizes = array<i64: 8, 32, 128>}
      : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
  return %alloc : memref<8x128xi32>
}

// -----
// CHECK-LABEL: @gemm_memref_bias
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[B:.*]]: memref<{{.*}}>, %[[bias:.*]]: memref<{{.*}}>) ->
func.func @gemm_memref_bias(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>, %bias: memref<8x128xi32>) -> memref<8x128xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 32
  // CHECK: %[[sliceBias:.*]] = memref.subview %[[bias]][%[[i]], %[[j]]] [8, 32] [1, 1] :
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[i]], %[[j]]] [8, 32] [1, 1] :
  // CHECK: linalg.add ins(%[[sliceBias]], %[[sliceOut]] : {{.*}}) outs(%[[sliceOut]] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 128
  // CHECK: %[[sliceA:.*]] = memref.subview %[[A]][%[[i]], %[[k]]] [8, 128] [1, 1] :
  // CHECK: %[[sliceB:.*]] = memref.subview %[[B]][%[[k]], %[[j]]] [128, 32] [1, 1] :
  // CHECK: cinm.op.gemm %[[sliceA]], %[[sliceB]] into %[[sliceOut]] :
  %alloc = memref.alloc() : memref<8x128xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x128xi32>)
  cinm.op.gemm %arg0, %arg1 plus %bias into %alloc {cinm.tile_sizes = array<i64: 8, 32, 128>}
      : memref<8x1024xi32>, memref<1024x128xi32> plus memref<8x128xi32> into memref<8x128xi32>
  return %alloc : memref<8x128xi32>
}

// -----
// CHECK-LABEL: @gemm_tensor
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[B:.*]]: tensor<{{.*}}>) ->
// CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%
// CHECK: affine.for %[[j:.*]] = 0 to 128 step 32 iter_args(%[[outer:.*]] =
// CHECK: %[[innerinit:.*]] = arith.constant dense<0> :
// CHECK: %[[x:.*]] = affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(%[[inner:.*]] = %[[innerinit]])
// CHECK: %[[sliceA:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [8, 128] [1, 1] :
// CHECK: %[[sliceB:.*]] = tensor.extract_slice %[[B]][%[[k]], %[[j]]] [128, 32] [1, 1] :
// CHECK: %[[r:.*]] = cinm.op.gemm %[[sliceA]], %[[sliceB]] plus %[[inner]] :
// CHECK: affine.yield %[[r]]
// CHECK: tensor.insert_slice %[[x]] into %[[outer]][%[[i]], %[[j]]] [8, 32] [1, 1] :
func.func @gemm_tensor(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {
  %r = cinm.op.gemm %A, %B {cinm.tile_sizes = array<i64: 8, 32, 128>}
      : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
  func.return %r : tensor<8x128xi32>
}

// -----
// CHECK-LABEL: @gemm_tensor_bias
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[B:.*]]: tensor<{{.*}}>, %[[bias:.*]]: tensor<{{.*}}>) ->
// CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%
// CHECK: affine.for %[[j:.*]] = 0 to 128 step 32 iter_args(%[[outer:.*]] =
// CHECK: %[[innerinit:.*]] = tensor.extract_slice %[[bias]][%[[i]], %[[j]]] [8, 32] [1, 1] :
// CHECK: %[[x:.*]] = affine.for %[[k:.*]] = 0 to 1024 step 128 iter_args(%[[inner:.*]] = %[[innerinit]])
// CHECK: %[[sliceA:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [8, 128] [1, 1] :
// CHECK: %[[sliceB:.*]] = tensor.extract_slice %[[B]][%[[k]], %[[j]]] [128, 32] [1, 1] :
// CHECK: %[[r:.*]] = cinm.op.gemm %[[sliceA]], %[[sliceB]] plus %[[inner]] :
// CHECK: affine.yield %[[r]]
// CHECK: tensor.insert_slice %[[x]] into %[[outer]][%[[i]], %[[j]]] [8, 32] [1, 1] :
func.func @gemm_tensor_bias(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>, %bias: tensor<8x128xi32>) -> tensor<8x128xi32> {
  %r = cinm.op.gemm %A, %B plus %bias {cinm.tile_sizes = array<i64: 8, 32, 128>}
      : tensor<8x1024xi32>, tensor<1024x128xi32> plus tensor<8x128xi32> -> tensor<8x128xi32>
  func.return %r : tensor<8x128xi32>
}

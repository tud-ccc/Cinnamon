// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @gemv_memref
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[x:.*]]: memref<{{.*}}>) ->
func.func @gemv_memref(%arg0: memref<64x256xi32>, %arg1: memref<256xi32>) -> memref<64xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: linalg.fill ins({{.*}}) outs(%[[out]] :
  // CHECK: affine.for %[[i:.*]] = 0 to 64 step 8
  // CHECK-NOT: iter_args
  // CHECK: %[[outSlice:.*]] = memref.subview %[[out]][%[[i]]] [8] [1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 256 step 32
  // CHECK: %[[aTile:.*]] = memref.subview %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[xTile:.*]] = memref.subview %[[x]][%[[k]]] [32] [1] :
  // CHECK: cinm.op.gemv %[[aTile]], %[[xTile]] into %[[outSlice]] :
  %alloc = memref.alloc() : memref<64xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64xi32>)
  cinm.op.gemv %arg0, %arg1 into %alloc {cinm.tile_sizes = array<i64: 8, 32>} : memref<64x256xi32>, memref<256xi32> into memref<64xi32>
  return %alloc : memref<64xi32>
}

// -----
// CHECK-LABEL: @gemv_memref_bias
// CHECK-SAME: (%[[A:.*]]: memref<{{.*}}>, %[[x:.*]]: memref<{{.*}}>, %[[c:.*]]: memref<{{.*}}>) ->
func.func @gemv_memref_bias(%arg0: memref<64x256xi32>, %arg1: memref<256xi32>, %bias: memref<64xi32>) -> memref<64xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: affine.for %[[i:.*]] = 0 to 64 step 8
  // CHECK: %[[biasSlice:.*]] = memref.subview %[[c]][%[[i]]] [8] [1] :
  // CHECK: %[[outSlice:.*]] = memref.subview %[[out]][%[[i]]] [8] [1] :
  // CHECK: linalg.add ins(%[[biasSlice]], %[[outSlice]] : {{.*}}) outs(%[[outSlice]] :
  // CHECK: affine.for %[[k:.*]] = 0 to 256 step 32
  // CHECK: %[[aTile:.*]] = memref.subview %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[xTile:.*]] = memref.subview %[[x]][%[[k]]] [32] [1] :
  // CHECK: cinm.op.gemv %[[aTile]], %[[xTile]] into %[[outSlice]] :
  %0 = memref.alloc() : memref<64xi32>
  cinm.op.gemv %arg0, %arg1 plus %bias into %0 { cinm.tile_sizes = array<i64: 8, 32>}
      : memref<64x256xi32>, memref<256xi32> plus memref<64xi32> into memref<64xi32>
  return %0 : memref<64xi32>
}

// -----
// CHECK-LABEL: @gemv_tensor
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[x:.*]]: tensor<{{.*}}>) ->
func.func @gemv_tensor(%A: tensor<64x256xi32>, %x: tensor<256xi32>) -> tensor<64xi32> {
  // CHECK: affine.for %[[i:.*]] = 0 to 64 step 8 iter_args(%[[acc0:.*]] =
  // CHECK: %[[cst:.*]] = arith.constant dense<0> : tensor<8xi32>
  // CHECK: %[[init:.*]] = tensor.insert_slice %[[cst]] into %[[acc0]][%[[i]]] [8] [1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 256 step 32 iter_args(%[[acc1:.*]] = %[[init]])
  // CHECK: %[[aTile:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[xTile:.*]] = tensor.extract_slice %[[x]][%[[k]]] [32] [1] :
  // CHECK: %[[sliceAcc:.*]] = tensor.extract_slice %[[acc1]][%[[i]]] [8] [1] :
  // CHECK: %[[r:.*]] = cinm.op.gemv %[[aTile]], %[[xTile]] plus %[[sliceAcc]] into %[[sliceAcc]] :
  // CHECK: tensor.insert_slice %[[r]] into %[[acc1]][%[[i]]] [8] [1] :
  %0 = cinm.op.gemv %A, %x {cinm.tile_sizes = array<i64: 8, 32>}: tensor<64x256xi32>, tensor<256xi32> -> tensor<64xi32>
  func.return %0 : tensor<64xi32>
}

// -----
// CHECK-LABEL: @gemv_tensor_bias
// CHECK-SAME: (%[[A:.*]]: tensor<{{.*}}>, %[[x:.*]]: tensor<{{.*}}>, %[[bias:.*]]: tensor<{{.*}}>) ->
func.func @gemv_tensor_bias(%A: tensor<64x256xi32>, %x: tensor<256xi32>, %bias: tensor<64xi32>) -> tensor<64xi32> {
  // CHECK: affine.for %[[i:.*]] = 0 to 64 step 8 iter_args(%[[acc0:.*]] =
  // CHECK: %[[biasSlice:.*]] = tensor.extract_slice %[[bias]][%[[i]]] [8] [1] :
  // CHECK: %[[init:.*]] = tensor.insert_slice %[[biasSlice]] into %[[acc0]][%[[i]]] [8] [1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 256 step 32 iter_args(%[[acc1:.*]] = %[[init]])
  // CHECK: %[[aTile:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[xTile:.*]] = tensor.extract_slice %[[x]][%[[k]]] [32] [1] :
  // CHECK: %[[sliceAcc:.*]] = tensor.extract_slice %[[acc1]][%[[i]]] [8] [1] :
  // CHECK: %[[r:.*]] = cinm.op.gemv %[[aTile]], %[[xTile]] plus %[[sliceAcc]] into %[[sliceAcc]] :
  // CHECK: tensor.insert_slice %[[r]] into %[[acc1]][%[[i]]] [8] [1] :
  %0 = cinm.op.gemv %A, %x plus %bias {cinm.tile_sizes = array<i64: 8, 32>}: tensor<64x256xi32>, tensor<256xi32> plus tensor<64xi32> -> tensor<64xi32>
  func.return %0 : tensor<64xi32>
}

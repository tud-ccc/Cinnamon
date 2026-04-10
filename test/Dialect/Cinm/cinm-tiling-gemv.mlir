// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @gemv_memref
func.func @gemv_memref(%arg0: memref<64x256xi32>, %arg1: memref<256xi32>) -> memref<64xi32> {
  // CHECK: cinm.compute (%[[A:.*]] = %{{.*}}, %[[x:.*]] = %{{.*}}) ->
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: linalg.fill ins({{.*}}) outs(%[[out]] :
  // CHECK: affine.for %[[i:.*]] = 0 to 64 step 8
  // CHECK-NOT: iter_args
  // CHECK: %[[outSlice:.*]] = memref.subview %[[out]][%[[i]]] [8] [1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 256 step 32
  // CHECK: %[[aTile:.*]] = memref.subview %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[xTile:.*]] = memref.subview %[[x]][%[[k]]] [32] [1] :
  // CHECK: cinm.op.gemv %[[aTile]], %[[xTile]] into %[[outSlice]] {cinm.notile} :
  %0 = cinm.compute (%A = %arg0 : memref<64x256xi32>, %x = %arg1 : memref<256xi32>) -> memref<64xi32>
      attributes {workgroupShape = array<i64: 8, 1, 1>, bufferSizesInBytes = array<i64: 0, 0, 512>,
                  tileSizes = array<i64: 8, 32>} {
    %alloc = memref.alloc() : memref<64xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64xi32>)
    cinm.op.gemv %A, %x into %alloc : memref<64x256xi32>, memref<256xi32> into memref<64xi32>
    cinm.yield %alloc : memref<64xi32>
  }
  return %0 : memref<64xi32>
}

// -----
// CHECK-LABEL: @gemv_memref_bias
// CHECK-SAME: ({{.*}}, %[[bias:.*]]: memref<64xi32>)
func.func @gemv_memref_bias(%arg0: memref<64x256xi32>, %arg1: memref<256xi32>, %bias: memref<64xi32>) -> memref<64xi32> {
  // CHECK: cinm.compute (%[[A:.*]] = %{{.*}}, %[[x:.*]] = %{{.*}}, %[[c:.*]] = %{{.*}}) ->
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: affine.for %[[i:.*]] = 0 to 64 step 8
  // CHECK: %[[biasSlice:.*]] = memref.subview %[[c]][%[[i]]] [8] [1] :
  // CHECK: %[[outSlice:.*]] = memref.subview %[[out]][%[[i]]] [8] [1] :
  // CHECK: linalg.add ins(%[[biasSlice]], %[[outSlice]] : {{.*}}) outs(%[[outSlice]] :
  // CHECK: affine.for %[[k:.*]] = 0 to 256 step 32
  // CHECK: %[[aTile:.*]] = memref.subview %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[xTile:.*]] = memref.subview %[[x]][%[[k]]] [32] [1] :
  // CHECK: cinm.op.gemv %[[aTile]], %[[xTile]] into %[[outSlice]] {cinm.notile} :
  %0 = cinm.compute (%A = %arg0 : memref<64x256xi32>, %x = %arg1 : memref<256xi32>, %c = %bias : memref<64xi32>) -> memref<64xi32>
      attributes {workgroupShape = array<i64: 8, 1, 1>, bufferSizesInBytes = array<i64: 0, 0, 512>,
                  tileSizes = array<i64: 8, 32>} {
    %alloc = memref.alloc() : memref<64xi32>
    cinm.op.gemv %A, %x plus %c into %alloc
        : memref<64x256xi32>, memref<256xi32> plus memref<64xi32> into memref<64xi32>
    cinm.yield %alloc : memref<64xi32>
  }
  return %0 : memref<64xi32>
}

// -----
// CHECK-LABEL: @gemv_tensor
// CHECK: cinm.compute (%[[A:.*]] = %{{.*}}, %[[x:.*]] = %{{.*}}) ->
// CHECK: affine.for %[[i:.*]] = 0 to 64 step 8 iter_args(%
// CHECK: %[[acc0:.*]] = arith.constant dense<0> : tensor<8xi32>
// CHECK: %[[red:.*]] = affine.for %[[k:.*]] = 0 to 256 step 32 iter_args(%[[acc:.*]] = %[[acc0]])
// CHECK: %[[aTile:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
// CHECK: %[[xTile:.*]] = tensor.extract_slice %[[x]][%[[k]]] [32] [1] :
// CHECK: %[[r:.*]] = cinm.op.gemv %[[aTile]], %[[xTile]] plus %[[acc]] {cinm.notile} :
// CHECK: affine.yield %[[r]]
// CHECK: tensor.insert_slice %[[red]] into %{{.*}}[%[[i]]] [8] [1] :
func.func @gemv_tensor(%A: tensor<64x256xi32>, %x: tensor<256xi32>) -> tensor<64xi32> {
  %r0 = cinm.compute (%a = %A : tensor<64x256xi32>, %b = %x : tensor<256xi32>) -> tensor<64xi32>
      attributes {workgroupShape = array<i64: 8, 1, 1>, bufferSizesInBytes = array<i64: 0, 0, 512>,
                  tileSizes = array<i64: 8, 32>} {
    %r = cinm.op.gemv %a, %b : tensor<64x256xi32>, tensor<256xi32> -> tensor<64xi32>
    cinm.yield %r : tensor<64xi32>
  }
  func.return %r0 : tensor<64xi32>
}

// -----
// CHECK-LABEL: @gemv_tensor_bias
// CHECK: cinm.compute (%[[A:.*]] = %{{.*}}, %[[x:.*]] = %{{.*}}, %[[bias:.*]] = %{{.*}}) ->
// CHECK: affine.for %[[i:.*]] = 0 to 64 step 8 iter_args(%
// CHECK: %[[biasSlice:.*]] = tensor.extract_slice %[[bias]][%[[i]]] [8] [1] :
// CHECK: %[[red:.*]] = affine.for %[[k:.*]] = 0 to 256 step 32 iter_args(%[[acc:.*]] = %[[biasSlice]])
// CHECK: %[[aTile:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [8, 32] [1, 1] :
// CHECK: %[[xTile:.*]] = tensor.extract_slice %[[x]][%[[k]]] [32] [1] :
// CHECK: %[[r:.*]] = cinm.op.gemv %[[aTile]], %[[xTile]] plus %[[acc]] {cinm.notile} :
// CHECK: affine.yield %[[r]]
// CHECK: tensor.insert_slice %[[red]] into %{{.*}}[%[[i]]] [8] [1] :
func.func @gemv_tensor_bias(%A: tensor<64x256xi32>, %x: tensor<256xi32>, %bias: tensor<64xi32>) -> tensor<64xi32> {
  %r0 = cinm.compute (%a = %A : tensor<64x256xi32>, %b = %x : tensor<256xi32>, %c = %bias : tensor<64xi32>) -> tensor<64xi32>
      attributes {workgroupShape = array<i64: 8, 1, 1>, bufferSizesInBytes = array<i64: 0, 0, 512>,
                  tileSizes = array<i64: 8, 32>} {
    %r = cinm.op.gemv %a, %b plus %c : tensor<64x256xi32>, tensor<256xi32> plus tensor<64xi32> -> tensor<64xi32>
    cinm.yield %r : tensor<64xi32>
  }
  func.return %r0 : tensor<64xi32>
}

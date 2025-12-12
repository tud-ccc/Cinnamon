// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @gemm_memref

func.func @gemm_memref(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>) -> memref<8x128xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: linalg.fill ins({{.*}}) outs(%[[out]] :
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 128
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[i]], %[[j]]] [8, 128] [1, 1] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 32
  // CHECK: %[[sliceA:.*]] = memref.subview %arg0[%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[sliceB:.*]] = memref.subview %arg1[%[[k]], %[[j]]] [32, 128] [1, 1] :
  // CHECK: cinm.op.gemm %[[sliceA]], %[[sliceB]] into %[[sliceOut]] {cinm.notile} :
  %0 = cinm.compute attributes {workgroupShape = array<i64: 8, 128, 1>, bufferSizesInBytes=array<i64: 0,0,512>} -> memref<8x128xi32> {
    %alloc = memref.alloc() : memref<8x128xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x128xi32>)
    cinm.op.gemm %arg0, %arg1 into %alloc : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
    cinm.yield %alloc : memref<8x128xi32>
  }
  return %0 : memref<8x128xi32>
}

// -----
// CHECK-LABEL: @gemm_memref_bias
// CHECK-SAME: ({{.*}}, %[[bias:.*]]: memref<8x128xi32>)

func.func @gemm_memref_bias(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>, %bias: memref<8x128xi32>) -> memref<8x128xi32> {
  // CHECK: %[[out:.*]] = memref.alloc()
  // CHECK: affine.for %[[i:.*]] = 0 to 8 step 8
  // CHECK: affine.for %[[j:.*]] = 0 to 128 step 128
  // CHECK: %[[sliceBias:.*]] = memref.subview %arg2[%[[i]], %[[j]]] [8, 128] [1, 1] :
  // CHECK: %[[sliceOut:.*]] = memref.subview %[[out]][%[[i]], %[[j]]] [8, 128] [1, 1] :
  // CHECK: linalg.add ins(%[[sliceBias]], %[[sliceOut]] : {{.*}}) outs(%[[sliceOut]] :
  // CHECK: affine.for %[[k:.*]] = 0 to 1024 step 32
  // CHECK: %[[sliceA:.*]] = memref.subview %arg0[%[[i]], %[[k]]] [8, 32] [1, 1] :
  // CHECK: %[[sliceB:.*]] = memref.subview %arg1[%[[k]], %[[j]]] [32, 128] [1, 1] :
  // CHECK: cinm.op.gemm %[[sliceA]], %[[sliceB]] into %[[sliceOut]] {cinm.notile} :
  %0 = cinm.compute attributes {workgroupShape = array<i64: 8, 128, 1>, bufferSizesInBytes=array<i64: 0,0,512>} -> memref<8x128xi32> {
    %alloc = memref.alloc() : memref<8x128xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x128xi32>)
    cinm.op.gemm %arg0, %arg1 plus %bias into %alloc : memref<8x1024xi32>, memref<1024x128xi32> plus memref<8x128xi32> into memref<8x128xi32>
    cinm.yield %alloc : memref<8x128xi32>
  }
  return %0 : memref<8x128xi32>
}

// -----
// CHECK-LABEL: @gemm_tensor
// CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%
// CHECK: affine.for %[[j:.*]] = 0 to 128 step 128 iter_args(%[[outer:.*]] =
// CHECK: %[[innerinit:.*]] = arith.constant dense<0> :
// CHECK: %[[x:.*]] = affine.for %[[k:.*]] = 0 to 1024 step 32 iter_args(%[[inner:.*]] = %[[innerinit]])

// CHECK: %[[sliceA:.*]] = tensor.extract_slice %arg0[%[[i]], %[[k]]] [8, 32] [1, 1] :
// CHECK: %[[sliceB:.*]] = tensor.extract_slice %arg1[%[[k]], %[[j]]] [32, 128] [1, 1] :
// CHECK: %[[r:.*]] = cinm.op.gemm %[[sliceA]], %[[sliceB]] plus %[[inner]] {cinm.notile} :
// CHECK: affine.yield %[[r]] 
// CHECK: tensor.insert_slice %[[x]] into %[[outer]][%[[i]], %[[j]]] [8, 128] [1, 1] :
func.func @gemm_tensor(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {
  %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1>, bufferSizesInBytes=array<i64: 0,0,512> } -> tensor<8x128xi32> {
      %r = cinm.op.gemm %A, %B: tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
      cinm.yield %r : tensor<8x128xi32>
  }
  func.return %r0 : tensor<8x128xi32>
}

// -----
// CHECK-LABEL: @gemm_tensor_bias
// CHECK-SAME: ({{.*}}, %[[bias:.*]]: tensor<8x128xi32>)
// CHECK: affine.for %[[i:.*]] = 0 to 8 step 8 iter_args(%
// CHECK: affine.for %[[j:.*]] = 0 to 128 step 128 iter_args(%[[outer:.*]] =
// CHECK: %[[innerinit:.*]] = tensor.extract_slice %[[bias]][%[[i]], %[[j]]] [8, 128] [1, 1] :
// CHECK: %[[x:.*]] = affine.for %[[k:.*]] = 0 to 1024 step 32 iter_args(%[[inner:.*]] = %[[innerinit]])

// CHECK: %[[sliceA:.*]] = tensor.extract_slice %arg0[%[[i]], %[[k]]] [8, 32] [1, 1] :
// CHECK: %[[sliceB:.*]] = tensor.extract_slice %arg1[%[[k]], %[[j]]] [32, 128] [1, 1] :
// CHECK: %[[r:.*]] = cinm.op.gemm %[[sliceA]], %[[sliceB]] plus %[[inner]] {cinm.notile} :
// CHECK: affine.yield %[[r]] 
// CHECK: tensor.insert_slice %[[x]] into %[[outer]][%[[i]], %[[j]]] [8, 128] [1, 1] :
func.func @gemm_tensor_bias(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>, %bias: tensor<8x128xi32>) -> tensor<8x128xi32> {
  %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1>, bufferSizesInBytes=array<i64: 0,0,512> } -> tensor<8x128xi32> {
      %r = cinm.op.gemm %A, %B plus %bias: tensor<8x1024xi32>, tensor<1024x128xi32> plus tensor<8x128xi32> -> tensor<8x128xi32>
      cinm.yield %r : tensor<8x128xi32>
  }
  func.return %r0 : tensor<8x128xi32>
}
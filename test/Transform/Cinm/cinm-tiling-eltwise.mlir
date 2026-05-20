// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @sub
// CHECK-SAME: (%[[a:.*]]: tensor<{{.*}}>, %[[b:.*]]: tensor<{{.*}}>) ->
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128 iter_args(%
// CHECK: %[[aSlice:.*]] = tensor.extract_slice %[[a]][%[[i]]] [128] [1] :
// CHECK: %[[bSlice:.*]] = tensor.extract_slice %[[b]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise sub %[[aSlice]], %[[bSlice]] :
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%[[i]]] [128] [1] :
func.func @sub(%a: tensor<1024xi32>, %b: tensor<1024xi32>) -> tensor<1024xi32> {
  %d = cinm.op.elementwise sub %a, %b {cinm.tile_sizes = array<i64: 128>} : tensor<1024xi32>
  return %d: tensor<1024xi32>
}

// -----
// CHECK-LABEL: @sub_memref
// CHECK-SAME: (%[[a:.*]]: memref<{{.*}}>, %[[b:.*]]: memref<{{.*}}>, %[[c:.*]]: memref<{{.*}}>) ->
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128
// CHECK-NOT: iter_args
// CHECK: %[[aSlice:.*]] = memref.subview %[[a]][%[[i]]] [128] [1] :
// CHECK: %[[bSlice:.*]] = memref.subview %[[b]][%[[i]]] [128] [1] :
// CHECK: %[[cSlice:.*]] = memref.subview %[[c]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise sub %[[aSlice]], %[[bSlice]] into %[[cSlice]] :
func.func @sub_memref(%a: memref<1024xi32>, %b: memref<1024xi32>, %c: memref<1024xi32>) -> memref<1024xi32> {
  cinm.op.elementwise sub %a, %b into %c {cinm.tile_sizes = array<i64: 128>} : memref<1024xi32> into memref<1024xi32>
  return %c : memref<1024xi32>
}

// -----
// CHECK-LABEL: @exp
// CHECK-SAME: (%[[a:.*]]: tensor<{{.*}}>) ->
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128 iter_args(%
// CHECK: %[[aSlice:.*]] = tensor.extract_slice %[[a]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise exp %[[aSlice]] :
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%[[i]]] [128] [1] :
func.func @exp(%a: tensor<1024xi32>) -> tensor<1024xi32> {
  %d = cinm.op.elementwise exp %a {cinm.tile_sizes = array<i64: 128>} : tensor<1024xi32>
  return %d: tensor<1024xi32>
}

// -----
// CHECK-LABEL: @exp_memref
// CHECK-SAME: (%[[a:.*]]: memref<{{.*}}>, %[[c:.*]]: memref<{{.*}}>) ->
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128
// CHECK-NOT: iter_args
// CHECK: %[[aSlice:.*]] = memref.subview %[[a]][%[[i]]] [128] [1] :
// CHECK: %[[cSlice:.*]] = memref.subview %[[c]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise exp %[[aSlice]] into %[[cSlice]] :
func.func @exp_memref(%a: memref<1024xi32>, %c: memref<1024xi32>) -> memref<1024xi32> {
  cinm.op.elementwise exp %a into %c {cinm.tile_sizes = array<i64: 128>} : memref<1024xi32> into memref<1024xi32>
  return %c : memref<1024xi32>
}

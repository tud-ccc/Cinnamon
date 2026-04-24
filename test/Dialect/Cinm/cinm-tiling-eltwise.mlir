// RUN: cinm-opt %s --cinm-isolate-compute-blocks --cinm-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @sub
// CHECK: cinm.compute (%[[a:.*]] = %{{.*}}, %[[b:.*]] = %{{.*}}) ->
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128 iter_args(%
// CHECK: %[[aSlice:.*]] = tensor.extract_slice %[[a]][%[[i]]] [128] [1] :
// CHECK: %[[bSlice:.*]] = tensor.extract_slice %[[b]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise sub %[[aSlice]], %[[bSlice]] {cinm.notile} :
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%[[i]]] [128] [1] :
func.func @sub(%a: tensor<1024xi32>, %b: tensor<1024xi32>) -> tensor<1024xi32> {
	%res = cinm.compute_  -> tensor<1024xi32> attributes { tileSizes = array<i64: 128> } {
		%d = cinm.op.elementwise sub %a, %b: tensor<1024xi32>
		cinm.yield %d : tensor<1024xi32>
	}
	return %res: tensor<1024xi32>
}
// -----
// CHECK-LABEL: @sub_memref
// CHECK: cinm.compute (%[[a:.*]] = %{{.*}}, %[[b:.*]] = %{{.*}}, %[[c:.*]] = %{{.*}})
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128
// CHECK-NOT: iter_args
// CHECK: %[[aSlice:.*]] = memref.subview %[[a]][%[[i]]] [128] [1] :
// CHECK: %[[bSlice:.*]] = memref.subview %[[b]][%[[i]]] [128] [1] :
// CHECK: %[[cSlice:.*]] = memref.subview %[[c]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise sub %[[aSlice]], %[[bSlice]] into %[[cSlice]] {cinm.notile} :
func.func @sub_memref(%a: memref<1024xi32>, %b: memref<1024xi32>, %c: memref<1024xi32>) {
	cinm.compute_  attributes { tileSizes = array<i64: 128> } {
		cinm.op.elementwise sub %a, %b into %c: memref<1024xi32> into memref<1024xi32>
		cinm.yield
	}
  return
}

// -----
// CHECK-LABEL: @exp
// CHECK: cinm.compute (%[[a:.*]] = %{{.*}}) ->
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128 iter_args(%
// CHECK: %[[aSlice:.*]] = tensor.extract_slice %[[a]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise exp %[[aSlice]] {cinm.notile} :
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%[[i]]] [128] [1] :
func.func @exp(%a: tensor<1024xi32>) -> tensor<1024xi32> {
	%res = cinm.compute_  -> tensor<1024xi32> attributes { tileSizes = array<i64: 128> } {
		%d = cinm.op.elementwise exp %a: tensor<1024xi32>
		cinm.yield %d : tensor<1024xi32>
	}
	return %res: tensor<1024xi32>
}
// -----
// CHECK-LABEL: @exp_memref
// CHECK: cinm.compute (%[[a:.*]] = %{{.*}}, %[[c:.*]] = %{{.*}})
// CHECK: affine.for %[[i:.*]] = 0 to 1024 step 128
// CHECK-NOT: iter_args
// CHECK: %[[aSlice:.*]] = memref.subview %[[a]][%[[i]]] [128] [1] :
// CHECK: %[[cSlice:.*]] = memref.subview %[[c]][%[[i]]] [128] [1] :
// CHECK: cinm.op.elementwise exp %[[aSlice]] into %[[cSlice]] {cinm.notile} :
func.func @exp_memref(%a: memref<1024xi32>, %c: memref<1024xi32>) {
	cinm.compute_  attributes { tileSizes = array<i64: 128> } {
		cinm.op.elementwise exp %a into %c: memref<1024xi32> into memref<1024xi32>
		cinm.yield
	}
  return
}

// func.func @max(%a: tensor<1024xi32>) -> i32 {
// 	%res = cinm.compute (%a0 = %a : tensor<1024xi32>) -> i32 attributes { workgroupShape = array<i64: 4>, bufferSizesInBytes = array<i64: 1024> } {
// 		%d = cinm.op.reduce max (%a0): tensor<1024xi32> -> i32
// 		cinm.yield %d : i32
// 	}
// 	return %res: i32
// }

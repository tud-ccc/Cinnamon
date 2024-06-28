// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s


// CHECK-LABEL: @gemmSquare
// CHECK-SAME:  (%[[A:.*]]: tensor<1024x1024xi32>, %[[B:.*]]:
// CHECK:       affine.for %[[i:.*]] = 0 to 1024 step 2
// CHECK-NEXT:  affine.for %[[j:.*]] = 0 to 1024 step 64 iter_args(%[[arg0:.*]] =
// CHECK-NEXT:  %[[cst0:.*]] = arith.constant dense<0>
// CHECK-NEXT:  %[[res:.*]] = affine.for %[[k:.*]] = 0 to 1024 step 16 iter_args(%[[arg:.*]] = %[[cst0]]) -> (tensor<2x64xi32>) {
// CHECK-NEXT:  %[[blockA:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [2, 16] [1, 1]
// CHECK-NEXT:  %[[blockB:.*]] = tensor.extract_slice %[[B]][%[[k]], %[[j]]] [16, 64] [1, 1]
// CHECK-NEXT:  %[[tile:.*]] = cinm.op.gemm %[[blockA]], %[[blockB]] plus %[[arg]] {cinm.notile}
// CHECK-NEXT:  affine.yield %[[tile]]
// CHECK-NEXT:  }
// CHECK-NEXT:  tensor.insert_slice %[[res]] into %[[arg0]][%[[i]], %[[j]]] [2, 64] [1, 1]

func.func @gemmSquare(%a: tensor<1024x1024xi32>, %b: tensor<1024x1024xi32>) -> tensor<1024x1024xi32>{
	%res = cinm.compute attributes { workgroupShape=array<i64: 2, 64>, bufferSizesInBytes=array<i64: 256>} -> tensor<1024x1024xi32> {
		%d = cinm.op.gemm %a, %b : (tensor<1024x1024xi32>, tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
		cinm.yield %d: tensor<1024x1024xi32>
	}
	return %res: tensor<1024x1024xi32>
}


// -----

// CHECK-LABEL: @gemv

func.func @gemv(%a: tensor<1024x1024xi32>, %b: tensor<1024xi32>) -> tensor<1024xi32>{
	%res = cinm.compute attributes { workgroupShape=array<i64: 2, 64>, bufferSizesInBytes=array<i64: 256>} -> tensor<1024xi32> {
		%d = cinm.op.gemv %a, %b : (tensor<1024x1024xi32>, tensor<1024xi32>) -> tensor<1024xi32>
		cinm.yield %d: tensor<1024xi32>
	}
	return %res: tensor<1024xi32>
}

// -----

// CHECK-LABEL: @max

func.func @max(%a: tensor<1024xi32>) -> i32 {
	%res = cinm.compute attributes { workgroupShape=array<i64: 2, 64>, bufferSizesInBytes=array<i64: 256>} -> i32 {
		%d = cinm.op.reduce max (%a): tensor<1024xi32>
		cinm.yield %d : i32
	}
	return %res: i32
}
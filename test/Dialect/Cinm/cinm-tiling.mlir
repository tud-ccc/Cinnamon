// RUN: cinm-opt %s --cinm-tiling=reduction-tile-size=16 -split-input-file | FileCheck %s


// CHECK-LABEL: @gemmSquare
// CHECK-SAME:  (%[[A:.*]]: tensor<1024x1024xi32>, %[[B:.*]]:
// CHECK:       affine.for %[[i:.*]] = 0 to 1024 step 64
// CHECK-NEXT:  affine.for %[[j:.*]] = 0 to 1024 step 64
// CHECK-NEXT:  %[[blockA:.*]] = tensor.extract_slice %[[A]][%[[i]], 0] [64, 1024] [1, 1]
// CHECK-NEXT:  %[[blockB:.*]] = tensor.extract_slice %[[B]][0, %[[j]]] [1024, 64] [1, 1]
// CHECK-NEXT:  tensor.generate
// CHECK-NEXT:  ^{{.*}}(%[[ti:.*]]: index, %[[tj:.*]]: index):
// CHECK-NEXT:  %[[row:.*]] = tensor.extract_slice %[[blockA]][%[[ti]], 0] [1, 1024] [1, 1]
// CHECK-NEXT:  %[[col:.*]] = tensor.extract_slice %[[blockB]][0, %[[tj]]] [1024, 1] [1, 1]

// CHECK:  		%[[batchedRow:.*]] = tensor.reshape %[[row]](%{{.*}}) : (tensor<1024xi32>, tensor<2xi64>) -> tensor<64x16xi32>
// CHECK-NEXT:  %[[batchedCol:.*]] = tensor.reshape %[[col]](%{{.*}}) : (tensor<1024xi32>, tensor<2xi64>) -> tensor<64x16xi32>
// CHECK-NEXT:  %[[stage1:.*]] = linalg.reduce ins(%[[batchedRow]], %[[batchedCol]] : {{.*}}) outs(%{{.*}} : tensor<64xi32>) dimensions = [1]
// CHECK-NEXT:    (%[[ei:.*]]: i32, %[[ej:.*]]: i32, %[[init:.*]]: i32)
// CHECK-NEXT:  	%[[mul:.*]] = arith.muli %[[ei]], %[[ej]]
// CHECK-NEXT:  	%[[add:.*]] = arith.addi %[[mul]], %[[init]]
// CHECK-NEXT:  	linalg.yield %[[add]]

// CHECK:       linalg.reduce ins(%[[stage1]] : tensor<64xi32>)

func.func @gemmSquare(%a: tensor<1024x1024xi32>, %b: tensor<1024x1024xi32>) -> tensor<1024x1024xi32>{
	%res = cinm.compute -> tensor<1024x1024xi32> {
		%d = cinm.op.gemm %a, %b : (tensor<1024x1024xi32>, tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
		cinm.yield %d: tensor<1024x1024xi32>
	}
	return %res: tensor<1024x1024xi32>
}


// -----

// CHECK-LABEL: @gemv

func.func @gemv(%a: tensor<1024x1024xi32>, %b: tensor<1024xi32>) -> tensor<1024xi32>{
	%res = cinm.compute -> tensor<1024xi32> {
		%d = cinm.op.gemv %a, %b : (tensor<1024x1024xi32>, tensor<1024xi32>) -> tensor<1024xi32>
		cinm.yield %d: tensor<1024xi32>
	}
	return %res: tensor<1024xi32>
}

// -----

// CHECK-LABEL: @max

func.func @max(%a: tensor<1024xi32>) -> i32 {
	%res = cinm.compute -> i32 {
		%d = cinm.op.reduce max (%a): tensor<1024xi32>
		cinm.yield %d : i32
	}
	return %res: i32
}
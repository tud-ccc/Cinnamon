// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s


// CHECK-LABEL: @gemmSquare
// CHECK-SAME:  (%[[A:.*]]: tensor<1024x1024xi32>, %[[B:.*]]: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
// CHECK:       %[[res0:.*]] = affine.for %[[i:.*]] = 0 to 1024 iter_args({{.*}})
// CHECK-NEXT:   %[[res1:.*]] = affine.for %[[j:.*]] = 0 to 1024 step 1024 iter_args(%[[acc0:.*]] = {{.*}})
// CHECK:         %[[res2:.*]] = affine.for %[[k:.*]] = 0 to 1024 step 256 iter_args(%[[acc1:.*]] = {{.*}})
// CHECK-NEXT:     %[[blockA:.*]] = tensor.extract_slice %[[A]][%[[i]], %[[k]]] [1, 256] [1, 1]
// CHECK-NEXT:     %[[blockB:.*]] = tensor.extract_slice %[[B]][%[[k]], %[[j]]] [256, 1024] [1, 1]
// CHECK-NEXT:     %[[res3:.*]] = cinm.op.gemm %[[blockA]], %[[blockB]] plus %[[acc1]] {cinm.notile}
// CHECK-NEXT:     affine.yield %[[res3]] : tensor<1x1024xi32>
// CHECK:         %[[ins:.*]] = tensor.insert_slice %[[res2]] into %[[acc0]][%[[i]], %[[j]]]
// CHECK-NEXT:    affine.yield %[[ins]] : tensor<1024x1024xi32>
// CHECK:        affine.yield %[[res1]] : tensor<1024x1024xi32>
// CHECK:       cinm.yield %[[res0]] : tensor<1024x1024xi32>

func.func @gemmSquare(%a: tensor<1024x1024xi32>, %b: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {
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
// CHECK-SAME:  (%[[input:.*]]: tensor<1024xi32>) -> i32
// CHECK-NEXT:  %[[res:.*]] = cinm.compute attributes {{{.*}}} -> i32 {
// CHECK:       %[[gen:.*]] = tensor.generate {
// CHECK-NEXT:  ^{{.*}}(%[[idx:.*]]: {{.*}}):
// CHECK-NEXT:    %[[idxOffset:.*]] = arith.muli %[[idx]]
// CHECK-NEXT:    %[[extracted:.*]] = tensor.extract_slice %[[input]][%[[idx]]] [256] [1]
// CHECK-NEXT:    %[[redInner:.*]] = linalg.reduce ins(%[[extracted]] : {{.*}}) outs({{.*}}) dimensions = [0]
// CHECK-NEXT:      (%[[in0:.*]]: {{.*}}, %[[acc0:.*]]: {{.*}})
// CHECK-NEXT:        %[[res0:.*]] = arith.maxsi %[[in0]], %[[acc0]]
// CHECK-NEXT:        linalg.yield %[[res0]]

// CHECK:         %[[extracted0:.*]] = tensor.extract %[[redInner]][] : tensor<i32>
// CHECK-NEXT:    tensor.yield %[[extracted0]]

// CHECK:       %[[redOuter:.*]] = linalg.reduce ins(%[[gen]] : tensor<4xi32>) outs({{.*}}) dimensions = [0]
// CHECK-NEXT:    (%[[in1:.*]]: {{.*}}, %[[acc1:.*]]: {{.*}})
// CHECK-NEXT:      %[[res1:.*]] = arith.maxsi %[[in1]], %[[acc1]]
// CHECK-NEXT:      linalg.yield %[[res1]]

// CHECK:       %[[extracted1:.*]] = tensor.extract %[[redOuter]][]
// CHECK-NEXT:  cinm.yield %[[extracted1]]

func.func @max(%a: tensor<1024xi32>) -> i32 {
	%res = cinm.compute attributes { workgroupShape = array<i64: 4>, bufferSizesInBytes = array<i64: 1024> } -> i32 {
		%d = cinm.op.reduce max (%a): tensor<1024xi32> -> i32
		cinm.yield %d : i32
	}
	return %res: i32
}

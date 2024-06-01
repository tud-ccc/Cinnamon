// RUN: cinm-opt %s --cinm-tiling | FileCheck %s


// CHECK-LABEL: gemmSquare
  func.func @gemmSquare(%a: tensor<1024x1024xi32>, %b: tensor<1024x1024xi32>) -> tensor<1024x1024xi32>{
	%d = cinm.op.gemm %a, %b : (tensor<1024x1024xi32>, tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
	return %d: tensor<1024x1024xi32>
  }


// ------

// CHECK-LABEL: gemv

  func.func @gemv(%a: tensor<1024x1024xi32>, %b: tensor<1024xi32>) -> tensor<1024xi32>{
	%d = cinm.op.gemv %a, %b : (tensor<1024x1024xi32>, tensor<1024xi32>) -> tensor<1024xi32>
	return %d: tensor<1024xi32>
  }

// ------

// CHECK-LABEL: max

  func.func @max(%a: tensor<1024xi32>) -> i32 {
	%d = cinm.op.max %a: tensor<1024xi32>
	return %d: i32
  }
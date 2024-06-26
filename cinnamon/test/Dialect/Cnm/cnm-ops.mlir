// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: matmul

#map1 = affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 16 + d2)>
#map = affine_map<(d0) -> (d0)>

  func.func @va_8(%arg0: tensor<8x2097152xi32>, %arg1: tensor<8x2097152xi32>) {
    %cst = arith.constant dense<0> : tensor<16384x1024xi32>
    %cst_0 = arith.constant dense<[16384, 1024]> : tensor<2xi64>
    %cst_1 = arith.constant dense<16777216> : tensor<1xi64>
    %reshape = tensor.reshape %arg0(%cst_1) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %reshape_2 = tensor.reshape %arg1(%cst_1) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %0 = cnm.workgroup : !cnm.workgroup<16x64x16>
    %reshape_3 = tensor.reshape %reshape(%cst_0) : (tensor<16777216xi32>, tensor<2xi64>) -> tensor<16384x1024xi32>
    %1 = cnm.alloc() for %0 : !cnm.buffer<1024xi32 on 16x64x16, level 0>
    cnm.scatter %reshape_3 into %1[#map1] of %0 : tensor<16384x1024xi32> into !cnm.buffer<1024xi32 on 16x64x16, level 0>
    %reshape_4 = tensor.reshape %reshape_2(%cst_0) : (tensor<16777216xi32>, tensor<2xi64>) -> tensor<16384x1024xi32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<1024xi32 on 16x64x16, level 0>
    cnm.scatter %reshape_4 into %2[#map1] of %0 : tensor<16384x1024xi32> into !cnm.buffer<1024xi32 on 16x64x16, level 0>
    %3 = cnm.alloc() for %0 : !cnm.buffer<1024xi32 on 16x64x16, level 0>
    cnm.scatter %cst into %3[#map1] of %0 : tensor<16384x1024xi32> into !cnm.buffer<1024xi32 on 16x64x16, level 0>
    cnm.launch %0 in(%1, %2 : !cnm.buffer<1024xi32 on 16x64x16, level 0>, !cnm.buffer<1024xi32 on 16x64x16, level 0>) out(%3 : !cnm.buffer<1024xi32 on 16x64x16, level 0>) on !cnm.workgroup<16x64x16> {
    ^bb0(%arg2: memref<1024xi32>, %arg3: memref<1024xi32>, %arg4: memref<1024xi32>):
      linalg.add ins(%arg2, %arg3 : memref<1024xi32>, memref<1024xi32>) outs(%arg4 : memref<1024xi32>)
    }
    %4 = tensor.empty() : tensor<16384x1024xi32>
    %5 = cnm.gather %3[#map1] of %0 into %4 : !cnm.buffer<1024xi32 on 16x64x16, level 0> into tensor<16384x1024xi32>
    cnm.free_workgroup %0 : !cnm.workgroup<16x64x16>
    return
}


func.func @transform(%A: memref<2x512xi32>, %O: memref<2x512xi32>) {
  cnm.compute
  ins(%A[#map]: memref<2x512xi32>)
  outs(%O[#map]: memref<2x512xi32>)
  on hierarchy<2>
  do (%a1: memref<512xi32>,
      %o1: memref<512xi32>)  {
    affine.parallel (%i) = (0) to (512) {
      %x = memref.load %a1[%i]: memref<512xi32>
      %cst2 = arith.constant 2: i32
      %t2 = arith.muli %x, %cst2: i32
      memref.store %t2, %o1[%i] : memref<512xi32>
    }
    cnm.terminator
  }
   cnm.compute
  ins(%A[#map]: memref<2x512xi32>)
  outs(%O[#map]: memref<2x512xi32>)
  on hierarchy<2>
  do (%a1: memref<512xi32>,
      %o1: memref<512xi32>)  {
    affine.parallel (%i) = (0) to (512) {
      %x = memref.load %a1[%i]: memref<512xi32>
      %cst2 = arith.constant 2: i32
      %t2 = arith.muli %x, %cst2: i32
      memref.store %t2, %o1[%i] : memref<512xi32>
    }
    cnm.terminator
  }
  return

}

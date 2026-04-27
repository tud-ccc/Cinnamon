// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm --canonicalize %s | FileCheck %s

// CHECK-LABEL: @mm_dimm8_opt

func.func @mm_dimm8_opt(%arg0: tensor<16x1024xi32>, %arg1: tensor<1024x64xi32>) -> tensor<16x64xi32> {
    %0 = cinm.compute (%arg2 = %arg0 : tensor<16x1024xi32>, %arg3 = %arg1 : tensor<1024x64xi32>) -> tensor<16x64xi32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 8, 128, 1>} {
      %cst = arith.constant dense<0> : tensor<16x64xi32>
      %1 = cinm.op.gemm %arg2, %arg3 plus %cst {cinm.notile} : tensor<16x1024xi32>, tensor<1024x64xi32> plus tensor<16x64xi32> -> tensor<16x64xi32>
      cinm.yield %1 : tensor<16x64xi32>
    }
    return %0 : tensor<16x64xi32>
  }
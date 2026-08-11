// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm --canonicalize %s | FileCheck %s

#upmem_platform = #upmem.platform<type=v1A, dimensions = 8x128>
#upmem = #upmem.array<8x128x1, #upmem_platform>

// CHECK-LABEL: @mm_dimm8_opt

func.func @mm_dimm8_opt(%arg0: tensor<16x1024xi32>, %arg1: tensor<1024x64xi32>) -> tensor<16x64xi32> {
    %0 = cinm.compute on accelerator #upmem -> tensor<16x64xi32> {
      %cst = arith.constant dense<0> : tensor<16x64xi32>
      %1 = cinm.op.gemm %arg0, %arg1 plus %cst : tensor<16x1024xi32>, tensor<1024x64xi32> plus tensor<16x64xi32> -> tensor<16x64xi32>
      cinm.yield %1 : tensor<16x64xi32>
    }
    return %0 : tensor<16x64xi32>
  }

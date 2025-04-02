"func.func"() <{function_type = (tensor<8x1024xi32>, tensor<1024x128xi32>) -> tensor<8x128xi32>, sym_name = "mm_dimm8_nopt"}> ({
^bb0(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024x128xi32>):
  %0 = "cnm.workgroup"() : () -> !cnm.workgroup<8x128x1>
  %1 = "tensor.empty"() : () -> tensor<128x1024xi32>
  %2 = "linalg.transpose"(%arg1, %1) <{permutation = array<i64: 1, 0>}> ({
  ^bb0(%arg8: i32, %arg9: i32):
    "linalg.yield"(%arg8) : (i32) -> ()
  }) : (tensor<1024x128xi32>, tensor<128x1024xi32>) -> tensor<128x1024xi32>
  %3 = "cnm.alloc"(%0) : (!cnm.workgroup<8x128x1>) -> !cnm.buffer<1024xi32 on 8x128x1, level 0>
  %4 = "cnm.alloc"(%0) : (!cnm.workgroup<8x128x1>) -> !cnm.buffer<1024xi32 on 8x128x1, level 0>
  %5 = "cnm.alloc"(%0) : (!cnm.workgroup<8x128x1>) -> !cnm.buffer<i32 on 8x128x1, level 0>
  "cnm.scatter"(%arg0, %3, %0) <{scatterMap = affine_map<(d0, d1, d2) -> (d1 mod 8)>}> : (tensor<8x1024xi32>, !cnm.buffer<1024xi32 on 8x128x1, level 0>, !cnm.workgroup<8x128x1>) -> ()
  "cnm.scatter"(%2, %4, %0) <{scatterMap = affine_map<(d0, d1, d2) -> (d1)>}> : (tensor<128x1024xi32>, !cnm.buffer<1024xi32 on 8x128x1, level 0>, !cnm.workgroup<8x128x1>) -> ()
  %6 = "arith.constant"() <{value = dense<0> : tensor<8x128xi32>}> : () -> tensor<8x128xi32>
  "cnm.scatter"(%6, %5, %0) <{scatterMap = affine_map<(d0, d1, d2) -> (d0, d1)>}> : (tensor<8x128xi32>, !cnm.buffer<i32 on 8x128x1, level 0>, !cnm.workgroup<8x128x1>) -> ()
  "cnm.launch"(%0, %3, %4, %5) <{operandSegmentSizes = array<i32: 1, 2, 1>}> ({
  ^bb0(%arg2: memref<1024xi32>, %arg3: memref<1024xi32>, %arg4: memref<i32>):
    "linalg.contract"(%arg2, %arg3, %arg4) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(a) -> ()>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg5: i32, %arg6: i32, %arg7: i32):
      %9 = "arith.muli"(%arg5, %arg6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %10 = "arith.addi"(%arg7, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%10) : (i32) -> ()
    }) : (memref<1024xi32>, memref<1024xi32>, memref<i32>) -> ()
    "cnm.terminator"() : () -> ()
  }) : (!cnm.workgroup<8x128x1>, !cnm.buffer<1024xi32 on 8x128x1, level 0>, !cnm.buffer<1024xi32 on 8x128x1, level 0>, !cnm.buffer<i32 on 8x128x1, level 0>) -> ()
  %7 = "tensor.empty"() : () -> tensor<8x128xi32>
  %8 = "cnm.gather"(%5, %0, %7) <{gatherMap = affine_map<(d0, d1, d2) -> (d0, d1)>}> : (!cnm.buffer<i32 on 8x128x1, level 0>, !cnm.workgroup<8x128x1>, tensor<8x128xi32>) -> tensor<8x128xi32>
  "cnm.free_workgroup"(%0) : (!cnm.workgroup<8x128x1>) -> ()
  "func.return"(%8) : (tensor<8x128xi32>) -> ()
}) : () -> ()
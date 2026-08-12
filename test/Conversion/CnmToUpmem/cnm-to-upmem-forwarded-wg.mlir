// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize --cse --canonicalize --convert-cnm-to-upmem | FileCheck %s

// A workgroup allocated OUTSIDE the function and forwarded in as a block
// argument (the whole-program schedule: one alloc per group, at the top of
// the container function). The lowering must recognize it through
// cnm::CnmWorkgroupTypeInterface and use it instead of allocating a fresh
// set; the program load stays here (which binary the set holds is this
// code's decision), and the set is NOT freed -- it is not this function's
// to release.

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>

#map31 = affine_map<(d0) -> (d0)>
#map41 = affine_map<(d0) -> ()>

#upmem_platform = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#upmem_1_16_1 = #upmem.array<16x1, #upmem_platform>

// CHECK-LABEL: func.func @forwarded
// CHECK-SAME: %[[DPU:[^:]*]]: !upmem.hierarchy<16x1>
// CHECK-NOT: upmem.alloc_dpus
// CHECK: upmem.load_program @dpu_kernels::@program on %[[DPU]] : !upmem.hierarchy<16x1>
// CHECK-NOT: upmem.alloc_dpus
// CHECK: upmem.scatter{{.*}} of %[[DPU]]
// CHECK: upmem.wait_for %[[DPU]] : !upmem.hierarchy<16x1>
// CHECK: upmem.gather{{.*}} of %[[DPU]]
// CHECK-NOT: upmem.free_dpus
func.func @forwarded(%dpus: !upmem.hierarchy<16x1>) -> tensor<16x1xi32> {
  %cst = arith.constant dense<0> : tensor<16x1xi32>
  %a = arith.constant dense<1> : tensor<16x64xi32>
  %b = arith.constant dense<2> : tensor<1x64xi32>
  %1 = cnm.workgroup : !cnm.workgroup<#upmem_1_16_1>
  %5 = cnm.declare_buffer() for %1 : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  %6 = cnm.declare_buffer() for %1 : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  %7 = cnm.declare_buffer() for %1 : !cnm.buffer<i32 on #upmem_1_16_1, #upmem.wram>
  cnm.scatter %a into %5[#map] of %1 : tensor<16x64xi32> into !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  cnm.scatter %b into %6[#map1] of %1 : tensor<1x64xi32> into !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  cnm.scatter %cst into %7[#map2] of %1 : tensor<16x1xi32> into !cnm.buffer<i32 on #upmem_1_16_1, #upmem.wram>
  cnm.launch %1 ins(%arg4 = %5 : <64xi32, #upmem.wram>, %arg5 = %6 : <64xi32, #upmem.wram>) outs(%arg6 = %7 : <i32, #upmem.wram>) on !cnm.workgroup<#upmem_1_16_1> {
    linalg.contract indexing_maps = [#map31, #map31, #map41] ins(%arg4, %arg5 : memref<64xi32, #upmem.wram>, memref<64xi32, #upmem.wram>) outs(%arg6 : memref<i32, #upmem.wram>)
  }
  %out = tensor.empty(): tensor<16x1xi32>
  %8 = cnm.gather %7[#map2] of %1 into %out : !cnm.buffer<i32 on #upmem_1_16_1, #upmem.wram> into tensor<16x1xi32>
  cnm.free_workgroup %1 : !cnm.workgroup<#upmem_1_16_1>
  return %8 : tensor<16x1xi32>
}

// -----

// The same body with NO forwarded workgroup in scope still allocates and
// frees its own set: forwarding is opt-in by putting a matching
// workgroup-typed argument in scope, and a mismatched shape does not match
// (this function's argument is 32x1, the workgroup is 16x1).

// CHECK-LABEL: func.func @not_forwarded_shape_mismatch
// CHECK: %[[LOCAL:.*]] = upmem.alloc_dpus : !upmem.hierarchy<16x1>
// CHECK: upmem.load_program @dpu_kernels::@program{{.*}} on %[[LOCAL]] : !upmem.hierarchy<16x1>
// CHECK: upmem.free_dpus %[[LOCAL]] : !upmem.hierarchy<16x1>
func.func @not_forwarded_shape_mismatch(%other: !upmem.hierarchy<32x1>) -> tensor<16x1xi32> {
  %cst = arith.constant dense<0> : tensor<16x1xi32>
  %a = arith.constant dense<1> : tensor<16x64xi32>
  %b = arith.constant dense<2> : tensor<1x64xi32>
  %1 = cnm.workgroup : !cnm.workgroup<#upmem_1_16_1>
  %5 = cnm.declare_buffer() for %1 : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  %6 = cnm.declare_buffer() for %1 : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  %7 = cnm.declare_buffer() for %1 : !cnm.buffer<i32 on #upmem_1_16_1, #upmem.wram>
  cnm.scatter %a into %5[#map] of %1 : tensor<16x64xi32> into !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  cnm.scatter %b into %6[#map1] of %1 : tensor<1x64xi32> into !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.wram>
  cnm.scatter %cst into %7[#map2] of %1 : tensor<16x1xi32> into !cnm.buffer<i32 on #upmem_1_16_1, #upmem.wram>
  cnm.launch %1 ins(%arg4 = %5 : <64xi32, #upmem.wram>, %arg5 = %6 : <64xi32, #upmem.wram>) outs(%arg6 = %7 : <i32, #upmem.wram>) on !cnm.workgroup<#upmem_1_16_1> {
    linalg.contract indexing_maps = [#map31, #map31, #map41] ins(%arg4, %arg5 : memref<64xi32, #upmem.wram>, memref<64xi32, #upmem.wram>) outs(%arg6 : memref<i32, #upmem.wram>)
  }
  %out = tensor.empty(): tensor<16x1xi32>
  %8 = cnm.gather %7[#map2] of %1 into %out : !cnm.buffer<i32 on #upmem_1_16_1, #upmem.wram> into tensor<16x1xi32>
  cnm.free_workgroup %1 : !cnm.workgroup<#upmem_1_16_1>
  return %8 : tensor<16x1xi32>
}

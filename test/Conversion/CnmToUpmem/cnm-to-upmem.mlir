// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize --cse --canonicalize --convert-cnm-to-upmem | FileCheck %s

// CHECK-DAG: #[[MAP:[^ ]*]] = affine_map<(d0, d1) -> (d1, 0)>
// CHECK-DAG: #[[MAP1:[^ ]*]] = affine_map<(d0, d1) -> (0, 0)>

// CHECK: memref.global "private" constant @__constant_16x1xi32 : memref<16x1xi32> = dense<0>
// CHECK-LABEL: func.func @main
// CHECK: %[[CST:.*]] = memref.get_global @__constant_16x1xi32 : memref<16x1xi32>
// CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<64x64xi32>
// CHECK: %[[DPU:.*]] = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<1x16x1>
// CHECK: affine.for %[[I:.*]] = 0 to 64 step 16 {
// CHECK: affine.for %[[J:.*]] = 0 to 64 {
// CHECK: %[[SV_A:.*]] = memref.subview %[[ALLOC]][%[[I]], 0] [16, 64] [1, 1] : memref<64x64xi32> to memref<16x64xi32, {{.*}}>
// CHECK: %[[SV_B:.*]] = memref.subview %[[ALLOC]][0, %[[J]]] [64, 1] [1, 1] : memref<64x64xi32> to memref<64x1xi32, {{.*}}>
// CHECK: %[[ALLOC_T:.*]] = memref.alloc() {{.*}} : memref<1x64xi32>
// CHECK: linalg.transpose ins(%[[SV_B]] : memref<64x1xi32, {{.*}}>) outs(%[[ALLOC_T]] : memref<1x64xi32>) permutation = [1, 0]
// CHECK: upmem.scatter %[[SV_A]][64, #[[MAP]]] onto @buf_1 of %[[DPU]] : memref<16x64xi32, {{.*}}> onto !upmem.hierarchy<1x16x1>
// CHECK: upmem.scatter %[[ALLOC_T]][64, #[[MAP1]]] onto @buf_0 of %[[DPU]] : memref<1x64xi32> onto !upmem.hierarchy<1x16x1>
// CHECK: upmem.scatter %[[CST]][1, #[[MAP]]] onto @buf of %[[DPU]] : memref<16x1xi32> onto !upmem.hierarchy<1x16x1>
// CHECK: upmem.wait_for %[[DPU]] : !upmem.hierarchy<1x16x1>
// CHECK: %[[SV_OUT:.*]] = memref.subview %[[ALLOC]][%[[I]], %[[J]]] [16, 1] [1, 1] : memref<64x64xi32> to memref<16x1xi32, {{.*}}>
// CHECK: upmem.gather %[[SV_OUT]][1, #[[MAP]]] from @buf of %[[DPU]] : memref<16x1xi32, {{.*}}> from !upmem.hierarchy<1x16x1>
// CHECK: upmem.free_dpus %[[DPU]] : !upmem.hierarchy<1x16x1>
// CHECK: module @dpu_kernels
// CHECK: upmem.dpu_program @program() tasklets(1) {
// CHECK: %[[WRAM_C:.*]] = pwram_alloc() : memref<i32, "wram">
// CHECK: %[[MRAM_C:.*]] = static_alloc @buf(mram) : memref<1xi32, "mram">
// CHECK: %[[WRAM_B:.*]] = pwram_alloc() : memref<64xi32, "wram">
// CHECK: %[[MRAM_B:.*]] = static_alloc @buf_0(mram) : memref<1x64xi32, "mram">
// CHECK: %[[WRAM_A:.*]] = pwram_alloc() : memref<64xi32, "wram">
// CHECK: %[[MRAM_A:.*]] = static_alloc @buf_1(mram) : memref<1x64xi32, "mram">
// CHECK: %[[T0:.*]] = tasklet_dim()
// CHECK: %[[SV0:.*]] = memref.subview %[[MRAM_C]][%[[T0]]] [1] [1] : memref<1xi32, "mram"> to memref<i32, {{.*}}, "mram">
// CHECK: local_transfer %[[SV0]] into %[[WRAM_C]] : memref<i32, {{.*}}, "mram"> to memref<i32, "wram">
// CHECK: %[[T1:.*]] = tasklet_dim()
// CHECK: %[[SV1:.*]] = memref.subview %[[MRAM_B]][%[[T1]], 0] [1, 64] [1, 1] : memref<1x64xi32, "mram"> to memref<64xi32, {{.*}}, "mram">
// CHECK: local_transfer %[[SV1]] into %[[WRAM_B]] : memref<64xi32, {{.*}}, "mram"> to memref<64xi32, "wram">
// CHECK: %[[T2:.*]] = tasklet_dim()
// CHECK: %[[SV2:.*]] = memref.subview %[[MRAM_A]][%[[T2]], 0] [1, 64] [1, 1] : memref<1x64xi32, "mram"> to memref<64xi32, {{.*}}, "mram">
// CHECK: local_transfer %[[SV2]] into %[[WRAM_A]] : memref<64xi32, {{.*}}, "mram"> to memref<64xi32, "wram">
// CHECK: linalg.reduce ins(%[[WRAM_A]], %[[WRAM_B]] : memref<64xi32, "wram">, memref<64xi32, "wram">) outs(%[[WRAM_C]] : memref<i32, "wram">) dimensions = [0]
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: %[[T3:.*]] = tasklet_dim()
// CHECK: %[[SV3:.*]] = memref.subview %[[MRAM_C]][%[[T3]]] [1] [1] : memref<1xi32, "mram"> to memref<i32, {{.*}}, "mram">
// CHECK: local_transfer %[[WRAM_C]] into %[[SV3]] : memref<i32, "wram"> to memref<i32, {{.*}}, "mram">

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, 0)>

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem_1_16_1 = #upmem.array<1x16x1, #upmem_platform>

module {
  func.func @main() {
    %cst = arith.constant dense<0> : tensor<16x1xi32>
    %0 = tensor.empty() : tensor<64x64xi32>
    %1 = cnm.workgroup : !cnm.workgroup<#upmem_1_16_1>

    %2 = affine.for %arg0 = 0 to 64 step 16 iter_args(%arg1 = %0) -> (tensor<64x64xi32>) {
      %3 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %arg1) -> (tensor<64x64xi32>) {
        %extracted_slice = tensor.extract_slice %0[%arg0, 0] [16, 64] [1, 1] : tensor<64x64xi32> to tensor<16x64xi32>
        %extracted_slice_0 = tensor.extract_slice %0[0, %arg2] [64, 1] [1, 1] : tensor<64x64xi32> to tensor<64x1xi32>
        %4 = tensor.empty() : tensor<1x64xi32>
        %transposed = linalg.transpose ins(%extracted_slice_0 : tensor<64x1xi32>) outs(%4 : tensor<1x64xi32>) permutation = [1, 0]
        %5 = cnm.alloc() for %1 : !cnm.buffer<64xi32 on #upmem_1_16_1, "wram">
        %6 = cnm.alloc() for %1 : !cnm.buffer<64xi32 on #upmem_1_16_1, "wram">
        %7 = cnm.alloc() for %1 : !cnm.buffer<i32 on #upmem_1_16_1, "wram">
        cnm.scatter %extracted_slice into %5[#map] of %1 : tensor<16x64xi32> into !cnm.buffer<64xi32 on #upmem_1_16_1, "wram">
        cnm.scatter %transposed into %6[#map1] of %1 : tensor<1x64xi32> into !cnm.buffer<64xi32 on #upmem_1_16_1, "wram">
        cnm.scatter %cst into %7[#map2] of %1 : tensor<16x1xi32> into !cnm.buffer<i32 on #upmem_1_16_1, "wram">
        cnm.launch %1 ins(%arg4 = %5 : <64xi32, "wram">, %arg5 = %6 : <64xi32, "wram">) outs(%arg6 = %7 : <i32, "wram">) on !cnm.workgroup<#upmem_1_16_1> {
          linalg.reduce ins(%arg4, %arg5 : memref<64xi32, "wram">, memref<64xi32, "wram">) outs(%arg6 : memref<i32, "wram">) dimensions = [0]
            (%in: i32, %in_1: i32, %init: i32) {
              %9 = arith.muli %in, %in_1 : i32
              %10 = arith.addi %9, %init : i32
              linalg.yield %10 : i32
            }
        }
        %out = tensor.empty(): tensor<16x1xi32>
        %8 = cnm.gather %7[#map2] of %1 into %out : !cnm.buffer<i32 on #upmem_1_16_1, "wram"> into tensor<16x1xi32>
        %inserted_slice = tensor.insert_slice %8 into %arg3[%arg0, %arg2] [16, 1] [1, 1] : tensor<16x1xi32> into tensor<64x64xi32>
        affine.yield %inserted_slice : tensor<64x64xi32>
      }
      affine.yield %3 : tensor<64x64xi32>
    }
    cnm.free_workgroup %1 : !cnm.workgroup<#upmem_1_16_1>
    return
  }
}

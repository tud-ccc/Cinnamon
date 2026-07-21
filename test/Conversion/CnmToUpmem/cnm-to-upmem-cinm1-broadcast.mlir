// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem=cinm1-codegen=true | FileCheck %s

// Under cinm1-codegen, WRAM is never shared across tasklets (each tasklet
// gets a private buffer). But %b's scatter map doesn't depend on the thread
// dim (d2), so the MRAM buffer for it should still be a single shared copy
// (no per-tasklet leading dimension, no numTasklets multiplier on
// transferCount) -- every tasklet just loads that same MRAM location into
// its own private WRAM buffer. %a's scatter map does depend on the thread
// dim, so it keeps the usual per-tasklet MRAM layout.

// CHECK-DAG: #[[MAPA:[^ ]*]] = affine_map<(d0, d1) -> (d1, 0, 0)>
// CHECK-DAG: #[[MAPB:[^ ]*]] = affine_map<(d0, d1) -> (d1, 0)>

// CHECK-LABEL: func.func @main
// CHECK: upmem.scatter %{{.*}}[16, #[[MAPA]]] onto @buf_0 of %[[DPU:.*]] : memref<4x2x8xi32> onto !upmem.hierarchy<1x4x2>
// CHECK: upmem.scatter %{{.*}}[8, #[[MAPB]]] onto @buf of %[[DPU]] : memref<4x8xi32> onto !upmem.hierarchy<1x4x2>

// CHECK: module @dpu_kernels
// CHECK: upmem.dpu_program @program() tasklets(2) {
// CHECK: %[[PWRAM_B:.*]] = upmem.pwram_alloc() : memref<8xi32, #upmem.wram>
// CHECK: %[[MRAM_B:.*]] = upmem.static_alloc @buf(mram) : memref<8xi32, #upmem.mram>
// CHECK: %[[PWRAM_A:.*]] = upmem.pwram_alloc() : memref<8xi32, #upmem.wram>
// CHECK: %[[MRAM_A:.*]] = upmem.static_alloc @buf_0(mram) : memref<2x8xi32, #upmem.mram>
// CHECK: upmem.local_transfer %[[MRAM_B]] into %[[PWRAM_B]] : memref<8xi32, #upmem.mram> to memref<8xi32, #upmem.wram>
// CHECK-NOT: scf.if
// CHECK: %[[T1:.*]] = upmem.tasklet_dim()
// CHECK: %[[SVA:.*]] = memref.subview %[[MRAM_A]][%[[T1]], 0]
// CHECK: upmem.local_transfer %[[SVA]] into %[[PWRAM_A]]
// CHECK-NOT: scf.if
// CHECK-NOT: upmem.barrier

#mapA = affine_map<(d0, d1, d2) -> (d1, d2)>
#mapB = affine_map<(d0, d1, d2) -> (d1)>

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem_1_4_2 = #upmem.array<1x4x2, #upmem_platform>

module {
  func.func @main(%a: tensor<4x2x8xi32>, %b: tensor<4x8xi32>) {
    %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_4_2>
    %bufA = cnm.alloc() for %wg : !cnm.buffer<8xi32 on #upmem_1_4_2>
    %bufB = cnm.alloc() for %wg : !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.scatter %a into %bufA[#mapA] of %wg : tensor<4x2x8xi32> into !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.scatter %b into %bufB[#mapB] of %wg : tensor<4x8xi32> into !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.launch %wg ins(%arg0 = %bufA : <8xi32>, %arg1 = %bufB : <8xi32>) on !cnm.workgroup<#upmem_1_4_2> {
    }
    cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_4_2>
    return
  }
}

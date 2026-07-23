// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem | FileCheck %s
// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem=cinm1-codegen=true | FileCheck %s --check-prefix=CINM1

// %c's scatter map is the empty map: it doesn't depend on (rank, dpu,
// tasklet) at all, so every DPU's tasklets all read byte-for-byte identical
// data straight from %c's own start. By default this lowers to
// upmem.broadcast -- one runtime call broadcasting the whole buffer to every
// DPU -- instead of a upmem.scatter whose affine map always happens to
// return the same (zero) offset.
//
// %d's scatter map is tasklet-broadcast too (dim d2 unused) but still
// depends on the dpu dim d1: different DPUs must see different data, so
// upmem.broadcast (which has no affine map at all) would be wrong here. This
// must keep using upmem.scatter regardless of cinm1-codegen.

// CHECK-DAG: #[[MAPD:[^ ]*]] = affine_map<(d0, d1) -> (d1, 0)>

// CHECK-LABEL: func.func @main
// CHECK: upmem.broadcast %{{.*}} onto @buf_0 of %{{.*}} : memref<8xi32> onto !upmem.hierarchy<1x4x2>
// CHECK-LABEL: func.func @dpu_varying
// CHECK: upmem.scatter %{{.*}}[8 elts, #[[MAPD]]] onto @buf_0 of %{{.*}} : memref<4x8xi32> onto !upmem.hierarchy<1x4x2>

// Under cinm1-codegen, the upmem.broadcast shortcut is disabled -- the
// DPU-side code doesn't expect it -- so %c also falls back to the classic
// upmem.scatter (rank, dpu) form with an all-zero map, matching pre-existing
// behavior.
// CINM1-DAG: #[[MAPZ:[^ ]*]] = affine_map<(d0, d1) -> (0)>
// CINM1-LABEL: func.func @main
// CINM1: upmem.scatter %{{.*}}[8 elts, #[[MAPZ]]] onto @buf of %{{.*}} : memref<8xi32> onto !upmem.hierarchy<1x4x2>

#mapC = affine_map<(d0, d1, d2) -> ()>
#mapD = affine_map<(d0, d1, d2) -> (d1)>

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem_1_4_2 = #upmem.array<1x4x2, #upmem_platform>

module {
  func.func @main(%c: tensor<8xi32>) {
    %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_4_2>
    %bufC = cnm.alloc() for %wg : !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.scatter %c into %bufC[#mapC] of %wg : tensor<8xi32> into !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.launch %wg ins(%arg0 = %bufC : <8xi32>) on !cnm.workgroup<#upmem_1_4_2> {
    }
    cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_4_2>
    return
  }

  func.func @dpu_varying(%d: tensor<4x8xi32>) {
    %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_4_2>
    %bufD = cnm.alloc() for %wg : !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.scatter %d into %bufD[#mapD] of %wg : tensor<4x8xi32> into !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.launch %wg ins(%arg0 = %bufD : <8xi32>) on !cnm.workgroup<#upmem_1_4_2> {
    }
    cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_4_2>
    return
  }
}

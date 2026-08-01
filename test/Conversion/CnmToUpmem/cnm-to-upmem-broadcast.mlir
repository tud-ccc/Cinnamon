// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem | FileCheck %s
// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem=use-bc-xfer-codegen=false | FileCheck %s --check-prefix=NOBC
// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem=cinm1-codegen=true | FileCheck %s --check-prefix=CINM1

// %c's scatter map is the empty map: it doesn't depend on (rank, dpu,
// tasklet) at all, so every DPU's tasklets all read byte-for-byte identical
// data straight from %c's own start. By default this lowers to
// upmem.broadcast -- one runtime call broadcasting the whole buffer to every
// DPU -- instead of a upmem.scatter_on_array whose affine map always happens to
// return the same (zero) offset.
//
// %d's scatter map is tasklet-broadcast too (dim d2 unused) but still
// depends on the dpu dim d1: different DPUs must see different data, so
// upmem.broadcast (which has no affine map at all) would be wrong here. This
// must keep using upmem.scatter_on_array regardless of cinm1-codegen.

// CHECK-DAG: #[[MAPD:[^ ]*]] = affine_map<(d0, d1) -> (d1, 0)>

// CHECK-LABEL: func.func @main
// CHECK: upmem.broadcast %{{.*}} onto @buf_0 of %{{.*}} : memref<8xi32> onto !upmem.hierarchy<1x4x2>
// CHECK-LABEL: func.func @dpu_varying
// CHECK: upmem.scatter_on_array %{{.*}}[8 elts, #[[MAPD]]] onto @buf_0 of %{{.*}} : memref<4x8xi32> onto !upmem.hierarchy<1x4x2>

// use-bc-xfer-codegen=false turns the shortcut off, so %c falls back to the
// classic upmem.scatter_on_array (rank, dpu) form with an all-zero map.
// NOBC-DAG: #[[MAPZ:[^ ]*]] = affine_map<(d0, d1) -> (0)>
// NOBC-LABEL: func.func @main
// NOBC: upmem.scatter_on_array %{{.*}}[8 elts, #[[MAPZ]]] onto @buf_0 of %{{.*}} : memref<8xi32> onto !upmem.hierarchy<1x4x2>

// cinm1-codegen does *not* affect the broadcast shortcut -- it selects the
// DPU-side codegen style: no WRAM sharing between tasklets (a private
// pwram_alloc per tasklet rather than a shared wram static_alloc) and no
// `noinit` on the MRAM allocation.
// CINM1-LABEL: func.func @main
// CINM1: upmem.broadcast %{{.*}} onto @buf of %{{.*}} : memref<8xi32> onto !upmem.hierarchy<1x4x2>
// CINM1: upmem.dpu_program @program() tasklets(2) {
// CINM1: memref.alloca() : memref<8xi32, #upmem.wram>
// CINM1: upmem.static_alloc @buf(mram) : memref<8xi32, #upmem.mram>

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

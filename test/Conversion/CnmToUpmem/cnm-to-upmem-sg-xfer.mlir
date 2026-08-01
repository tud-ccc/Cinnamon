// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem | FileCheck %s
// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem=use-sg-xfer-codegen=false | FileCheck %s --check-prefix=NOSG

// %a's scatter map addresses tasklet t's block at row 2*t of the host
// buffer (instead of row t), so consecutive tasklets' blocks are not
// contiguous in the host buffer (there is a gap of one unused row between
// them). By default (use-sg-xfer-codegen=true, the default) this should be
// lowered to the upmem.scatter_blocks form -- keeping the block dim in the
// scatter map and using a transferCount of just one block -- instead of
// forcing a flat, incorrect memcpy.

// CHECK-DAG: #[[MAPA3:[^ ]*]] = affine_map<(d0, d1, d2) -> (d1, d2 * 2, 0)>

// CHECK-LABEL: func.func @main
// CHECK: upmem.scatter_blocks %{{.*}}[8 elts, #[[MAPA3]], 2 blocks] onto @buf of %[[DPU:.*]] : memref<4x3x8xi32> onto !upmem.hierarchy<1x4x2>

// With use-sg-xfer-codegen=false, the pass falls back to unconditionally
// collapsing to the (rank, dpu) form (transferCount = 16, both tasklets'
// worth), matching the pre-existing behavior from before this codegen
// strategy was added. That collapse reads rows 0 and 1 where the map asks for
// rows 0 and 2, i.e. the wrong bytes: it is pinned here because it is the
// CINM 1.0 measurement baseline, and it is what --upmem-specialize-transfers
// is meant to make unreachable (design K3).
// NOSG-DAG: #[[MAPA2:[^ ]*]] = affine_map<(d0, d1) -> (d1, 0, 0)>
// NOSG-LABEL: func.func @main
// NOSG: upmem.scatter_on_array %{{.*}}[16 elts, #[[MAPA2]]] onto @buf of %{{.*}} : memref<4x3x8xi32> onto !upmem.hierarchy<1x4x2>

#mapA = affine_map<(d0, d1, d2) -> (d1, d2 * 2)>

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem_1_4_2 = #upmem.array<1x4x2, #upmem_platform>

module {
  func.func @main(%a: tensor<4x3x8xi32>) {
    %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_4_2>
    %bufA = cnm.alloc() for %wg : !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.scatter %a into %bufA[#mapA] of %wg : tensor<4x3x8xi32> into !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.launch %wg ins(%arg0 = %bufA : <8xi32>) on !cnm.workgroup<#upmem_1_4_2> {
    }
    cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_4_2>
    return
  }
}

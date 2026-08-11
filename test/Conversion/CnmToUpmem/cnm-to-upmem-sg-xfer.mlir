// RUN: cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem --upmem-specialize-transfers | FileCheck %s
// RUN: not cinm-opt %s --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --convert-cnm-to-upmem --upmem-specialize-transfers=use-sg-xfer-codegen=false 2>&1 | FileCheck %s --check-prefix=NOSG

// %a's scatter map addresses tasklet t's block at row 2*t of the host
// buffer (instead of row t), so consecutive tasklets' blocks are not
// contiguous in the host buffer (there is a gap of one unused row between
// them). The conversion emits the general block form and
// --upmem-specialize-transfers cannot narrow it: the two blocks are not one
// run, so it stays upmem.scatter_blocks and each block is fetched from where
// it really is.

// CHECK-DAG: #[[MAPA3:[^ ]*]] = affine_map<(d0, d1) -> (d0, d1 * 2, 0)>

// CHECK-LABEL: func.func @main
// CHECK: upmem.scatter_blocks %{{.*}}[8 elts, #[[MAPA3]], 2 blocks] onto @buf of %[[DPU:.*]] : memref<4x3x8xi32> onto !upmem.hierarchy<4x2>

// With use-sg-xfer-codegen=false the SDK's scatter transfer API is off, so
// this transfer has no legal form: the packing that would have made the two
// blocks one run was supposed to happen upstream
// (--cnm-ensure-scatter-gather-contiguous, which cinm1.py runs exactly when
// the flag is off). Before this was a separate pass, the conversion collapsed
// it anyway -- reading rows 0 and 1 where the map asks for rows 0 and 2 --
// and the wrong bytes went to the DPUs silently.
// NOSG: error: {{.*}}cannot be narrowed to a single per-DPU block

#mapA = affine_map<(d0, d1) -> (d0, d1 * 2)>

#upmem_platform = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#upmem_1_4_2 = #upmem.array<4x2, #upmem_platform>

module {
  func.func @main(%a: tensor<4x3x8xi32>) {
    %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_4_2>
    %bufA = cnm.declare_buffer() for %wg : !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.scatter %a into %bufA[#mapA] of %wg : tensor<4x3x8xi32> into !cnm.buffer<8xi32 on #upmem_1_4_2>
    cnm.launch %wg ins(%arg0 = %bufA : <8xi32>) on !cnm.workgroup<#upmem_1_4_2> {
    }
    cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_4_2>
    return
  }
}

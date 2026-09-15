// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s

#upmem = #upmem.platform<type = v1A, dpus = 4096, tasklets = 1>
#upmem_1_16_1 = #upmem.array<16x1, #upmem>

// cnm.local_transfer moves a block between memory levels of the device, so it
// lives inside a launch body and works on memrefs rather than !cnm.buffer.

// CHECK-LABEL: @stage_through_wram
func.func @stage_through_wram() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_16_1>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.mram>
  %out = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.mram>
  cnm.launch %wg ins(%a = %in : <64xi32, #upmem.mram>)
                 outs(%b = %out : <64xi32, #upmem.mram>)
                 on !cnm.workgroup<#upmem_1_16_1> {
    %wram = memref.alloc() : memref<16xi32, #upmem.wram>
    affine.for %i = 0 to 64 step 16 {
      // A tile of the MRAM buffer is addressed with an ordinary subview.
      %tile = memref.subview %a[%i] [16] [1] : memref<64xi32, #upmem.mram> to memref<16xi32, strided<[1], offset: ?>, #upmem.mram>
      // CHECK: cnm.local_transfer %{{.*}} into %{{.*}} : memref<16xi32, strided<[1], offset: ?>, #upmem.mram> to memref<16xi32, #upmem.wram>
      cnm.local_transfer %tile into %wram : memref<16xi32, strided<[1], offset: ?>, #upmem.mram> to memref<16xi32, #upmem.wram>

      %otile = memref.subview %b[%i] [16] [1] : memref<64xi32, #upmem.mram> to memref<16xi32, strided<[1], offset: ?>, #upmem.mram>
      // ... and back out again, in the other direction.
      // CHECK: cnm.local_transfer %{{.*}} into %{{.*}} : memref<16xi32, #upmem.wram> to memref<16xi32, strided<[1], offset: ?>, #upmem.mram>
      cnm.local_transfer %wram into %otile : memref<16xi32, #upmem.wram> to memref<16xi32, strided<[1], offset: ?>, #upmem.mram>
    }
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_16_1>
  return
}

// Nothing requires the two levels to differ, or either operand to carry a
// level at all -- the op is just a copy between memory spaces.
// CHECK-LABEL: @plain_memrefs
func.func @plain_memrefs(%a: memref<4x8xf32>, %b: memref<4x8xf32>) {
  // CHECK: cnm.local_transfer %{{.*}} into %{{.*}} : memref<4x8xf32> to memref<4x8xf32>
  cnm.local_transfer %a into %b : memref<4x8xf32> to memref<4x8xf32>
  return
}

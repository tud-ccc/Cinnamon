// RUN: cinm-opt %s --cnm-hoist-workgroups | FileCheck %s

#upmem_2_4_16 = #upmem.array<8x16, <type = v1A, dpus = 4096, tasklets = 1>>

// Workgroup and its allocs should be hoisted out of all loops, placed before
// the outermost loop. Allocs must immediately follow their workgroup.

// CHECK-LABEL: func.func @hoist_wg_and_allocs
// CHECK:       %[[WG:.*]] = cnm.workgroup
// CHECK-NEXT:  %[[BUF0:.*]] = cnm.declare_buffer() for %[[WG]]
// CHECK-NEXT:  %[[BUF1:.*]] = cnm.declare_buffer() for %[[WG]]
// CHECK-NEXT:  %[[BUF2:.*]] = cnm.declare_buffer() for %[[WG]]
// CHECK:       affine.for
// CHECK:         affine.for
// CHECK:           affine.for
// CHECK-NOT:         = cnm.workgroup
// CHECK-NOT:         cnm.declare_buffer
func.func @hoist_wg_and_allocs(%arg0: tensor<8x1024xi32>) -> tensor<8x2048xi32> {
  %out = tensor.empty() : tensor<8x2048xi32>
  %result = affine.for %i = 0 to 8 iter_args(%acc0 = %out) -> tensor<8x2048xi32> {
    %r0 = affine.for %j = 0 to 2048 step 128 iter_args(%acc1 = %acc0) -> tensor<8x2048xi32> {
      %r1 = affine.for %k = 0 to 1024 step 256 iter_args(%acc2 = %acc1) -> tensor<8x2048xi32> {
        %wg   = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
        %buf0 = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
        %buf1 = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
        %buf2 = cnm.declare_buffer() for %wg : !cnm.buffer<i32 on #upmem_2_4_16>
        cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
        affine.yield %acc2 : tensor<8x2048xi32>
      }
      affine.yield %r1 : tensor<8x2048xi32>
    }
    affine.yield %r0 : tensor<8x2048xi32>
  }
  return %result : tensor<8x2048xi32>
}

// Two independent workgroups in the same loop: each alloc must be placed
// immediately after its own workgroup, not interleaved with the other one.

// CHECK-LABEL: func.func @hoist_two_independent_wgs
// CHECK:       %[[WG0:.*]] = cnm.workgroup
// CHECK-NEXT:  %[[BUF_A:.*]] = cnm.declare_buffer() for %[[WG0]]
// CHECK:       %[[WG1:.*]] = cnm.workgroup
// CHECK-NEXT:  %[[BUF_B:.*]] = cnm.declare_buffer() for %[[WG1]]
// CHECK:       affine.for
// CHECK-NOT:     = cnm.workgroup
// CHECK-NOT:     cnm.declare_buffer
func.func @hoist_two_independent_wgs(%arg0: tensor<1x256xi32>) -> tensor<1x256xi32> {
  %out = tensor.empty() : tensor<1x256xi32>
  %result = affine.for %k = 0 to 1024 step 256 iter_args(%acc = %out) -> tensor<1x256xi32> {
    %wg0  = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
    %buf_a = cnm.declare_buffer() for %wg0 : !cnm.buffer<256xi32 on #upmem_2_4_16>
    %wg1  = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
    %buf_b = cnm.declare_buffer() for %wg1 : !cnm.buffer<256xi32 on #upmem_2_4_16>
    cnm.free_workgroup %wg0 : !cnm.workgroup<#upmem_2_4_16>
    cnm.free_workgroup %wg1 : !cnm.workgroup<#upmem_2_4_16>
    affine.yield %acc : tensor<1x256xi32>
  }
  return %result : tensor<1x256xi32>
}

// Workgroup already at function scope should not be moved.

// CHECK-LABEL: func.func @no_hoist_already_top_level
// CHECK:       %[[WG:.*]] = cnm.workgroup
// CHECK-NEXT:  %[[BUF:.*]] = cnm.declare_buffer() for %[[WG]]
// CHECK-NOT:   affine.for
// CHECK:       affine.for
func.func @no_hoist_already_top_level(%arg0: tensor<1x256xi32>) -> tensor<1x256xi32> {
  %wg  = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  %out = tensor.empty() : tensor<1x256xi32>
  %result = affine.for %k = 0 to 1024 step 256 iter_args(%acc = %out) -> tensor<1x256xi32> {
    affine.yield %acc : tensor<1x256xi32>
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return %result : tensor<1x256xi32>
}

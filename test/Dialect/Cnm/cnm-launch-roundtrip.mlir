// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s

// The shorthand buffer type printed by cnm.launch must carry the buffer's
// memory level, otherwise the printed IR does not parse back (the operand
// would be given a level-less buffer type conflicting with its definition).

#upmem = #upmem.platform<type = v1A, dpus = 4096, tasklets = 1>
#upmem_1_16_1 = #upmem.array<16x1, #upmem>

// CHECK-LABEL: @launch_mram_level
func.func @launch_mram_level() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_16_1>
  // CHECK: cnm.declare_buffer() for %{{.*}} : !cnm.buffer<64xi32 on #{{.*}}, #upmem.mram>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.mram>
  %out = cnm.declare_buffer() for %wg : !cnm.buffer<i32 on #upmem_1_16_1, #upmem.mram>
  // CHECK: cnm.launch %{{.*}} ins(%{{.*}} = %{{.*}} : <64xi32, #upmem.mram>) outs(%{{.*}} = %{{.*}} : <i32, #upmem.mram>)
  cnm.launch %wg ins(%a = %in : <64xi32, #upmem.mram>)
                 outs(%b = %out : <i32, #upmem.mram>)
                 on !cnm.workgroup<#upmem_1_16_1> {
    // CHECK: memref<64xi32, #upmem.mram>
    affine.for %i = 0 to 64 {
      %v = affine.load %a[%i] : memref<64xi32, #upmem.mram>
      %acc = affine.load %b[] : memref<i32, #upmem.mram>
      %sum = arith.addi %acc, %v : i32
      affine.store %sum, %b[] : memref<i32, #upmem.mram>
    }
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_16_1>
  return
}

// CHECK-LABEL: @launch_no_level
func.func @launch_no_level() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_16_1>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #upmem_1_16_1>
  %out = cnm.declare_buffer() for %wg : !cnm.buffer<i32 on #upmem_1_16_1>
  // A level-less buffer must keep printing without a trailing comma.
  // CHECK: cnm.launch %{{.*}} ins(%{{.*}} = %{{.*}} : <64xi32>) outs(%{{.*}} = %{{.*}} : <i32>)
  cnm.launch %wg ins(%a = %in : <64xi32>)
                 outs(%b = %out : <i32>)
                 on !cnm.workgroup<#upmem_1_16_1> {
    affine.for %i = 0 to 64 {
      %v = affine.load %a[%i] : memref<64xi32>
      %acc = affine.load %b[] : memref<i32>
      %sum = arith.addi %acc, %v : i32
      affine.store %sum, %b[] : memref<i32>
    }
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_16_1>
  return
}

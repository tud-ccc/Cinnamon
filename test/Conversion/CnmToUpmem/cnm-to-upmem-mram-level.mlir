// RUN: cinm-opt %s --split-input-file --convert-cnm-to-upmem | FileCheck %s

// An MRAM-level buffer means the launch body computes on MRAM directly and has
// already been given its own staging (--upmem-tile-mram-buffers). This pass
// must then bind the block argument straight to the MRAM allocation instead of
// wrapping the body in a second WRAM round-trip, and lower the body's staging
// ops to their UPMEM equivalents.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#scatterA = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#scatterY = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<4x2, #pf>

// CHECK-LABEL: func.func @gemv
func.func @gemv(%hostA: memref<8x16x64xi32>, %hostY: memref<8x16xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %a = cnm.declare_buffer() for %wg : !cnm.buffer<16x64xi32 on #acc, #upmem.mram>
  %y = cnm.declare_buffer() for %wg : !cnm.buffer<16xi32 on #acc, #upmem.mram>
  cnm.scatter %hostA into %a[#scatterA] of %wg : memref<8x16x64xi32> into !cnm.buffer<16x64xi32 on #acc, #upmem.mram>
  cnm.scatter %hostY into %y[#scatterY] of %wg : memref<8x16xi32> into !cnm.buffer<16xi32 on #acc, #upmem.mram>

  // The MRAM allocation carries the leading tasklet dimension, as usual.
  // CHECK: %[[MY:.*]] = upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<2x16xi32, #upmem.mram>
  // CHECK: %[[MA:.*]] = upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<2x16x64xi32, #upmem.mram>

  // No WRAM buffer is created for it and no transfer wraps the body: the block
  // argument is this tasklet's slice of the MRAM allocation.
  // CHECK: %[[T:.*]] = upmem.tasklet_dim()
  // CHECK: %[[VA:.*]] = memref.subview %[[MA]][%[[T]], 0, 0] [1, 16, 64] [1, 1, 1] : memref<2x16x64xi32, #upmem.mram> to memref<16x64xi32, {{.*}}, #upmem.mram>
  // CHECK: %[[VY:.*]] = memref.subview %[[MY]][%{{.*}}, 0] [1, 16] [1, 1] : memref<2x16xi32, #upmem.mram> to memref<16xi32, {{.*}}, #upmem.mram>

  // The body's own staging becomes UPMEM ops: memref.alloc in WRAM is a
  // per-tasklet allocation, and memref.dealloc has nothing to do because the
  // WRAM partition is reclaimed when the kernel returns.
  // CHECK: %[[W:.*]] = memref.alloca() : memref<16xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %[[VY]] into %[[W]]
  // CHECK: upmem.local_transfer %[[VA]] into %{{.*}}
  // CHECK-NOT: memref.alloc()
  // CHECK-NOT: memref.dealloc
  // CHECK-NOT: cnm.local_transfer
  cnm.launch %wg ins(%A = %a : <16x64xi32, #upmem.mram>)
                 outs(%Y = %y : <16xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    %w = memref.alloca() : memref<16xi32, #upmem.wram>
    cnm.local_transfer %Y into %w : memref<16xi32, #upmem.mram> to memref<16xi32, #upmem.wram>
    %wa = memref.alloca() : memref<16x64xi32, #upmem.wram>
    cnm.local_transfer %A into %wa : memref<16x64xi32, #upmem.mram> to memref<16x64xi32, #upmem.wram>
    %wx = memref.alloca() : memref<64xi32, #upmem.wram>
    linalg.contract indexing_maps = [#m, #v, #r]
      ins(%wa, %wx : memref<16x64xi32, #upmem.wram>, memref<64xi32, #upmem.wram>)
      outs(%w : memref<16xi32, #upmem.wram>)
    cnm.local_transfer %w into %Y : memref<16xi32, #upmem.wram> to memref<16xi32, #upmem.mram>
  }
  cnm.gather %y[#scatterY] of %wg into %hostY : !cnm.buffer<16xi32 on #acc, #upmem.mram> into memref<8x16xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}

// -----

#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<4x2, #pf>
#bcast = affine_map<(d0, d1) -> (d0)>

// When the scatter map does not depend on the tasklet dimension, every tasklet
// sees the same MRAM buffer: no leading tasklet dimension and nothing to slice.
// CHECK-LABEL: func.func @broadcast
func.func @broadcast(%host: memref<4x64xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  cnm.scatter %host into %b[#bcast] of %wg : memref<4x64xi32> into !cnm.buffer<64xi32 on #acc, #upmem.mram>
  // CHECK: %[[MB:.*]] = upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<64xi32, #upmem.mram>
  // CHECK-NOT: memref.subview
  // CHECK: %[[W:.*]] = memref.alloca() : memref<64xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %[[MB]] into %[[W]]
  cnm.launch %wg ins(%B = %b : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    %w = memref.alloca() : memref<64xi32, #upmem.wram>
    cnm.local_transfer %B into %w : memref<64xi32, #upmem.mram> to memref<64xi32, #upmem.wram>
    memref.dealloc %w : memref<64xi32, #upmem.wram>
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}

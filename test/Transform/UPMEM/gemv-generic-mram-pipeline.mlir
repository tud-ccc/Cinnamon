// RUN: cinm-opt %s \
// RUN:   --convert-cinm-to-cnm=cnm-buffer-level=mram --canonicalize --cse \
// RUN:   --eliminate-empty-tensors --one-shot-bufferize --cse --canonicalize \
// RUN:   --upmem-tile-mram-buffers=tile-sizes=16,128 --canonicalize --cse \
// RUN:   --cnm-ensure-scatter-gather-contiguous \
// RUN:   --convert-cnm-to-upmem \
// RUN: | FileCheck %s

// The whole generic path from a cinm op to an UPMEM DPU program with an
// MRAM/WRAM split, driven by pass flags only -- no inference plugin, no search.
// This is what the plugin's pipeline has to reproduce, so a failure here is a
// pipeline bug rather than a plugin bug.
//
// Pass ordering worth noting: staging runs *after* bufferization, because it
// works on memrefs, and --cnm-ensure-scatter-gather-contiguous runs before the
// backend conversion, because upmem.scatter requires each DPU's elements to be
// contiguous in the host buffer.

#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// CHECK-LABEL: func.func @gemv
func.func @gemv(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  // Host side: allocate the DPUs, scatter the operands, wait, gather back.
  // CHECK: %[[DPU:.*]] = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<1x16x1>
  // CHECK: upmem.scatter %{{.*}} onto @{{.*}} of %[[DPU]]
  // CHECK: upmem.wait_for %[[DPU]]
  // CHECK: upmem.gather %{{.*}} from @{{.*}} of %[[DPU]]
  // CHECK: upmem.free_dpus %[[DPU]]

  // Device side.
  // CHECK: upmem.dpu_program @program() tasklets(1) {

  // Every buffer is in MRAM, and nothing is staged around the kernel body.
  // CHECK-DAG: %[[MY:.*]] = upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<1x64xi32, #upmem.mram>
  // CHECK-DAG: %[[MX:.*]] = upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<512xi32, #upmem.mram>
  // CHECK-DAG: %[[MA:.*]] = upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<1x64x512xi32, #upmem.mram>

  // 1024 rows over 16 DPUs = 64 rows each; the block arguments are this
  // tasklet's slice of the MRAM allocations.
  // CHECK: %[[VA:.*]] = memref.subview %[[MA]]{{.*}} to memref<64x512xi32, {{.*}}, #upmem.mram>
  // CHECK: %[[VY:.*]] = memref.subview %[[MY]]{{.*}} to memref<64xi32, {{.*}}, #upmem.mram>

  // The 64x512 tile does not fit WRAM, so the body walks it in 16x128 tiles.
  // The output tile is staged once outside the reduction loop.
  // CHECK: scf.for
  // CHECK: %[[WY:.*]] = upmem.pwram_alloc() : memref<16xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WY]]
  // CHECK: scf.for
  // CHECK: %[[WA:.*]] = upmem.pwram_alloc() : memref<16x128xi32, #upmem.wram>
  // CHECK: %[[WX:.*]] = upmem.pwram_alloc() : memref<128xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WA]]
  // CHECK: upmem.local_transfer %{{.*}} into %[[WX]]
  // CHECK: linalg.contract {{.*}} ins(%[[WA]], %[[WX]] : memref<16x128xi32, #upmem.wram>, memref<128xi32, #upmem.wram>) outs(%[[WY]] : memref<16xi32, #upmem.wram>)
  // CHECK: }
  // CHECK: upmem.local_transfer %[[WY]] into %{{.*}} : memref<16xi32, #upmem.wram> to memref<16xi32, {{.*}}, #upmem.mram>
  // CHECK: upmem.return

  // Nothing from the middle of the stack should survive.
  // CHECK-NOT: cnm.
  // CHECK-NOT: cinm.op.
  %r0 = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %r = cinm.op.gemv %A, %x : tensor<1024x512xi32>, tensor<512xi32> -> tensor<1024xi32>
    cinm.yield %r : tensor<1024xi32>
  }
  func.return %r0 : tensor<1024xi32>
}

// RUN: cinm-opt %s \
// RUN:   --convert-linalg-to-cnm=cnm-buffer-level=mram --canonicalize --cse \
// RUN:   --eliminate-empty-tensors --one-shot-bufferize --cse --canonicalize \
// RUN:   --upmem-tile-mram-buffers --canonicalize --cse \
// RUN:   --cnm-ensure-scatter-gather-contiguous \
// RUN:   --convert-cnm-to-upmem \
// RUN: | FileCheck %s

// The generic path starting from a `linalg` op rather than a `cinm` op, which
// is where the pipeline is headed (design §G8: distribute linalg, so that
// fusion can run first). The companion test gemv-generic-mram-pipeline.mlir
// runs the same chain from `cinm.op.gemv` through `--convert-cinm-to-cnm`;
// both must reach the same DPU program.
//
// Two attributes drive the whole thing and nothing else does:
//   cnm.tile_sizes         -- block size per iteration dim, for the workgroup
//   upmem.leaf_tile_sizes  -- block size per iteration dim, for WRAM
// Note they are in the same unit (block sizes), which is the point of §G2.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// CHECK-LABEL: func.func @gemv
func.func @gemv(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>

  // Host side.
  // CHECK: %[[DPU:.*]] = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<1x16x1>
  // CHECK: upmem.scatter_on_array %{{.*}} onto @{{.*}} of %[[DPU]]
  // CHECK: upmem.wait_for %[[DPU]]
  // CHECK: upmem.gather_from_array %{{.*}} from @{{.*}} of %[[DPU]]
  // CHECK: upmem.free_dpus %[[DPU]]

  // Device side. 1024 rows over 16 leaves = 64 rows each; the 64x512 tile does
  // not fit WRAM, so the body walks it in 16x128 tiles with the output staged
  // once outside the reduction loop.
  // CHECK: upmem.dpu_program @program() tasklets(1) {
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<1x64xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<512xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<64x512xi32, #upmem.mram>

  // CHECK: scf.for
  // CHECK: %[[WY:.*]] = memref.alloca() : memref<16xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WY]]
  // CHECK: scf.for
  // CHECK: %[[WA:.*]] = memref.alloca() : memref<16x128xi32, #upmem.wram>
  // CHECK: %[[WX:.*]] = memref.alloca() : memref<128xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WA]]
  // CHECK: upmem.local_transfer %{{.*}} into %[[WX]]
  // CHECK: linalg.contract {{.*}} ins(%[[WA]], %[[WX]] : memref<16x128xi32, #upmem.wram>, memref<128xi32, #upmem.wram>) outs(%[[WY]] : memref<16xi32, #upmem.wram>)
  // CHECK: }
  // CHECK: upmem.local_transfer %[[WY]] into %{{.*}} : memref<16xi32, #upmem.wram> to memref<16xi32, {{.*}}, #upmem.mram>
  // CHECK: upmem.return

  // Nothing from the middle of the stack should survive.
  // CHECK-NOT: cnm.
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 64, 512>,
       upmem.leaf_tile_sizes = array<i64: 16, 128>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

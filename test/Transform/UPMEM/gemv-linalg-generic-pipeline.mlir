// RUN: cinm-opt %s --linalg-generalize-named-ops \
// RUN:   --convert-linalg-to-cnm='cnm-buffer-level=mram leaf-tile-attr=upmem.leaf_tile_sizes' \
// RUN:   --canonicalize --cse \
// RUN:   --eliminate-empty-tensors --one-shot-bufferize --cse --canonicalize \
// RUN:   --upmem-tile-mram-buffers --canonicalize --cse \
// RUN:   --cnm-ensure-scatter-gather-contiguous \
// RUN:   --convert-cnm-to-upmem --upmem-specialize-transfers \
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
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// CHECK-LABEL: func.func @gemv
func.func @gemv(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>

  // Host side. Both operands arrive as dynamically strided buffers, so each is
  // repacked into a contiguous one before its transfer -- a cnm.compact_buffer
  // rather than a memref.copy, so the cost stays attributable.
  //
  // The matrix goes out as scatter_blocks rather than one block per DPU: with
  // the leaf buffer laid out for the tile the body stages, a leaf's share is
  // no longer one run of the host matrix but one run per k-tile.
  // CHECK: %[[DPU:.*]] = upmem.alloc_dpus : !upmem.hierarchy<16x1>
  // CHECK: upmem.load_program @dpu_kernels::@program on %[[DPU]] : !upmem.hierarchy<16x1>
  // CHECK: cnm.compact_buffer %{{.*}} into %[[PA:.*]][#{{.*}}] : memref<1024x512xi32, strided<[?, ?], offset: ?>> into memref<1024x512xi32>
  // CHECK: upmem.scatter_blocks %{{.*}}[128 elts, #{{.*}}, 256 blocks] onto @{{.*}} of %[[DPU]]
  // CHECK: cnm.compact_buffer %{{.*}} into %{{.*}}[#{{.*}}] : memref<512xi32, strided<[?], offset: ?>> into memref<512xi32>
  // CHECK: upmem.broadcast %{{.*}} onto @{{.*}} of %[[DPU]]
  // CHECK: upmem.wait_for %[[DPU]]
  // CHECK: upmem.gather_from_array %{{.*}} from @{{.*}} of %[[DPU]]
  // CHECK: upmem.free_dpus %[[DPU]]

  // Device side. 1024 rows over 16 leaves = 64 rows each; the 64x512 tile does
  // not fit WRAM, so the body walks it in 16x128 tiles with the output staged
  // once outside the reduction loop.
  //
  // Both iteration dimensions are staged in chunks (64/16 and 512/128), so
  // each is cut in two and the chunk dimensions sit outermost in every buffer
  // that carries them: A is [m-chunk][k-chunk][m][k]. Same elements as the
  // 64x512 tile, ordered so that one staged 16x128 tile is a single run --
  // which is what upmem.local_transfer, being a DMA, requires.
  // CHECK: upmem.dpu_program @program() tasklets(1) {
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<1x4x16xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<4x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<4x4x16x128xi32, #upmem.mram>

  // CHECK: affine.for
  // CHECK: %[[WY:.*]] = memref.alloca() : memref<1x16xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WY]]
  // CHECK: affine.for
  // CHECK: %[[WA:.*]] = memref.alloca() : memref<1x1x16x128xi32, #upmem.wram>
  // CHECK: %[[WX:.*]] = memref.alloca() : memref<1x128xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WA]] : memref<1x1x16x128xi32, strided<[8192, 2048, 128, 1], offset: ?>, #upmem.mram> to memref<1x1x16x128xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WX]]
  // CHECK: linalg.generic {{.*}} ins(%[[WA]], %[[WX]] : memref<1x1x16x128xi32, #upmem.wram>, memref<1x128xi32, #upmem.wram>) outs(%[[WY]] : memref<1x16xi32, #upmem.wram>)
  // CHECK: }
  // CHECK: upmem.local_transfer %[[WY]] into %{{.*}} : memref<1x16xi32, #upmem.wram> to memref<1x16xi32, {{.*}}, #upmem.mram>
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

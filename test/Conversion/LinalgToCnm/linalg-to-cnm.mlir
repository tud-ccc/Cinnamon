// RUN: cinm-opt %s --split-input-file --convert-linalg-to-cnm | FileCheck %s
// RUN: cinm-opt %s --split-input-file --convert-linalg-to-cnm=cnm-buffer-level=mram | FileCheck %s --check-prefix=MRAM

// Distributing a linalg op onto a workgroup, driven only by `cnm.tile_sizes`
// (one *block size* per iteration dimension). The pass takes no decisions of
// its own: buffer shapes and scatter maps follow from the block sizes and the
// op's indexing maps.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// 1024x512 gemv, blocks 64x512: 16 tiles of m, 1 of k, and the workgroup has
// 1*16*1 = 16 leaves.
// CHECK-LABEL: func.func @gemv
// MRAM-LABEL: func.func @gemv
func.func @gemv(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>

  // CHECK: %[[WG:.*]] = cnm.workgroup : !cnm.workgroup<

  // The buffer shape is the block size of each dimension the operand indexes.
  // A is scattered as it stands: the map names, for every leaf and every
  // element of its buffer, the element of A it comes from. No relayout.
  // CHECK-NOT: linalg.transpose
  // CHECK-NOT: tensor.reshape
  // CHECK: %[[BA:.*]] = cnm.declare_buffer() for %[[WG]] : !cnm.buffer<64x512xi32 on
  // CHECK: cnm.scatter %arg0 into %[[BA]][#{{.*}}] of %[[WG]] {cinm.debug_tag = "dyn"} : tensor<1024x512xi32> into !cnm.buffer<64x512xi32

  // x is indexed only by the reduction dimension, which is not split, so every
  // leaf gets the same slice: a broadcast.
  // CHECK: %[[BX:.*]] = cnm.declare_buffer() for %[[WG]] : !cnm.buffer<512xi32 on
  // CHECK: cnm.scatter %arg1 into %[[BX]][#{{.*}}] of %[[WG]] {cinm.debug_tag = "dyn"} : tensor<512xi32> into !cnm.buffer<512xi32

  // The destination is a fresh tensor.empty, so its undefined contents are not
  // scattered: the alloc is followed straight by the launch.
  // CHECK: %[[BY:.*]] = cnm.declare_buffer() for %[[WG]] : !cnm.buffer<64xi32 on
  // CHECK-NOT: cnm.scatter

  // The body is the same op on leaf-sized memrefs, with its indexing maps
  // unchanged and the tile sizes consumed.
  // CHECK: cnm.launch %[[WG]] ins(%[[A:.*]] = %[[BA]] : <64x512xi32>, %[[X:.*]] = %[[BX]] : <512xi32>) outs(%[[Y:.*]] = %[[BY]] : <64xi32>)
  // CHECK: linalg.contract indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}] ins(%[[A]], %[[X]] : memref<64x512xi32>, memref<512xi32>) outs(%[[Y]] : memref<64xi32>)
  // CHECK-NOT: cnm.tile_sizes

  // CHECK: %[[G:.*]] = cnm.gather %[[BY]][#{{.*}}] of %[[WG]] into %{{.*}} : !cnm.buffer<64xi32{{.*}}> into tensor<1024xi32>
  // CHECK: cnm.free_workgroup %[[WG]]

  // With a level requested, it lands on both the buffer type and the launch
  // body's memrefs.
  // MRAM: cnm.declare_buffer() for %{{.*}} : !cnm.buffer<64x512xi32 on {{.*}}, #upmem.mram>
  // MRAM: linalg.contract {{.*}} ins(%{{.*}}, %{{.*}} : memref<64x512xi32, #upmem.mram>, memref<512xi32, #upmem.mram>) outs(%{{.*}} : memref<64xi32, #upmem.mram>)
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 64, 512>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// Not contract-specific: any linalg op with projected-permutation indexing
// maps distributes the same way, region and all.
// CHECK-LABEL: func.func @reduce
func.func @reduce(%A: tensor<1024x512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>
  // CHECK-DAG: cnm.declare_buffer() {{.*}} : !cnm.buffer<64x512xi32 on
  // CHECK-DAG: cnm.declare_buffer() {{.*}} : !cnm.buffer<64xi32 on
  // The payload region comes along.
  // CHECK: linalg.reduce ins(%{{.*}} : memref<64x512xi32>) outs(%{{.*}} : memref<64xi32>) dimensions = [1]
  // CHECK: arith.addi
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.reduce ins(%A : tensor<1024x512xi32>) outs(%init : tensor<1024xi32>)
      dimensions = [1]
      {cnm.tile_sizes = array<i64: 64, 512>}
      (%in: i32, %acc: i32) {
        %s = arith.addi %in, %acc : i32
        linalg.yield %s : i32
      }
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// An `outs` that is not a fresh tensor.empty is a real accumulator and has to
// be scattered in.
// CHECK-LABEL: func.func @accumulate
func.func @accumulate(%A: tensor<1024x512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: %[[BY:.*]] = cnm.declare_buffer() {{.*}} : !cnm.buffer<64xi32 on
  // CHECK: cnm.scatter %{{.*}} into %[[BY]]
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.reduce ins(%A : tensor<1024x512xi32>) outs(%y : tensor<1024xi32>)
      dimensions = [1]
      {cnm.tile_sizes = array<i64: 64, 512>}
      (%in: i32, %acc: i32) {
        %s = arith.addi %in, %acc : i32
        linalg.yield %s : i32
      }
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<4x4, #pf>

// Two parallel dimensions split across a 2-D workgroup. The tile-space order
// is parallel-outer/reduction-inner and both sides are linearized, so leaf
// `l` handles tile (l / 4, l % 4).
// CHECK-LABEL: func.func @elementwise_2d
func.func @elementwise_2d(%a: tensor<64x64xi32>, %b: tensor<64x64xi32>) -> tensor<64x64xi32> {
  %init = tensor.empty() : tensor<64x64xi32>
  // Both dimensions are tiled. Under the old shape-suffix contract this
  // needed a real transpose to present the tiles outermost; the map now says
  // where each element goes, so nothing moves on the host.
  // CHECK-NOT: linalg.transpose
  // CHECK-NOT: tensor.reshape
  // CHECK: cnm.declare_buffer() {{.*}} : !cnm.buffer<16x16xi32 on
  // CHECK: cnm.scatter %{{.*}} : tensor<64x64xi32> into !cnm.buffer<16x16xi32
  // CHECK: cnm.gather {{.*}} into tensor<64x64xi32>
  %r = cinm.compute on accelerator #acc -> tensor<64x64xi32> {
    %g = linalg.add {cnm.tile_sizes = array<i64: 16, 16>}
      ins(%a, %b : tensor<64x64xi32>, tensor<64x64xi32>)
      outs(%init : tensor<64x64xi32>) -> tensor<64x64xi32>
    cinm.yield %g : tensor<64x64xi32>
  }
  func.return %r : tensor<64x64xi32>
}

// -----

#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// An op without the attribute is left alone -- this pass only distributes
// what it has been given block sizes for.
// CHECK-LABEL: func.func @untouched
func.func @untouched(%A: tensor<1024x512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>
  // CHECK-NOT: cnm.workgroup
  // CHECK: linalg.reduce
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.reduce ins(%A : tensor<1024x512xi32>) outs(%init : tensor<1024xi32>)
      dimensions = [1]
      (%in: i32, %acc: i32) {
        %s = arith.addi %in, %acc : i32
        linalg.yield %s : i32
      }
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

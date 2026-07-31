// RUN: cinm-opt %s --split-input-file --convert-linalg-to-cnm=allow-float-reassociation=true | FileCheck %s

// Spreading a reduction dimension across the workgroup (design §G4). This is
// the configuration class the template flow expresses as dpuCols/taskletCols
// and that the old --convert-cinm-to-cnm could not represent at all: it only
// ever mapped *parallel* dimensions onto workgroup elements and gave every
// leaf the whole reduction extent.
//
// `allow-float-reassociation` is on so that the float case below can be
// checked here; it does not affect the integer cases. The default-off
// behaviour is covered in linalg-to-cnm-invalid.mlir.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// 1024x512 gemv with blocks 256x128: 4 m-tiles and 4 k-tiles fill the 16
// leaves, so each leaf computes a partial sum over a quarter of K.
// CHECK-LABEL: func.func @gemv_split_k
func.func @gemv_split_k(%A: tensor<1024x512xi32>, %x: tensor<512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  // The reduction dimension is split into (4 tiles, 128 each), which turns the
  // tile index into an extra *parallel* dimension of the op.
  // CHECK: %[[EA:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]] output_shape [1024, 4, 128]
  // CHECK: %[[EX:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0, 1]] output_shape [4, 128]

  // Every leaf starts from the combiner's neutral element, not from %y --
  // otherwise the incoming accumulator would be added once per leaf (§G5).
  // CHECK: %[[P:.*]] = tensor.empty() : tensor<1024x4xi32>
  // CHECK: %[[Z:.*]] = arith.constant 0 : i32
  // CHECK: linalg.fill ins(%[[Z]] : i32) outs(%[[P]] : tensor<1024x4xi32>)

  // The leaf buffers carry the split dimension with extent 1: one k-tile each.
  // CHECK: cnm.workgroup
  // CHECK-DAG: cnm.alloc() {{.*}} : !cnm.buffer<256x1x128xi32 on
  // CHECK-DAG: cnm.alloc() {{.*}} : !cnm.buffer<1x128xi32 on
  // CHECK-DAG: cnm.alloc() {{.*}} : !cnm.buffer<256x1xi32 on

  // The launch body is the partial op: the split dimension is parallel, only
  // the 128-wide remainder is still a reduction.
  // CHECK: cnm.launch
  // CHECK: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK: ins(%{{.*}}, %{{.*}} : memref<256x1x128xi32>, memref<1x128xi32>) outs(%{{.*}} : memref<256x1xi32>)

  // The partials come back with the k-tile as a separate dimension, and the
  // merge accumulates them into the *original* %y, folding it in exactly once.
  // CHECK: cnm.gather
  // CHECK: %[[MERGED:.*]] = tensor.reshape %{{.*}} -> tensor<1024x4xi32>
  // CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]} ins(%[[MERGED]] : tensor<1024x4xi32>) outs(%arg2 : tensor<1024xi32>)
  // CHECK: arith.addi
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%y : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// Not contract-specific: the combiner and its neutral element are read off the
// payload region, so a linalg.reduce with a `max` body seeds the leaves with
// the minimum representable value rather than zero.
// CHECK-LABEL: func.func @reduce_max_split_k
func.func @reduce_max_split_k(%A: tensor<1024x512xi32>, %o: tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: %[[NEUTRAL:.*]] = arith.constant -2147483648 : i32
  // CHECK: linalg.fill ins(%[[NEUTRAL]] : i32)
  // CHECK: cnm.launch
  // CHECK: arith.maxsi
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.reduce ins(%A : tensor<1024x512xi32>) outs(%o : tensor<1024xi32>)
      dimensions = [1]
      {cnm.tile_sizes = array<i64: 256, 128>}
      (%in: i32, %acc: i32) {
        %s = arith.maxsi %in, %acc : i32
        linalg.yield %s : i32
      }
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// With the opt-in, a float reduction splits too. The result differs from the
// unsplit one by reassociation, which is the whole reason the flag exists.
// CHECK-LABEL: func.func @gemv_split_k_f32
func.func @gemv_split_k_f32(%A: tensor<1024x512xf32>, %x: tensor<512xf32>, %y: tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: linalg.fill ins(%{{.*}} : f32)
  // CHECK: cnm.alloc() {{.*}} : !cnm.buffer<256x1x128xf32 on
  // CHECK: cnm.launch
  %r = cinm.compute on accelerator #acc -> tensor<1024xf32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>}
      ins(%A, %x : tensor<1024x512xf32>, tensor<512xf32>)
      outs(%y : tensor<1024xf32>) -> tensor<1024xf32>
    cinm.yield %g : tensor<1024xf32>
  }
  func.return %r : tensor<1024xf32>
}

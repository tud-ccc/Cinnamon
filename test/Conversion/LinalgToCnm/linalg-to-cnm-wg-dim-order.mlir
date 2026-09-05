// The default rule of design §G3, and the two ways of overriding it.
// RUN: cinm-opt %s --convert-linalg-to-cnm | FileCheck %s --check-prefixes=CHECK,RULE
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order-index=0 | FileCheck %s --check-prefixes=CHECK,RULE
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=0,1,2 | FileCheck %s --check-prefixes=CHECK,RULE
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order-index=1 | FileCheck %s --check-prefixes=CHECK,FLIP
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=1,0,2 | FileCheck %s --check-prefixes=CHECK,FLIP
// A dimension tiled once takes no workgroup axis, so moving the (unsplit)
// reduction remainder from last to first cannot change anything. This is why
// the index only ranks the distributed dimensions: the other orders of the
// same op are duplicates of these two.
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=2,1,0 | FileCheck %s --check-prefixes=CHECK,FLIP
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=1,2,0 | FileCheck %s --check-prefixes=CHECK,FLIP

// And what is rejected.
// RUN: not cinm-opt %s --convert-linalg-to-cnm="workgroup-dim-order=0,1,2 workgroup-dim-order-index=1" 2>&1 | FileCheck %s --check-prefix=BOTH
// RUN: not cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=0,1 2>&1 | FileCheck %s --check-prefix=SHORT
// RUN: not cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=0,1,1 2>&1 | FileCheck %s --check-prefix=REPEAT
// RUN: not cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=0,1,5 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order-index=2 2>&1 | FileCheck %s --check-prefix=INDEX

// Which tile dimension varies fastest across the leaves is a choice, not a
// consequence: it decides which operands the leaves sharing a hardware node
// share rather than replicate (design §G3). The default rule puts the
// reduction-derived dimension outermost; `workgroup-dim-order` states an order
// outright, and `workgroup-dim-order-index` names one by its rank in
// lexicographic order, which is the form a search can enumerate.
//
// The op below is the split gemv of linalg-to-cnm-split-reduction.mlir: 4
// m-tiles and 4 k-tiles over 16 leaves. After the split its iteration
// dimensions are 0 = the k-tile index (parallel, 4 tiles), 1 = m (parallel, 4
// tiles), 2 = the k remainder (reduction, 1 tile), so there are exactly two
// distinct orders.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dpus = 64>
#acc = #upmem.array<16x1, #pf>

// The vector is indexed by the reduction dimension alone, so its scatter map
// is where the two orders differ visibly.
//
// The maps are pointwise, so their domain is the two workgroup dimensions
// followed by every dimension of the buffer they scatter into.
//
// k-tile outermost: `leaf floordiv 4`, constant over each run of four
// consecutive leaves, which share their k-tile and differ in m.
// RULE-DAG: #[[VEC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 4, d3)>
// RULE-DAG: #[[MAT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ((d0 mod 4) * 256 + d2, d0 floordiv 4, d4)>
// RULE-DAG: #[[OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 4, (d0 mod 4) * 256 + d3)>
//
// k-tile innermost: `leaf mod 4`, so adjacent leaves differ in their k-tile and
// the vector is replicated across each run of four instead.
// FLIP-DAG: #[[VEC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 mod 4, d3)>
// FLIP-DAG: #[[MAT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2 + (d0 floordiv 4) * 256, d0 mod 4, d4)>
// FLIP-DAG: #[[OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 mod 4, d3 + (d0 floordiv 4) * 256)>

// CHECK-LABEL: func.func @gemv_split_k
func.func @gemv_split_k(%A: tensor<1024x512xi32>, %x: tensor<512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  // Only the maps move. The buffer shapes are a property of the block sizes,
  // and the launch body of the tile counts, so the order leaves both alone.
  // CHECK: cnm.scatter %{{.*}}[#[[MAT]]] {{.*}} : tensor<1024x4x128xi32> into !cnm.buffer<256x1x128xi32 on
  // CHECK: cnm.scatter %{{.*}}[#[[VEC]]] {{.*}} : tensor<4x128xi32> into !cnm.buffer<1x128xi32 on
  // CHECK: cnm.scatter %{{.*}}[#[[OUT]]] {{.*}} : tensor<4x1024xi32> into !cnm.buffer<1x256xi32 on
  // CHECK: cnm.launch
  // CHECK: cnm.gather %{{.*}}[#[[OUT]]] {{.*}} into tensor<4x1024xi32>

  // BOTH: 'workgroup-dim-order' and 'workgroup-dim-order-index' are two ways of stating the same thing; pass at most one
  // SHORT: 'workgroup-dim-order' has 2 entries but this op has 3 iteration dimension(s)
  // REPEAT: 'workgroup-dim-order' must be a permutation of [0, 3), but 1 is out of range or repeated
  // RANGE: 'workgroup-dim-order' must be a permutation of [0, 3), but 5 is out of range or repeated
  // INDEX: 'workgroup-dim-order-index' is 2, but this op spreads 2 iteration dimension(s) over the workgroup, so it has 2 distinct order(s)
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%y : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

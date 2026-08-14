// The order as a per-op attribute. Every RUN below must produce the same
// output: the attribute is what the op says about itself, so it wins over
// whatever the pass options say -- including options asking for the opposite
// order, and including one that would be an error if anything read it.
// RUN: cinm-opt %s --convert-linalg-to-cnm --mlir-print-local-scope | FileCheck %s
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order-index=0 --mlir-print-local-scope | FileCheck %s
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order=0,1,2 --mlir-print-local-scope | FileCheck %s
// RUN: cinm-opt %s --convert-linalg-to-cnm=workgroup-dim-order-index=99 --mlir-print-local-scope | FileCheck %s

// The block sizes reach this pass per op, as `cnm.tile_sizes` stamped by the
// search. The order is stamped the same way -- as `cnm.workgroup_dim_order` or
// `cnm.workgroup_dim_order_index` -- so a compute block holding two ops can
// give them different orders, which a pass option cannot do. See design §G3.
// The invalid combinations are in linalg-to-cnm-invalid.mlir.
//
// Both ops below are the split gemv of linalg-to-cnm-split-reduction.mlir, and
// both ask for the order the default rule does *not* give: the k-tile index
// innermost, so adjacent leaves differ in their k-tile and the vector operand
// is replicated across them rather than shared. Its scatter map is where that
// shows: `leaf mod 4` here, `leaf floordiv 4` under the rule.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dpus = 64>
#acc = #upmem.array<16x1, #pf>

// CHECK-LABEL: func.func @by_index
func.func @by_index(%A: tensor<1024x512xi32>, %x: tensor<512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: cnm.scatter %{{.*}}[affine_map<(d0, d1, d2, d3) -> (d0 mod 4, d3)>] {{.*}} : tensor<4x128xi32> into !cnm.buffer<1x128xi32 on
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>,
       cnm.workgroup_dim_order_index = 1 : i64}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%y : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// The permutation form of the same choice. It is stated over the dimensions
// the op has *after* the reduction split, which is why an op written with two
// iteration dimensions takes three entries.
// CHECK-LABEL: func.func @by_permutation
func.func @by_permutation(%A: tensor<1024x512xi32>, %x: tensor<512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: cnm.scatter %{{.*}}[affine_map<(d0, d1, d2, d3) -> (d0 mod 4, d3)>] {{.*}} : tensor<4x128xi32> into !cnm.buffer<1x128xi32 on
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>,
       cnm.workgroup_dim_order = array<i64: 1, 0, 2>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%y : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// The attribute is consumed here, so it does not travel into the launch body
// with the rest of the discardable attributes.
// CHECK: cnm.launch
// CHECK-NOT: cnm.workgroup_dim_order

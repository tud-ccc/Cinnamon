// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm=cnm-buffer-level=mram --canonicalize %s | FileCheck %s

// With a level selected, the launch body has to stay tilable, because staging
// it down to the leaf level is a later pass's job. So it holds a memref-mode
// cinm op (which implements CinmTilingInterface) rather than linalg.
//
// The shapes here need the MRAM budget: sized against WRAM they would be
// rejected outright, which is why this file has no no-flag RUN line. The
// no-flag behavior is covered by cinm-to-cnm-buffer-level.mlir.

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<1x16x1, #upmem_platform>

// CHECK-LABEL: @gemv
func.func @gemv(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  // 1024 rows over 16 leaves = 64 rows each, so the per-leaf tile is a real
  // 64x512 matrix-vector product. At 128 KiB it does not fit WRAM.
  // CHECK: cnm.launch %{{.*}} ins(%[[A:.*]] = %{{.*}} : <64x512xi32, #upmem.mram>, %[[X:.*]] = %{{.*}} : <512xi32, #upmem.mram>) outs(%[[Y:.*]] = %{{.*}} : <64xi32, #upmem.mram>)
  // CHECK: cinm.op.gemv %[[A]], %[[X]] into %[[Y]] : memref<64x512xi32, #upmem.mram>, memref<512xi32, #upmem.mram> into memref<64xi32, #upmem.mram>
  // CHECK-NOT: linalg.contract
  %r0 = cinm.compute on accelerator #upmem -> tensor<1024xi32> {
    %r = cinm.op.gemv %A, %x : tensor<1024x512xi32>, tensor<512xi32> -> tensor<1024xi32>
    cinm.yield %r : tensor<1024xi32>
  }
  func.return %r0 : tensor<1024xi32>
}

// -----

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<1x16x1, #upmem_platform>

// CHECK-LABEL: @reduce
func.func @reduce(%a: tensor<1024x512xi32>) -> tensor<1024xi32> {
  // CHECK: cnm.launch %{{.*}} ins(%[[IN:.*]] = %{{.*}} : <64x512xi32, #upmem.mram>) outs(%[[OUT:.*]] = %{{.*}} : <64xi32, #upmem.mram>)
  // CHECK: cinm.op.reduce add(%[[IN]]) into %[[OUT]] : memref<64x512xi32, #upmem.mram> into memref<64xi32, #upmem.mram>
  // CHECK-NOT: linalg.reduce
  %r0 = cinm.compute on accelerator #upmem -> tensor<1024xi32> {
    %r = cinm.op.reduce add (%a) : tensor<1024x512xi32> -> tensor<1024xi32>
    cinm.yield %r : tensor<1024xi32>
  }
  func.return %r0 : tensor<1024xi32>
}

// -----

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<1x16x1, #upmem_platform>

// One output element per leaf: the tile is a dot product, which cinm.op.gemv
// does not model, so the body stays linalg even with a level selected.
// CHECK-LABEL: @gemv_one_row_per_leaf
func.func @gemv_one_row_per_leaf(%A: tensor<16x1024xi32>, %x: tensor<1024xi32>) -> tensor<16xi32> {
  // CHECK: cnm.launch %{{.*}} ins(%{{.*}} : <1024xi32, #upmem.mram>, %{{.*}} : <1024xi32, #upmem.mram>) outs(%{{.*}} : <i32, #upmem.mram>)
  // CHECK: linalg.contract
  // CHECK-NOT: cinm.op.gemv
  %r0 = cinm.compute on accelerator #upmem -> tensor<16xi32> {
    %r = cinm.op.gemv %A, %x : tensor<16x1024xi32>, tensor<1024xi32> -> tensor<16xi32>
    cinm.yield %r : tensor<16xi32>
  }
  func.return %r0 : tensor<16xi32>
}

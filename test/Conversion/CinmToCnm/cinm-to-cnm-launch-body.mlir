// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm=cnm-buffer-level=mram --canonicalize %s | FileCheck %s

// Selecting a level changes where buffers live, not what the launch body is:
// the body stays linalg, over memrefs tagged with the level. That tag is the
// only thing --upmem-tile-mram-buffers needs to recognize an op it should
// stage down to the leaf level, so it works for any linalg op on buffers
// rather than for a fixed set of cinm ops.
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
  // CHECK: linalg.contract {{.*}} ins(%[[A]], %[[X]] : memref<64x512xi32, #upmem.mram>, memref<512xi32, #upmem.mram>) outs(%[[Y]] : memref<64xi32, #upmem.mram>)
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
  // CHECK: linalg.reduce ins(%[[IN]] : memref<64x512xi32, #upmem.mram>) outs(%[[OUT]] : memref<64xi32, #upmem.mram>)
  // CHECK-SAME: dimensions = [1]
  %r0 = cinm.compute on accelerator #upmem -> tensor<1024xi32> {
    %r = cinm.op.reduce add (%a) : tensor<1024x512xi32> -> tensor<1024xi32>
    cinm.yield %r : tensor<1024xi32>
  }
  func.return %r0 : tensor<1024xi32>
}

// -----

#upmem_platform = #upmem.platform<type=v1A, dimensions = 8x128>
#upmem = #upmem.array<8x128x1, #upmem_platform>

// Gemm gives each leaf exactly one output element, so its tile is a dot
// product rather than a matrix multiply. Nothing special is needed for it:
// its body is a linalg op on levelled memrefs like any other.
// CHECK-LABEL: @gemm
func.func @gemm(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {
  // CHECK: linalg.contract {{.*}} ins(%{{.*}}, %{{.*}} : memref<1024xi32, #upmem.mram>, memref<1024xi32, #upmem.mram>) outs(%{{.*}} : memref<i32, #upmem.mram>)
  %r0 = cinm.compute on accelerator #upmem -> tensor<8x128xi32> {
    %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
    cinm.yield %r : tensor<8x128xi32>
  }
  func.return %r0 : tensor<8x128xi32>
}

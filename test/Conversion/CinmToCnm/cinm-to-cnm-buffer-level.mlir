// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm=cnm-buffer-level=mram --canonicalize %s | FileCheck %s --check-prefix=MRAM
// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm=cnm-buffer-level=wram --canonicalize %s | FileCheck %s --check-prefix=WRAM
// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm --canonicalize %s | FileCheck %s --check-prefix=NONE

// cnm-buffer-level names a level of the accelerator's platform. The level goes
// in the cnm.buffer type and, per LaunchOp's contract, becomes the memory
// space of the launch body's memref block arguments.

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<1x16x1, #upmem_platform>

// MRAM-LABEL: @gemv
// WRAM-LABEL: @gemv
// NONE-LABEL: @gemv
func.func @gemv(%A: tensor<16x1024xi32>, %x: tensor<1024xi32>) -> tensor<16xi32> {
  // MRAM: cnm.alloc() for %{{.*}} : !cnm.buffer<1024xi32 on #{{.*}}, #upmem.mram>
  // MRAM: cnm.alloc() for %{{.*}} : !cnm.buffer<i32 on #{{.*}}, #upmem.mram>
  // MRAM: cnm.launch %{{.*}} ins(%{{.*}} : <1024xi32, #upmem.mram>, %{{.*}} : <1024xi32, #upmem.mram>) outs(%{{.*}} : <i32, #upmem.mram>)
  // Each leaf gets a single output element here, so its tile is a dot product,
  // which cinm.op.gemv does not model: the body stays linalg even with a level.
  // MRAM: linalg.contract {{.*}} ins(%{{.*}}, %{{.*}} : memref<1024xi32, #upmem.mram>, memref<1024xi32, #upmem.mram>) outs(%{{.*}} : memref<i32, #upmem.mram>)
  // MRAM-NOT: cinm.op.gemv

  // Same shapes, different level: the option is really consulted rather than
  // the level being hardcoded.
  // WRAM: cnm.alloc() for %{{.*}} : !cnm.buffer<1024xi32 on #{{.*}}, #upmem.wram>
  // WRAM: cnm.launch %{{.*}} ins(%{{.*}} : <1024xi32, #upmem.wram>, %{{.*}} : <1024xi32, #upmem.wram>) outs(%{{.*}} : <i32, #upmem.wram>)
  // WRAM: memref<1024xi32, #upmem.wram>

  // No flag: unlevelled buffers and plain memrefs, exactly as before.
  // NONE: cnm.alloc() for %{{.*}} : !cnm.buffer<1024xi32 on #{{.*}}>
  // NONE: cnm.launch %{{.*}} ins(%{{.*}} : <1024xi32>, %{{.*}} : <1024xi32>) outs(%{{.*}} : <i32>)
  // NONE: memref<1024xi32>
  // NONE-NOT: #upmem.mram
  // NONE-NOT: #upmem.wram
  %r0 = cinm.compute on accelerator #upmem -> tensor<16xi32> {
    %r = cinm.op.gemv %A, %x : tensor<16x1024xi32>, tensor<1024xi32> -> tensor<16xi32>
    cinm.yield %r : tensor<16xi32>
  }
  func.return %r0 : tensor<16xi32>
}

// -----

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<8x128x1, #upmem_platform>

// The gemm pattern builds its buffer types directly rather than going through
// convertCinmToCnm, so it needs its own coverage.

// MRAM-LABEL: @gemm
// WRAM-LABEL: @gemm
// NONE-LABEL: @gemm
func.func @gemm(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {
  // MRAM: cnm.alloc() for %{{.*}} : !cnm.buffer<1024xi32 on #{{.*}}, #upmem.mram>
  // MRAM: cnm.alloc() for %{{.*}} : !cnm.buffer<i32 on #{{.*}}, #upmem.mram>
  // The gemm pattern gives each leaf exactly one output element, so its tile is
  // always a dot product: unlike gemv, it has no shape in which a cinm op could
  // stand in the launch body.
  // MRAM: linalg.contract {{.*}} : memref<1024xi32, #upmem.mram>, memref<1024xi32, #upmem.mram>) outs(%{{.*}} : memref<i32, #upmem.mram>)
  // MRAM-NOT: cinm.op.gemm
  // WRAM: cnm.alloc() for %{{.*}} : !cnm.buffer<1024xi32 on #{{.*}}, #upmem.wram>
  // NONE: cnm.alloc() for %{{.*}} : !cnm.buffer<1024xi32 on #{{.*}}>
  %r0 = cinm.compute on accelerator #upmem -> tensor<8x128xi32> {
    %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
    cinm.yield %r : tensor<8x128xi32>
  }
  func.return %r0 : tensor<8x128xi32>
}

// -----

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<1x16x1, #upmem_platform>

// A reduce whose per-leaf tile keeps a parallel dimension: 1024 rows over 16
// leaves leaves 64 rows each, so the buffer is 64x8 and the reduction is over
// its *last* dimension, not dimension 0.
// MRAM-LABEL: @reduce_multirow
// WRAM-LABEL: @reduce_multirow
// NONE-LABEL: @reduce_multirow
func.func @reduce_multirow(%a: tensor<1024x8xi32>) -> tensor<1024xi32> {
  // MRAM: cinm.op.reduce add(%{{.*}}) into %{{.*}} : memref<64x8xi32, #upmem.mram> into memref<64xi32, #upmem.mram>
  // WRAM: cinm.op.reduce add(%{{.*}}) into %{{.*}} : memref<64x8xi32, #upmem.wram> into memref<64xi32, #upmem.wram>
  // NONE: linalg.reduce
  // NONE-SAME: dimensions = [1]
  %r0 = cinm.compute on accelerator #upmem -> tensor<1024xi32> {
    %r = cinm.op.reduce add (%a) : tensor<1024x8xi32> -> tensor<1024xi32>
    cinm.yield %r : tensor<1024xi32>
  }
  func.return %r0 : tensor<1024xi32>
}

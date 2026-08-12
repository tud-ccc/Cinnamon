// RUN: cinm-opt %s --split-input-file --verify-diagnostics

// -----

// load_program references a symbol that does not exist in the module.
module {
  func.func @test() {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{requires @dpu_kernels::@nonexistent to refer to an upmem.dpu_program op}}
    upmem.load_program @dpu_kernels::@nonexistent on %1 : !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      upmem.return
    }
  }
}

// -----

// load_program references a symbol that exists but is not a upmem.dpu_program.
module {
  func.func @test() {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{requires @dpu_kernels::@not_a_program to refer to an upmem.dpu_program op}}
    upmem.load_program @dpu_kernels::@not_a_program on %1 : !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    func.func @not_a_program() {
      return
    }
  }
}

// -----

// scatter references a buffer name that does not exist in the dpu_program.
module {
  func.func @test(%arg0: memref<8x128xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{buffer reference @nonexistent does not refer to any symbol in @program}}
    upmem.scatter_on_array %arg0[128 elts, affine_map<(d0) -> (d0 floordiv 128, 0)>] onto @nonexistent of %1
        : memref<8x128xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<8x128xi32, "mram">
      upmem.return
    }
  }
}

// -----

// gather references a buffer name that does not exist in the dpu_program.
module {
  func.func @test(%arg0: memref<8x128xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{buffer reference @nonexistent does not refer to any symbol in @program}}
    upmem.gather_from_array %arg0[128 elts, affine_map<(d0) -> (d0 floordiv 128, 0)>] from @nonexistent of %1
        : memref<8x128xi32> from !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<8x128xi32, "mram">
      upmem.return
    }
  }
}

// -----

// scatter references a symbol in the dpu_program that is not a upmem.static_alloc.
module {
  func.func @test(%arg0: memref<8x128xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{buffer reference @not_a_buf must refer to a named upmem.static_alloc op}}
    upmem.scatter_on_array %arg0[128 elts, affine_map<(d0) -> (d0 floordiv 128, 0)>] onto @not_a_buf of %1
        : memref<8x128xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      func.func @not_a_buf() {
        return
      }
      upmem.return
    }
  }
}

// -----

// scatter map has fewer results than the host buffer rank.
module {
  func.func @test(%arg0: memref<8x128xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{Scatter map should map (dpu) to a start index in the host buffer}}
    upmem.scatter_on_array %arg0[128 elts, affine_map<(d0) -> (d0)>] onto @buf of %1
        : memref<8x128xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<8x128xi32, "mram">
      upmem.return
    }
  }
}

// -----

// scatter map has more dimensions than the (dpu) form allows.
module {
  func.func @test(%arg0: memref<128xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{Scatter map should map (dpu) to a start index in the host buffer}}
    upmem.scatter_on_array %arg0[128 elts, affine_map<(d0, d1) -> (d0)>] onto @buf of %1
        : memref<128xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<128xi32, "mram">
      upmem.return
    }
  }
}

// -----

// upmem.scatter_on_array (unlike upmem.scatter_blocks) may not use the
// (dpu, block) form.
module {
  func.func @test(%arg0: memref<8x128x4xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x4>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x4>
    // expected-error @+1 {{Scatter map should map (dpu) to a start index in the host buffer}}
    upmem.scatter_on_array %arg0[32 elts, affine_map<(d0, d1) -> (d0, d1, 0)>] onto @buf of %1
        : memref<8x128x4xi32> onto !upmem.hierarchy<1024x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// gather_from_array's map may not use the two-dimensional form: that is what
// upmem.gather_blocks is for.
module {
  func.func @test(%arg0: memref<8x128x4xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x4>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x4>
    // expected-error @+1 {{Scatter map should map (dpu) to a start index in the host buffer}}
    upmem.gather_from_array %arg0[32 elts, affine_map<(d0, d1) -> (d0, d1, 0)>] from @buf of %1
        : memref<8x128x4xi32> from !upmem.hierarchy<1024x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// A valid upmem.scatter_blocks: the scatter map computes, for each block, its
// own start index in the host buffer. Blocks need not be contiguous with each
// other (here they come from non-adjacent rows of the host buffer), only each
// individual block (32 contiguous elements) must be. 128 DPUs x 4 blocks need
// 512 rows to draw from.
module {
  func.func @test(%arg0: memref<512x1024xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<128x4>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<128x4>
    upmem.scatter_blocks %arg0[32 elts, affine_map<(d0, d1) -> (d0 * 4 + d1, 0)>, 4 blocks] onto @buf of %1
        : memref<512x1024xi32> onto !upmem.hierarchy<128x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// A valid upmem.gather_blocks, the mirror image of the scatter above: each
// DPU's four blocks are written back to four non-adjacent host rows.
module {
  func.func @test(%arg0: memref<512x1024xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<128x4>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<128x4>
    upmem.gather_blocks %arg0[32 elts, affine_map<(d0, d1) -> (d0 * 4 + d1, 0)>, 4 blocks] from @buf of %1
        : memref<512x1024xi32> from !upmem.hierarchy<128x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// A block must fit inside one contiguous run, and where it starts is part of
// that question: this host memref stores rows of 128 elements 1024 apart, so
// only a block starting at column 0 has 128 elements behind it.
module {
  func.func @test(%arg0: memref<8x128xi32, strided<[1024, 1], offset: ?>>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<8x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<8x1>
    // expected-error @+1 {{a transferred block starts at offset 64 of a contiguous run of 128 elements}}
    upmem.scatter_on_array %arg0[128 elts, affine_map<(d0) -> (d0, 64)>] onto @buf of %1
        : memref<8x128xi32, strided<[1024, 1], offset: ?>> onto !upmem.hierarchy<8x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<128xi32, "mram">
      upmem.return
    }
  }
}

// -----

// The same map is fine at column 0, where the whole row is behind it.
module {
  func.func @test(%arg0: memref<8x128xi32, strided<[1024, 1], offset: ?>>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<8x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<8x1>
    upmem.scatter_on_array %arg0[128 elts, affine_map<(d0) -> (d0, 0)>] onto @buf of %1
        : memref<8x128xi32, strided<[1024, 1], offset: ?>> onto !upmem.hierarchy<8x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<128xi32, "mram">
      upmem.return
    }
  }
}

// -----

// upmem.scatter_blocks requires the (dpu, block) scatter map form.
module {
  func.func @test(%arg0: memref<128x1024xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<128x4>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<128x4>
    // expected-error @+1 {{Scatter map should map (dpu, block) to a start index in the host buffer}}
    upmem.scatter_blocks %arg0[32 elts, affine_map<(d0) -> (d0, 0)>, 4 blocks] onto @buf of %1
        : memref<128x1024xi32> onto !upmem.hierarchy<128x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// scatter transferCount spans several rows of a strided (tiled) host buffer,
// so each DPU's elements would not actually be contiguous in memory: a
// 4-row x 1024-col tile of a 4096-wide matrix is not a contiguous run of
// 4096 elements.
module {
  func.func @test(%arg0: memref<1024x1024xi32, strided<[4096, 1], offset: ?>>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<256x4>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<256x4>
    // expected-error @+1 {{the number of transferred elements (4096) exceeds the largest contiguous run of elements (1024) in host buffer}}
    upmem.scatter_on_array %arg0[4096 elts, affine_map<(d0) -> (d0 * 4, 0)>] onto @buf of %1
        : memref<1024x1024xi32, strided<[4096, 1], offset: ?>> onto !upmem.hierarchy<256x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x1024xi32, "mram">
      upmem.return
    }
  }
}

// -----

// Same non-contiguity issue, but for gather.
module {
  func.func @test(%arg0: memref<1024x1024xi32, strided<[4096, 1], offset: ?>>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<256x4>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<256x4>
    // expected-error @+1 {{the number of transferred elements (4096) exceeds the largest contiguous run of elements (1024) in host buffer}}
    upmem.gather_from_array %arg0[4096 elts, affine_map<(d0) -> (d0 * 4, 0)>] from @buf of %1
        : memref<1024x1024xi32, strided<[4096, 1], offset: ?>> from !upmem.hierarchy<256x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x1024xi32, "mram">
      upmem.return
    }
  }
}

// -----

// A strided host buffer whose transferred elements ARE contiguous (a single
// full row of the tile) must still verify successfully.
module {
  func.func @test(%arg0: memref<1024x1024xi32, strided<[4096, 1], offset: ?>>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    upmem.scatter_on_array %arg0[1024 elts, affine_map<(d0) -> (d0, 0)>] onto @buf of %1
        : memref<1024x1024xi32, strided<[4096, 1], offset: ?>> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<1024xi32, "mram">
      upmem.return
    }
  }
}

// -----

// A valid broadcast: the host buffer's shape matches the target buffer's
// shape once the target's leading extent-1 dim is dropped.
module {
  func.func @test(%arg0: memref<32xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    upmem.broadcast %arg0 onto @buf of %1 : memref<32xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<1x32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// broadcast references a buffer name that does not exist in the dpu_program.
module {
  func.func @test(%arg0: memref<32xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{buffer reference @nonexistent does not refer to any symbol in @program}}
    upmem.broadcast %arg0 onto @nonexistent of %1 : memref<32xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// broadcast references a symbol in the dpu_program that is not a
// upmem.static_alloc.
module {
  func.func @test(%arg0: memref<32xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{buffer reference @not_a_buf must refer to a named upmem.static_alloc op}}
    upmem.broadcast %arg0 onto @not_a_buf of %1 : memref<32xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      func.func @not_a_buf() {
        return
      }
      upmem.return
    }
  }
}

// -----

// broadcast host buffer shape is not compatible with the target buffer's
// shape, even up to extent-1 dimensions.
module {
  func.func @test(%arg0: memref<32xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{host buffer shape 'memref<32xi32>' is not compatible with target buffer 'memref<2x16xi32, "mram">' (shapes must be equal up to extent-1 dimensions)}}
    upmem.broadcast %arg0 onto @buf of %1 : memref<32xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<2x16xi32, "mram">
      upmem.return
    }
  }
}

// -----

// broadcast host buffer must have a static shape.
module {
  func.func @test(%arg0: memref<?xi32>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{host buffer must have a static shape}}
    upmem.broadcast %arg0 onto @buf of %1 : memref<?xi32> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<32xi32, "mram">
      upmem.return
    }
  }
}

// -----

// broadcast host buffer must be entirely contiguous: a 4-row x 8-col tile of
// a 16-wide matrix is not one contiguous run of 32 elements.
module {
  func.func @test(%arg0: memref<4x8xi32, strided<[16, 1], offset: ?>>) {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x1>
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x1>
    // expected-error @+1 {{the number of transferred elements (32) exceeds the largest contiguous run of elements (8) in host buffer}}
    upmem.broadcast %arg0 onto @buf of %1
        : memref<4x8xi32, strided<[16, 1], offset: ?>> onto !upmem.hierarchy<1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x8xi32, "mram">
      upmem.return
    }
  }
}

// -----

// load_program checks that the program's tasklet count matches the hierarchy.
module {
  func.func @test() {
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1024x16>
    // expected-error @+1 {{loads a program compiled for 1 tasklet(s) onto a hierarchy of 16 tasklet(s) per DPU}}
    upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<1024x16>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      upmem.return
    }
  }
}

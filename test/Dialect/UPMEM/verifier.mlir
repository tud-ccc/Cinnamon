// RUN: cinm-opt %s --split-input-file --verify-diagnostics

// -----

// alloc_dpus references a symbol that does not exist in the module.
module {
  func.func @test() {
    // expected-error @+1 {{requires @dpu_kernels::@nonexistent to refer to an upmem.dpu_program op}}
    %1 = upmem.alloc_dpus with program @dpu_kernels::@nonexistent : !upmem.hierarchy<8x128x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      upmem.return
    }
  }
}

// -----

// alloc_dpus references a symbol that exists but is not a upmem.dpu_program.
module {
  func.func @test() {
    // expected-error @+1 {{requires @dpu_kernels::@not_a_program to refer to an upmem.dpu_program op}}
    %1 = upmem.alloc_dpus with program @dpu_kernels::@not_a_program : !upmem.hierarchy<8x128x1>
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
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<8x128x1>
    // expected-error @+1 {{buffer reference @nonexistent does not refer to any symbol in @dpu_kernels::@program}}
    upmem.scatter %arg0[128, affine_map<(d0, d1) -> (d0, d1)>] onto @nonexistent of %1
        : memref<8x128xi32> onto !upmem.hierarchy<8x128x1>
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
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<8x128x1>
    // expected-error @+1 {{buffer reference @nonexistent does not refer to any symbol in @dpu_kernels::@program}}
    upmem.gather %arg0[128, affine_map<(d0, d1) -> (d0, d1)>] from @nonexistent of %1
        : memref<8x128xi32> from !upmem.hierarchy<8x128x1>
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
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<8x128x1>
    // expected-error @+1 {{buffer reference @not_a_buf must refer to a named upmem.static_alloc op}}
    upmem.scatter %arg0[128, affine_map<(d0, d1) -> (d0, d1)>] onto @not_a_buf of %1
        : memref<8x128xi32> onto !upmem.hierarchy<8x128x1>
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
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<8x128x1>
    // expected-error @+1 {{Scatter map should map (rank, dpu) to a start index in the host buffer}}
    upmem.scatter %arg0[128, affine_map<(d0, d1) -> (d0)>] onto @buf of %1
        : memref<8x128xi32> onto !upmem.hierarchy<8x128x1>
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

// scatter map has fewer than 2 dimensions (must be (rank, dpu) -> ...).
module {
  func.func @test(%arg0: memref<128xi32>) {
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<8x128x1>
    // expected-error @+1 {{Scatter map should map (rank, dpu) to a start index in the host buffer}}
    upmem.scatter %arg0[128, affine_map<(d0) -> (d0)>] onto @buf of %1
        : memref<128xi32> onto !upmem.hierarchy<8x128x1>
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

// scatter transferCount spans several rows of a strided (tiled) host buffer,
// so each DPU's elements would not actually be contiguous in memory: a
// 4-row x 1024-col tile of a 4096-wide matrix is not a contiguous run of
// 4096 elements.
module {
  func.func @test(%arg0: memref<1024x1024xi32, strided<[4096, 1], offset: ?>>) {
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<1x256x4>
    // expected-error @+1 {{transferCount (4096) exceeds the largest contiguous run of elements (1024) in host buffer}}
    upmem.scatter %arg0[4096, affine_map<(d0, d1) -> (d0 * 1024 + d1 * 4, 0)>] onto @buf of %1
        : memref<1024x1024xi32, strided<[4096, 1], offset: ?>> onto !upmem.hierarchy<1x256x4>
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
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<1x256x4>
    // expected-error @+1 {{transferCount (4096) exceeds the largest contiguous run of elements (1024) in host buffer}}
    upmem.gather %arg0[4096, affine_map<(d0, d1) -> (d0 * 1024 + d1 * 4, 0)>] from @buf of %1
        : memref<1024x1024xi32, strided<[4096, 1], offset: ?>> from !upmem.hierarchy<1x256x4>
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
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<1x1024x1>
    upmem.scatter %arg0[1024, affine_map<(d0, d1) -> (d0 * 1024 + d1, 0)>] onto @buf of %1
        : memref<1024x1024xi32, strided<[4096, 1], offset: ?>> onto !upmem.hierarchy<1x1024x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<1024xi32, "mram">
      upmem.return
    }
  }
}

// RUN: cinm-opt %s --canonicalize --cse --split-input-file | FileCheck %s

// A named static_alloc is what the host scatters into and gathers from, by
// symbol rather than by SSA use. DCE cannot see that reference, so without a
// side effect saying so it would drop the buffer as an allocation nobody reads
// -- leaving the host addressing a symbol the device program no longer defines.

// CHECK-LABEL: upmem.dpu_program @named_survives
module @dpu_kernels {
  upmem.dpu_program @named_survives() tasklets(1) {
    %wram = memref.alloca() : memref<64xi32, "wram">
    // CHECK: upmem.static_alloc @in(mram)
    %in = upmem.static_alloc @in(mram) : memref<64xi32, "mram">
    // The output buffer is only ever gathered from, by the host.
    // CHECK: upmem.static_alloc @out(mram)
    %out = upmem.static_alloc @out(mram) : memref<64xi32, "mram">
    upmem.local_transfer %in into %wram : memref<64xi32, "mram"> to memref<64xi32, "wram">
    upmem.return
  }
}

// -----

// An anonymous allocation has no such reader: nothing outside the program can
// name it, so an unused one is dead and should go.

// CHECK-LABEL: upmem.dpu_program @anonymous_is_dead
module @dpu_kernels {
  upmem.dpu_program @anonymous_is_dead() tasklets(1) {
    // CHECK-NOT: upmem.static_alloc
    %dead = upmem.static_alloc(mram) : memref<64xi32, "mram">
    upmem.return
  }
}

// -----

// A transfer map is simplified under the extents of its own domain. For the
// block forms the second dimension is the block index, bounded by
// numBlocksPerDpu (8 here) and not by the hierarchy's tasklet count (2): one
// tasklet's data may arrive as several blocks. Assuming the smaller bound
// would let `d1 floordiv 4` fold to 0 and `d1 mod 4` to `d1`, so every DPU
// would read its first four blocks four times over and never see the rest.

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 4, d1 mod 4, 0)>

// CHECK-LABEL: func.func @block_dim_is_bounded_by_num_blocks
func.func @block_dim_is_bounded_by_num_blocks(%host: memref<4x2x4x8xi32>) {
  %dpus = upmem.alloc_dpus : !upmem.hierarchy<4x2>
  upmem.load_program @dpu_kernels::@program on %dpus : !upmem.hierarchy<4x2>
  // CHECK: upmem.scatter_blocks %{{.*}}[8 elts, #[[MAP]], 8 blocks]
  upmem.scatter_blocks %host[8 elts, affine_map<(d0, d1) -> (d0, d1 floordiv 4, d1 mod 4, 0)>, 8 blocks] onto @buf of %dpus : memref<4x2x4x8xi32> onto !upmem.hierarchy<4x2>
  upmem.free_dpus %dpus : !upmem.hierarchy<4x2>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(2) {
    %buf = upmem.static_alloc @buf(mram) : memref<64xi32, "mram">
    upmem.return
  }
}

// -----

// Relinearizing an index from one tiling into another leaves telescoping
// sums like `d1 + 16*(d1 floordiv 128) - 16*(d1 floordiv 16) - ...`, whose
// pairs are the definition of a mod:  c*x - c*k*(x floordiv k) == c*(x mod k).
// Simplification recognizes them, and merges nested divisions
// ((x floordiv a) floordiv b == x floordiv (a*b)), so every result comes out
// as a plain extract of a bit field of the block index.

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1) -> ((d1 floordiv 256) mod 4, d1 mod 16 + ((d1 floordiv 128) mod 2) * 16, d0 * 16 + (d1 mod 128) floordiv 16 + (d1 floordiv 1024) * 8)>

// CHECK-LABEL: func.func @telescoping_sums_become_mods
func.func @telescoping_sums_become_mods(%host: memref<4x32x64xi32>) {
  %dpus = upmem.alloc_dpus : !upmem.hierarchy<4x16>
  upmem.load_program @dpu_kernels::@program on %dpus : !upmem.hierarchy<4x16>
  // CHECK: upmem.gather_blocks %{{.*}}[1 elts, #[[MAP]], 2048 blocks]
  upmem.gather_blocks %host[1 elts, affine_map<(d0, d1) -> (((d1 floordiv 128) floordiv 2) mod 4, d1 + (d1 floordiv 128) * 16 - (d1 floordiv 16) * 16 - ((d1 floordiv 128) floordiv 2) * 32, d0 * 16 + (d1 mod 128) floordiv 16 + ((d1 floordiv 128) floordiv 8) * 8)>, 2048 blocks] from @buf of %dpus : memref<4x32x64xi32> from !upmem.hierarchy<4x16>
  upmem.free_dpus %dpus : !upmem.hierarchy<4x16>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(16) {
    %buf = upmem.static_alloc @buf(mram) : memref<128xi32, "mram">
    upmem.return
  }
}

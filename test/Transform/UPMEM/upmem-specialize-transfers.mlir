// RUN: cinm-opt %s --upmem-specialize-transfers --split-input-file | FileCheck %s
// RUN: cinm-opt %s --upmem-specialize-transfers=use-bc-xfer-codegen=false --split-input-file | FileCheck %s --check-prefix=NOBC

// A DPU's eight blocks all read the same eight-element constant, and land
// back to back in a 8x1x8 MRAM buffer -- so one broadcast of a 8x1x8 constant
// puts the same bytes there. This is the zero seed of a split reduction.

// CHECK-DAG: memref.global "private" constant @[[TILE:[^ ]*]] : memref<8x1x8xi32> = dense<0>
// CHECK-LABEL: func.func @uniform_blocks
// CHECK: %[[T:.*]] = memref.get_global @[[TILE]]
// CHECK: upmem.broadcast %[[T]] onto @buf
// CHECK-NOT: upmem.scatter_blocks

// NOBC-LABEL: func.func @uniform_blocks
// NOBC: upmem.scatter_blocks
module {
  memref.global "private" constant @seed : memref<1x8xi32> = dense<0>
  func.func @uniform_blocks() {
    %0 = memref.get_global @seed : memref<1x8xi32>
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<512x8>
    upmem.scatter_blocks %0[8 elts, affine_map<(d0, d1) -> (0, 0)>, 8 blocks] onto @buf of %1
        : memref<1x8xi32> onto !upmem.hierarchy<512x8>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(8) {
      %buf = upmem.static_alloc @buf(mram) : memref<8x1x8xi32, "mram">
      %wram = upmem.static_alloc(wram) : memref<8x1x8xi32, "wram">
      upmem.local_transfer %buf into %wram : memref<8x1x8xi32, "mram"> to memref<8x1x8xi32, "wram">
      upmem.return
    }
  }
}

// -----

// The four blocks of a DPU are adjacent and in order (block b starts at
// column 8*b of the DPU's row), so they are one 32-element run and a flat
// per-DPU transfer moves them.

// CHECK-DAG: #[[ROW:[^ ]*]] = affine_map<(d0) -> (d0, 0)>
// CHECK-LABEL: func.func @adjacent_blocks
// CHECK: upmem.scatter_on_array %{{.*}}[32 elts, #[[ROW]]] onto @buf
// CHECK-NOT: upmem.scatter_blocks
module {
  func.func @adjacent_blocks(%arg0: memref<128x32xi32>) {
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<128x4>
    upmem.scatter_blocks %arg0[8 elts, affine_map<(d0, d1) -> (d0, d1 * 8)>, 4 blocks] onto @buf of %1
        : memref<128x32xi32> onto !upmem.hierarchy<128x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x8xi32, "mram">
      %wram = upmem.static_alloc(wram) : memref<4x8xi32, "wram">
      upmem.local_transfer %buf into %wram : memref<4x8xi32, "mram"> to memref<4x8xi32, "wram">
      upmem.return
    }
  }
}

// -----

// Same shape, but block b starts at column 16*b: there is a gap between
// consecutive blocks, so they are not one run and the block form stays.

// CHECK-LABEL: func.func @blocks_with_gaps
// CHECK: upmem.scatter_blocks
// CHECK-NOT: upmem.scatter_on_array
module {
  func.func @blocks_with_gaps(%arg0: memref<128x64xi32>) {
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<128x4>
    upmem.scatter_blocks %arg0[8 elts, affine_map<(d0, d1) -> (d0, d1 * 16)>, 4 blocks] onto @buf of %1
        : memref<128x64xi32> onto !upmem.hierarchy<128x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x8xi32, "mram">
      %wram = upmem.static_alloc(wram) : memref<4x8xi32, "wram">
      upmem.local_transfer %buf into %wram : memref<4x8xi32, "mram"> to memref<4x8xi32, "wram">
      upmem.return
    }
  }
}

// -----

// A map that ignores the DPU, over the whole host buffer: every DPU gets
// the same bytes from its start, which is exactly upmem.broadcast. No
// widening needed, so no new constant.

// CHECK-LABEL: func.func @whole_buffer
// CHECK: upmem.broadcast %arg0 onto @buf
// CHECK-NOT: upmem.scatter_on_array
module {
  func.func @whole_buffer(%arg0: memref<32xi32>) {
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<128x1>
    upmem.scatter_on_array %arg0[32 elts, affine_map<(d0) -> (0)>] onto @buf of %1
        : memref<32xi32> onto !upmem.hierarchy<128x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<32xi32, "mram">
      %wram = upmem.static_alloc(wram) : memref<32xi32, "wram">
      upmem.local_transfer %buf into %wram : memref<32xi32, "mram"> to memref<32xi32, "wram">
      upmem.return
    }
  }
}

// -----

// Each DPU reads its own row, so this is a real scatter and must stay one.

// CHECK-LABEL: func.func @per_dpu_rows
// CHECK: upmem.scatter_on_array
// CHECK-NOT: upmem.broadcast
module {
  func.func @per_dpu_rows(%arg0: memref<128x32xi32>) {
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<128x1>
    upmem.scatter_on_array %arg0[32 elts, affine_map<(d0) -> (d0, 0)>] onto @buf of %1
        : memref<128x32xi32> onto !upmem.hierarchy<128x1>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %buf = upmem.static_alloc @buf(mram) : memref<32xi32, "mram">
      %wram = upmem.static_alloc(wram) : memref<32xi32, "wram">
      upmem.local_transfer %buf into %wram : memref<32xi32, "mram"> to memref<32xi32, "wram">
      upmem.return
    }
  }
}

// -----

// A gather is never a broadcast, whatever its map says: two DPUs writing the
// same host region race. It may still collapse to the flat form.

// CHECK-DAG: #[[GROW:[^ ]*]] = affine_map<(d0) -> (d0, 0)>
// CHECK-LABEL: func.func @gather_never_broadcasts
// CHECK: upmem.gather_from_array %{{.*}}[32 elts, #[[GROW]]] from @buf
// CHECK-NOT: upmem.broadcast
module {
  func.func @gather_never_broadcasts(%arg0: memref<128x32xi32>) {
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<128x4>
    upmem.gather_blocks %arg0[8 elts, affine_map<(d0, d1) -> (d0, d1 * 8)>, 4 blocks] from @buf of %1
        : memref<128x32xi32> from !upmem.hierarchy<128x4>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x8xi32, "mram">
      %wram = upmem.static_alloc(wram) : memref<4x8xi32, "wram">
      upmem.local_transfer %buf into %wram : memref<4x8xi32, "mram"> to memref<4x8xi32, "wram">
      upmem.return
    }
  }
}

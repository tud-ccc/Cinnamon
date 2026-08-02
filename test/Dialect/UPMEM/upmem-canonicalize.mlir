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

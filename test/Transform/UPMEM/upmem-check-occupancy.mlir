// RUN: cinm-opt %s --split-input-file --upmem-check-occupancy -verify-diagnostics \
// RUN: | FileCheck %s

// The pass measures the program rather than modelling the configuration it was
// lowered from, so these cases are written as DPU programs with known
// allocation sizes. Capacities come from the accelerator: a v1A DPU has 64 MiB
// of MRAM, and 65536 - 8192 = 57344 bytes of WRAM once the runtime's share is
// deducted.

// CHECK-LABEL: func.func @fits
func.func @fits() {
  cinm.compute_block on accelerator #upmem.array<64x8, <type = v1A, dpus = 2048, tasklets = 24>> () {
    %d = upmem.alloc_dpus with program @kernels::@program : !upmem.hierarchy<64x8>
    upmem.free_dpus %d : !upmem.hierarchy<64x8>
    cinm.yield
  }
  return
}

module @kernels {
  // 256 bytes of MRAM, and 8 tasklets x (1024 reserve + 256) = 10240 of WRAM.
  // CHECK: upmem.dpu_program @program
  upmem.dpu_program @program() tasklets(8) {
    %mram = upmem.static_alloc @buf(mram) : memref<64xi32, #upmem.mram>
    %wram = memref.alloca() : memref<64xi32, #upmem.wram>
    upmem.local_transfer %mram into %wram
      : memref<64xi32, #upmem.mram> to memref<64xi32, #upmem.wram>
    upmem.return
  }
}

// -----

// Private WRAM is per tasklet: a buffer one tasklet could have, sixteen
// cannot. This is why the check needs the tasklet count and not just the
// buffer sizes.
func.func @private_wram_is_per_tasklet() {
  cinm.compute_block on accelerator #upmem.array<64x16, <type = v1A, dpus = 2048, tasklets = 24>> () {
    %d = upmem.alloc_dpus with program @kernels::@program : !upmem.hierarchy<64x16>
    upmem.free_dpus %d : !upmem.hierarchy<64x16>
    cinm.yield
  }
  return
}

module @kernels {
  // expected-error @below {{WRAM occupancy of 278528 bytes exceeds the 57344 bytes a DPU has (16 tasklets x 17408 bytes of stack, plus 0 bytes of static buffers)}}
  upmem.dpu_program @program() tasklets(16) {
    %wram = memref.alloca() : memref<4096xi32, #upmem.wram>
    upmem.return
  }
}

// -----

// Static WRAM is shared by a DPU's tasklets, so it is counted once -- but it
// is counted, on top of the per-tasklet stacks. Here the buffer alone exactly
// fills WRAM and the stacks are what push it over.
func.func @static_wram_is_shared() {
  cinm.compute_block on accelerator #upmem.array<64x2, <type = v1A, dpus = 2048, tasklets = 24>> () {
    %d = upmem.alloc_dpus with program @kernels::@program : !upmem.hierarchy<64x2>
    upmem.free_dpus %d : !upmem.hierarchy<64x2>
    cinm.yield
  }
  return
}

module @kernels {
  // expected-error @below {{WRAM occupancy of 59392 bytes exceeds the 57344 bytes a DPU has (2 tasklets x 1024 bytes of stack, plus 57344 bytes of static buffers)}}
  upmem.dpu_program @program() tasklets(2) {
    %wram = upmem.static_alloc @shared(wram) : memref<14336xi32, #upmem.wram>
    upmem.return
  }
}

// -----

func.func @mram_overflow() {
  cinm.compute_block on accelerator #upmem.array<64x1, <type = v1A, dpus = 2048, tasklets = 24>> () {
    %d = upmem.alloc_dpus with program @kernels::@program : !upmem.hierarchy<64x1>
    upmem.free_dpus %d : !upmem.hierarchy<64x1>
    cinm.yield
  }
  return
}

module @kernels {
  // 17 Mi elements x 4 bytes = 68 MiB against a 64 MiB bank.
  // expected-error @below {{MRAM occupancy of 71303168 bytes exceeds the 67108864 bytes a DPU has}}
  upmem.dpu_program @program() tasklets(1) {
    %mram = upmem.static_alloc @buf(mram) : memref<17825792xi32, #upmem.mram>
    upmem.return
  }
}

// -----

// A program nothing loads is dead -- --upmem-dedup-kernels can leave one
// behind -- and occupies nothing on the device, however large it is.
// CHECK-LABEL: upmem.dpu_program @never_loaded
module @kernels {
  upmem.dpu_program @never_loaded() tasklets(24) {
    %mram = upmem.static_alloc @buf(mram) : memref<17825792xi32, #upmem.mram>
    %wram = memref.alloca() : memref<1048576xi32, #upmem.wram>
    upmem.return
  }
}

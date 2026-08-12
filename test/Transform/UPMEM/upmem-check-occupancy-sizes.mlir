// RUN: cinm-opt %s --upmem-check-occupancy='mram-size=4096 wram-size=16384' | FileCheck %s
// RUN: cinm-opt %s --upmem-check-occupancy -verify-diagnostics

// Capacities can be given explicitly, which is what lets the check run on IR
// with no accelerator in scope. Given neither those options nor an
// accelerator, the pass says so rather than checking nothing: a feasibility
// check that silently passes everything is worse than no check at all.
//
// (The `expected-error` below is inert on the first RUN line, which does not
// pass -verify-diagnostics and does supply both capacities.)

// CHECK-LABEL: func.func @no_accelerator_in_scope
func.func @no_accelerator_in_scope() {
  %d = upmem.alloc_dpus : !upmem.hierarchy<64x8>
  // expected-error @below {{cannot check whether @program fits: no accelerator with an mram and a wram level is in scope, and the mram-size/wram-size options do not supply both capacities}}
  upmem.load_program @kernels::@program on %d : !upmem.hierarchy<64x8>
  upmem.free_dpus %d : !upmem.hierarchy<64x8>
  return
}

module @kernels {
  // 8 tasklets x (1024 reserve + 256 of private WRAM) = 10240 bytes, within
  // the 16384 the first RUN line allows.
  // CHECK: upmem.dpu_program @program
  upmem.dpu_program @program() tasklets(8) {
    %wram = memref.alloca() : memref<64xi32, #upmem.wram>
    upmem.return
  }
}

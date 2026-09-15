// RUN: cinm-opt %s --split-input-file --upmem-hoist-load-programs | FileCheck %s

// A set that only ever holds one program is loaded once, right after its
// allocation -- the per-iteration and per-member reloads disappear. This is
// the post-unwrap shape of a merged group: several former compute blocks
// launching the same program on the same forwarded set.

// CHECK-LABEL: func.func @uniform
func.func @uniform(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[SET:.*]] = upmem.alloc_dpus
  // CHECK-NEXT: upmem.load_program @kernels::@program on %[[SET]]
  %set = upmem.alloc_dpus : !upmem.hierarchy<4x1>
  // CHECK-NOT: upmem.load_program
  scf.for %i = %c0 to %n step %c1 {
    upmem.load_program @kernels::@program on %set : !upmem.hierarchy<4x1>
    upmem.wait_for %set : !upmem.hierarchy<4x1>
  }
  upmem.load_program @kernels::@program on %set : !upmem.hierarchy<4x1>
  upmem.wait_for %set : !upmem.hierarchy<4x1>
  upmem.free_dpus %set : !upmem.hierarchy<4x1>
  return
}

module @kernels {
  upmem.dpu_program @program() tasklets(1) {
    upmem.return
  }
}

// -----

// A timeshared set alternates between programs; which one is resident at a
// launch is decided by the placement of the loads, so they must all stay
// exactly where they are.

// CHECK-LABEL: func.func @timeshared
func.func @timeshared() {
  // CHECK: upmem.alloc_dpus
  %set = upmem.alloc_dpus : !upmem.hierarchy<4x1>
  // CHECK-NEXT: upmem.load_program @kernels::@a
  // CHECK-NEXT: upmem.wait_for
  // CHECK-NEXT: upmem.load_program @kernels::@b
  // CHECK-NEXT: upmem.wait_for
  // CHECK-NEXT: upmem.load_program @kernels::@a
  // CHECK-NEXT: upmem.wait_for
  upmem.load_program @kernels::@a on %set : !upmem.hierarchy<4x1>
  upmem.wait_for %set : !upmem.hierarchy<4x1>
  upmem.load_program @kernels::@b on %set : !upmem.hierarchy<4x1>
  upmem.wait_for %set : !upmem.hierarchy<4x1>
  upmem.load_program @kernels::@a on %set : !upmem.hierarchy<4x1>
  upmem.wait_for %set : !upmem.hierarchy<4x1>
  upmem.free_dpus %set : !upmem.hierarchy<4x1>
  return
}

module @kernels {
  upmem.dpu_program @a() tasklets(1) {
    upmem.return
  }
  upmem.dpu_program @b() tasklets(1) {
    upmem.return
  }
}

// -----

// A set that enters as a block argument has its allocation elsewhere; there
// is no place to hoist to, so the load stays.

// CHECK-LABEL: func.func @forwarded
func.func @forwarded(%set: !upmem.hierarchy<4x1>) {
  // CHECK: upmem.wait_for
  // CHECK-NEXT: upmem.load_program @kernels::@program
  upmem.wait_for %set : !upmem.hierarchy<4x1>
  upmem.load_program @kernels::@program on %set : !upmem.hierarchy<4x1>
  upmem.wait_for %set : !upmem.hierarchy<4x1>
  return
}

module @kernels {
  upmem.dpu_program @program() tasklets(1) {
    upmem.return
  }
}

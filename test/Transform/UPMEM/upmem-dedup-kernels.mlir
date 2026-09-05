// RUN: cinm-opt %s --upmem-dedup-kernels | FileCheck %s

// Every conversion-emitted kernel module names its program @program, so
// equivalence and load retargeting must key on the program op, not on its
// symbol name. @a and @c hold identical programs and fold into one; @b's
// program differs and must keep both its body and its load. (Keying by
// name used to retarget every load to whichever equivalence class the
// walk recorded last under the shared name -- a miscompile as soon as a
// module held two distinct programs.)

// CHECK-LABEL: func.func @host
func.func @host() {
  // CHECK: upmem.load_program @a::@program
  // CHECK: upmem.load_program @b::@program
  // CHECK: upmem.load_program @a::@program
  %s0 = upmem.alloc_dpus : !upmem.hierarchy<2x1>
  %s1 = upmem.alloc_dpus : !upmem.hierarchy<2x1>
  %s2 = upmem.alloc_dpus : !upmem.hierarchy<2x1>
  upmem.load_program @a::@program on %s0 : !upmem.hierarchy<2x1>
  upmem.load_program @b::@program on %s1 : !upmem.hierarchy<2x1>
  upmem.load_program @c::@program on %s2 : !upmem.hierarchy<2x1>
  upmem.free_dpus %s0 : !upmem.hierarchy<2x1>
  upmem.free_dpus %s1 : !upmem.hierarchy<2x1>
  upmem.free_dpus %s2 : !upmem.hierarchy<2x1>
  return
}

// CHECK: module @a
// CHECK: upmem.dpu_program @program
module @a {
  upmem.dpu_program @program() tasklets(1) {
    %buf = upmem.static_alloc @buf(mram) : memref<16xi32, #upmem.mram>
    upmem.return
  }
}

// CHECK: module @b
// CHECK: upmem.dpu_program @program
module @b {
  upmem.dpu_program @program() tasklets(1) {
    %buf = upmem.static_alloc @buf(mram) : memref<32xi32, #upmem.mram>
    upmem.return
  }
}

// The duplicate of @a::@program is erased; its module stays behind empty.
// CHECK: module @c
// CHECK-NOT: upmem.dpu_program
module @c {
  upmem.dpu_program @program() tasklets(1) {
    %buf = upmem.static_alloc @buf(mram) : memref<16xi32, #upmem.mram>
    upmem.return
  }
}

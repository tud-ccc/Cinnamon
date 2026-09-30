// RUN: cinm-translate --allow-unregistered-dialect --mlir-to-upmem-cpp %s | FileCheck %s

// A kernel module that holds no kernel -- the whole program stayed on the
// host, or every kernel folded away -- translates to a header that names
// nothing, which is what cinm-compile-dpu expects for "no binary to build".

// CHECK: UPMEM-TRANSLATE:{{ *$}}
// CHECK-NOT: main
module {
  module @dpu_kernels {
  }
}

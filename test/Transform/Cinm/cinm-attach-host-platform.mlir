// RUN: cinm-opt %s --split-input-file --cinm-attach-host-platform="%cinm_host" 2>&1 | FileCheck %s
// RUN: cinm-opt %s --split-input-file --cinm-attach-host-platform 2>&1 | FileCheck %s --check-prefix=DEFAULTS
// RUN: cinm-opt %s --split-input-file --cinm-attach-host-platform="ops-per-second=8.6e12" 2>&1 | FileCheck %s --check-prefix=SOME

// A run that states the host warns about nothing.
// CHECK-NOT: warning

// One that leaves options at their defaults names each of them, once per
// module, with the value it used.
// DEFAULTS: warning: cinm-attach-host-platform: host parameters left at their bench-machine defaults (ops-per-second=2.100000e+12, dram-bytes-per-second=2.170000e+10, scalar-op-ns=3.000000e+00, vector-op-ns=5.000000e-01, vector-bytes=6.400000e+01, stream-bytes-per-second=1.080000e+10, copy-bytes-per-second=6.300000e+08)
// SOME: warning: cinm-attach-host-platform: host parameters left at their bench-machine defaults (dram-bytes-per-second=2.170000e+10,
// SOME-SAME: copy-bytes-per-second=6.300000e+08)
// SOME-NOT: ops-per-second=
// SOME: #cinm.host_platform<ops_per_second = 8.600000e+12, dram_bytes_per_second = 2.170000e+10

// A function that offloads but names no host gets the one the options give,
// first in its list.

// CHECK-LABEL: func.func @no_host_listed
// CHECK-SAME:    cinm.available_platforms = [#cinm.host_platform<ops_per_second = 2.100000e+12, dram_bytes_per_second = 2.170000e+10, scalar_op_ns = 3.000000e+00, vector_op_ns = 5.000000e-01, vector_bytes = 6.400000e+01, stream_bytes_per_second = 1.080000e+10, copy_bytes_per_second = 6.300000e+08>, #upmem]
// A compute op's own list is not touched: there a host entry would mean a
// host block, and the host in scope is found through the function's.
// CHECK:         cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]}
#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 24>
func.func @no_host_listed(%A: tensor<8x1024xi32>, %x: tensor<1024xi32>) -> tensor<8xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
    cinm.yield %g : tensor<8xi32>
  }
  return %r : tensor<8xi32>
}

// -----

// A function that already names a host keeps it: the program stated its
// machine.

// CHECK-LABEL: func.func @host_listed
// CHECK-SAME:    cinm.available_platforms = [#upmem, #cinm.host_platform<ops_per_second = 1.000000e+09, dram_bytes_per_second = 1.000000e+09,
#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 24>
func.func @host_listed(%A: tensor<8x1024xi32>, %x: tensor<1024xi32>) -> tensor<8xi32>
    attributes {cinm.available_platforms = [#upmem, #cinm.host_platform<ops_per_second = 1.0e9, dram_bytes_per_second = 1.0e9, scalar_op_ns = 3.0, vector_op_ns = 0.5, vector_bytes = 64, stream_bytes_per_second = 10.8e9, copy_bytes_per_second = 0.63e9>]} {
  %r = cinm.op.gemv %A, %x : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// -----

// A function without a list of its own whose compute op lists its
// accelerator still runs on a host: it gets a list holding only that.

// CHECK-LABEL: func.func @block_lists_only
// CHECK-SAME:    cinm.available_platforms = [#cinm.host_platform<ops_per_second = 2.100000e+12,
// CHECK:         cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]}
#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 24>
func.func @block_lists_only(%A: tensor<8x1024xi32>, %x: tensor<1024xi32>) -> tensor<8xi32> {
  %r = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
    cinm.yield %g : tensor<8xi32>
  }
  return %r : tensor<8xi32>
}

// -----

// A function without any list offloads nothing, and gets none.

// CHECK-LABEL: func.func @no_list
// CHECK-NOT:     cinm.available_platforms
func.func @no_list(%a: i32) -> i32 {
  return %a : i32
}

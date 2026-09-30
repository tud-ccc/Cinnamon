// RUN: cinm-opt %s --split-input-file --cinm-assign-platforms="host-ops-per-second=8.6e12" | FileCheck %s

// The host the verdicts were priced against, overrides included, is written
// onto the function, so every later pass that prices host code reads the
// same machine instead of the default one.

// CHECK-LABEL: func.func @no_host_listed
// CHECK-SAME:    cinm.available_platforms = [#cinm.host_platform<ops_per_second = 8.600000e+12, dram_bytes_per_second = 2.300000e+10, {{[^>]*}}>, #upmem]
#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 24>
func.func @no_host_listed(%A: tensor<8x1024xi32>, %x: tensor<1024xi32>) -> tensor<8xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.op.gemv %A, %x : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// -----

// A host the function already lists is replaced, keeping the parameters the
// override does not touch.

// CHECK-LABEL: func.func @host_listed
// CHECK-SAME:    cinm.available_platforms = [#cinm.host_platform<ops_per_second = 8.600000e+12, dram_bytes_per_second = 1.000000e+09, {{[^>]*}}>, #upmem]
// CHECK-NOT:     #cinm.host_platform<ops_per_second = 1.0
#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 24>
func.func @host_listed(%A: tensor<8x1024xi32>, %x: tensor<1024xi32>) -> tensor<8xi32>
    attributes {cinm.available_platforms = [#upmem, #cinm.host_platform<ops_per_second = 1.0e9, dram_bytes_per_second = 1.0e9>]} {
  %r = cinm.op.gemv %A, %x : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
  return %r : tensor<8xi32>
}

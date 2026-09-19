// RUN: cinm-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The bare attribute is the bench machine, and prints bare.

// CHECK-LABEL: func.func @bench_machine
// CHECK-SAME:    cinm.available_platforms = [#cinm.host_platform]
func.func @bench_machine() attributes {cinm.available_platforms = [#cinm.host_platform]} {
  return
}

// -----

// Only the parameters that differ from the default print, in declaration
// order; one given at its default value is not a difference.

// CHECK-LABEL: func.func @int8_host
// CHECK-SAME:    #cinm.host_platform<ops_per_second = 8.600000e+12, copy_bytes_per_second = 1.000000e+09>
func.func @int8_host() attributes {cinm.available_platforms = [#cinm.host_platform<copy_bytes_per_second = 1.0e9, vector_bytes = 64, ops_per_second = 8.6e12>]} {
  return
}

// -----

// expected-error @+1 {{unknown host platform parameter 'flops'}}
func.func @unknown() attributes {cinm.available_platforms = [#cinm.host_platform<flops = 1.0>]} {
  return
}

// -----

// expected-error @+1 {{host platform parameter 'vector_op_ns' given twice}}
func.func @twice() attributes {cinm.available_platforms = [#cinm.host_platform<vector_op_ns = 1.0, vector_op_ns = 2.0>]} {
  return
}

// -----

// A zero rate would divide the cost model by zero.

// expected-error @+1 {{host platform parameter 'stream_bytes_per_second' must be positive}}
func.func @zero() attributes {cinm.available_platforms = [#cinm.host_platform<stream_bytes_per_second = 0.0>]} {
  return
}

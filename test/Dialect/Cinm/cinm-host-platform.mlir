// RUN: cinm-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Every parameter is printed, in declaration order.

// CHECK-LABEL: func.func @bench_machine
// CHECK-SAME:    cinm.available_platforms = [#cinm.host_platform<ops_per_second = 2.100000e+12, dram_bytes_per_second = 2.170000e+10, scalar_op_ns = 3.000000e+00, vector_op_ns = 5.000000e-01, vector_bytes = 6.400000e+01, stream_bytes_per_second = 1.080000e+10, copy_bytes_per_second = 6.300000e+08>]
func.func @bench_machine() attributes {cinm.available_platforms = [#cinm.host_platform<ops_per_second = 2.1e12, dram_bytes_per_second = 21.7e9, scalar_op_ns = 3.0, vector_op_ns = 0.5, vector_bytes = 64, stream_bytes_per_second = 10.8e9, copy_bytes_per_second = 0.63e9>]} {
  return
}

// -----

// Parameters are given by key in any order, an integer literal as well as a
// float.

// CHECK-LABEL: func.func @any_order
// CHECK-SAME:    #cinm.host_platform<ops_per_second = 8.600000e+12, dram_bytes_per_second = 2.170000e+10, scalar_op_ns = 3.000000e+00, vector_op_ns = 5.000000e-01, vector_bytes = 6.400000e+01, stream_bytes_per_second = 1.080000e+10, copy_bytes_per_second = 1.000000e+09>
func.func @any_order() attributes {cinm.available_platforms = [#cinm.host_platform<copy_bytes_per_second = 1.0e9, vector_bytes = 64, ops_per_second = 8.6e12, scalar_op_ns = 3, stream_bytes_per_second = 10.8e9, dram_bytes_per_second = 21.7e9, vector_op_ns = 0.5>]} {
  return
}

// -----

// No parameter has a default: a host states the machine it describes.

// expected-error @+2 {{expected ','}}
// expected-error @+1 {{#cinm.host_platform takes exactly these parameters, each once, in any order}}
func.func @missing() attributes {cinm.available_platforms = [#cinm.host_platform<ops_per_second = 2.1e12, dram_bytes_per_second = 21.7e9>]} {
  return
}

// -----

// expected-error @+2 {{duplicate or unknown struct parameter name: flops}}
// expected-error @+1 {{#cinm.host_platform takes exactly these parameters, each once, in any order}}
func.func @unknown() attributes {cinm.available_platforms = [#cinm.host_platform<flops = 1.0, ops_per_second = 2.1e12, dram_bytes_per_second = 21.7e9, scalar_op_ns = 3.0, vector_op_ns = 0.5, vector_bytes = 64, stream_bytes_per_second = 10.8e9, copy_bytes_per_second = 0.63e9>]} {
  return
}

// -----

// expected-error @+2 {{duplicate or unknown struct parameter name: vector_op_ns}}
// expected-error @+1 {{#cinm.host_platform takes exactly these parameters, each once, in any order}}
func.func @twice() attributes {cinm.available_platforms = [#cinm.host_platform<vector_op_ns = 2.0, ops_per_second = 2.1e12, dram_bytes_per_second = 21.7e9, scalar_op_ns = 3.0, vector_op_ns = 0.5, vector_bytes = 64, stream_bytes_per_second = 10.8e9, copy_bytes_per_second = 0.63e9>]} {
  return
}

// -----

// A zero rate would divide the cost model by zero.

// expected-error @+2 {{host platform parameter 'stream_bytes_per_second' must be positive}}
// expected-error @+1 {{#cinm.host_platform takes exactly these parameters, each once, in any order}}
func.func @zero() attributes {cinm.available_platforms = [#cinm.host_platform<ops_per_second = 2.1e12, dram_bytes_per_second = 21.7e9, scalar_op_ns = 3.0, vector_op_ns = 0.5, vector_bytes = 64, stream_bytes_per_second = 0.0, copy_bytes_per_second = 0.63e9>]} {
  return
}

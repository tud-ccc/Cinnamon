// RUN: cinm-opt %s --cinm-attach-host-platform="%cinm_host" --cinm-assign-platforms | FileCheck %s

#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 24>

// The function lists the host its ops were priced against, ahead of the
// platforms it advertised.
// CHECK-LABEL: @gemm_gets_wrapped
// CHECK-SAME:  cinm.available_platforms = [#cinm.host_platform<{{[^>]*}}>, #upmem]
// CHECK:       cinm.compute -> tensor<8x128xi32> attributes {cinm.available_platforms = [#upmem]}
// CHECK:       cinm.op.gemm
func.func @gemm_gets_wrapped(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
  return %r : tensor<8x128xi32>
}

// CHECK-LABEL: @gemv_gets_wrapped
// CHECK:       cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]}
// CHECK:       cinm.op.gemv
func.func @gemv_gets_wrapped(%A: tensor<8x1024xi32>, %x: tensor<1024xi32>) -> tensor<8xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.op.gemv %A, %x : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// CHECK-LABEL: @elementwise_gets_wrapped
// CHECK:       cinm.compute -> tensor<4xi32> attributes {cinm.available_platforms = [#upmem]}
// CHECK:       cinm.op.elementwise add
func.func @elementwise_gets_wrapped(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.op.elementwise add %a, %b : tensor<4xi32>
  return %r : tensor<4xi32>
}

// Function without cinm.available_platforms: pass is a no-op.
// CHECK-LABEL: @no_platforms_no_wrap
// CHECK-NOT:   cinm.compute
// CHECK:       cinm.op.gemm
func.func @no_platforms_no_wrap(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {
  %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
  return %r : tensor<8x128xi32>
}

// Host platform never handles any op.
// CHECK-LABEL: @host_platform_no_wrap
// CHECK-NOT:   cinm.compute
// CHECK:       cinm.op.gemm
func.func @host_platform_no_wrap(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32>
    attributes {cinm.available_platforms = [#cinm.host_platform<ops_per_second = 2.1e12, dram_bytes_per_second = 21.7e9, scalar_op_ns = 3.0, vector_op_ns = 0.5, vector_bytes = 64, stream_bytes_per_second = 10.8e9, copy_bytes_per_second = 0.63e9>]} {
  %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
  return %r : tensor<8x128xi32>
}

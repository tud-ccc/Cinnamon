// RUN: cinm-opt %s --cinm-assign-platforms | FileCheck %s

#upmem = #upmem.platform<type=v1A, dimensions = 8x64x24>

// CHECK-LABEL: @gemm_gets_wrapped
// CHECK-SAME:  cinm.available_platforms = [#upmem]
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
    attributes {cinm.available_platforms = [#cinm.host_platform]} {
  %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
  return %r : tensor<8x128xi32>
}

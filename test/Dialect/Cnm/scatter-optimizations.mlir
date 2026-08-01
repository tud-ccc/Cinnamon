// RUN: cinm-opt %s --cnm-scatter-optimizations --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// A linalg.fill of a constant is uniform, so every workgroup element receives
// the same bytes whatever the scatter map says. The 4x2 host tiles collapse to
// one, and the map loses its results.

// CHECK: #[[BC:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-LABEL: func.func @fill
// CHECK:       %[[TILE:.*]] = arith.constant dense<0> : tensor<1x8xi32>
// CHECK:       cnm.scatter %[[TILE]] into %{{.*}}[#[[BC]]] of %{{.*}} : tensor<1x8xi32> into
// CHECK-NOT:   linalg.fill
func.func @fill() {
  %c0 = arith.constant 0 : i32
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.alloc() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %empty = tensor.empty() : tensor<4x2x1x8xi32>
  %filled = linalg.fill ins(%c0 : i32) outs(%empty : tensor<4x2x1x8xi32>) -> tensor<4x2x1x8xi32>
  cnm.scatter %filled into %buf[#map] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// Same for a splat constant, with a non-zero value to check it is carried over.

// CHECK: #[[BC:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-LABEL: func.func @splat_constant
// CHECK:       %[[TILE:.*]] = arith.constant dense<7> : tensor<1x8xi32>
// CHECK:       cnm.scatter %[[TILE]] into %{{.*}}[#[[BC]]] of %{{.*}} : tensor<1x8xi32> into
func.func @splat_constant() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.alloc() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<7> : tensor<4x2x1x8xi32>
  cnm.scatter %cst into %buf[#map] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// After bufferization the uniform value is a constant global; the tile has to
// be one too, so the pass emits a smaller global rather than an
// arith.constant.

// CHECK: #[[BC:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK: memref.global "private" constant @[[TILE:.*]] : memref<1x8xf32> = dense<1.000000e+00>
// CHECK-LABEL: func.func @get_global
// CHECK:       %[[T:.*]] = memref.get_global @[[TILE]] : memref<1x8xf32>
// CHECK:       cnm.scatter %[[T]] into %{{.*}}[#[[BC]]] of %{{.*}} : memref<1x8xf32> into
memref.global "private" constant @ones : memref<4x2x1x8xf32> = dense<1.000000e+00>
func.func @get_global() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.alloc() for %wg : !cnm.buffer<1x8xf32 on #wg>
  %g = memref.get_global @ones : memref<4x2x1x8xf32>
  cnm.scatter %g into %buf[#map] of %wg
      : memref<4x2x1x8xf32> into !cnm.buffer<1x8xf32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// A constant that is not a splat carries data the workgroup elements can tell
// apart, so it must be left alone. So must a value with no known contents.

// CHECK-LABEL: func.func @not_uniform
// CHECK:       cnm.scatter %{{.*}}[#map] of %{{.*}} : tensor<2x1x2xi32> into
// CHECK:       cnm.scatter %{{.*}}[#map] of %{{.*}} : memref<4x2x1x8xi32> into
func.func @not_uniform() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>

  %buf = cnm.alloc() for %wg : !cnm.buffer<2xi32 on #wg>
  %cst = arith.constant dense<[[[0, 1]], [[2, 3]]]> : tensor<2x1x2xi32>
  cnm.scatter %cst into %buf[#map] of %wg
      : tensor<2x1x2xi32> into !cnm.buffer<2xi32 on #wg>

  %buf2 = cnm.alloc() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %opaque = memref.alloc() : memref<4x2x1x8xi32>
  cnm.scatter %opaque into %buf2[#map] of %wg
      : memref<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>

  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#bc = affine_map<(d0, d1, d2) -> ()>
#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// A scatter that is already a single-tile broadcast is a fixpoint.

// CHECK-LABEL: func.func @already_broadcast
// CHECK:       %[[TILE:.*]] = arith.constant dense<0> : tensor<1x8xi32>
// CHECK:       cnm.scatter %[[TILE]] into
func.func @already_broadcast() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.alloc() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<0> : tensor<1x8xi32>
  cnm.scatter %cst into %buf[#bc] of %wg
      : tensor<1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

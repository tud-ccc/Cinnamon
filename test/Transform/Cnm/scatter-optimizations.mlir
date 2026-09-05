// RUN: cinm-opt %s --cnm-scatter-optimizations --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// A linalg.fill of a constant is uniform, so every workgroup element receives
// the same bytes whatever the scatter map says. The 4x2 host tiles collapse to
// one, and the map loses its results.

// CHECK: #[[BC:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-LABEL: func.func @fill
// CHECK:       %[[TILE:.*]] = arith.constant dense<0> : tensor<1x8xi32>
// CHECK:       cnm.scatter %[[TILE]] into %{{.*}}[#[[BC]]] of %{{.*}} : tensor<1x8xi32> into
// CHECK-NOT:   linalg.fill
func.func @fill() {
  %c0 = arith.constant 0 : i32
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %empty = tensor.empty() : tensor<4x2x1x8xi32>
  %filled = linalg.fill ins(%c0 : i32) outs(%empty : tensor<4x2x1x8xi32>) -> tensor<4x2x1x8xi32>
  cnm.scatter %filled into %buf[#map] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// Same for a splat constant, with a non-zero value to check it is carried over.

// CHECK: #[[BC:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-LABEL: func.func @splat_constant
// CHECK:       %[[TILE:.*]] = arith.constant dense<7> : tensor<1x8xi32>
// CHECK:       cnm.scatter %[[TILE]] into %{{.*}}[#[[BC]]] of %{{.*}} : tensor<1x8xi32> into
func.func @splat_constant() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<7> : tensor<4x2x1x8xi32>
  cnm.scatter %cst into %buf[#map] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// After bufferization the uniform value is a constant global; the tile has to
// be one too, so the pass emits a smaller global rather than an
// arith.constant.

// CHECK: #[[BC:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK: memref.global "private" constant @[[TILE:.*]] : memref<1x8xf32> = dense<1.000000e+00>
// CHECK-LABEL: func.func @get_global
// CHECK:       %[[T:.*]] = memref.get_global @[[TILE]] : memref<1x8xf32>
// CHECK:       cnm.scatter %[[T]] into %{{.*}}[#[[BC]]] of %{{.*}} : memref<1x8xf32> into
memref.global "private" constant @ones : memref<4x2x1x8xf32> = dense<1.000000e+00>
func.func @get_global() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xf32 on #wg>
  %g = memref.get_global @ones : memref<4x2x1x8xf32>
  cnm.scatter %g into %buf[#map] of %wg
      : memref<4x2x1x8xf32> into !cnm.buffer<1x8xf32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// A constant that is not a splat carries data the workgroup elements can tell
// apart, so it must be left alone. So must a value with no known contents.

// CHECK-LABEL: func.func @not_uniform
// CHECK:       cnm.scatter %{{.*}}[#map{{[0-9]*}}] of %{{.*}} : tensor<4x2x2xi32> into
// CHECK:       cnm.scatter %{{.*}}[#map{{[0-9]*}}] of %{{.*}} : memref<4x2x1x8xi32> into
func.func @not_uniform() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>

  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<2xi32 on #wg>
  %cst = arith.constant dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]],
                              [[8, 9], [10, 11]], [[12, 13], [14, 15]]]> : tensor<4x2x2xi32>
  cnm.scatter %cst into %buf[affine_map<(d0, d1, d2) -> (d0, d1, d2)>] of %wg
      : tensor<4x2x2xi32> into !cnm.buffer<2xi32 on #wg>

  %buf2 = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %opaque = memref.alloc() : memref<4x2x1x8xi32>
  cnm.scatter %opaque into %buf2[#map] of %wg
      : memref<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>

  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#bc = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// A scatter that is already a single-tile broadcast is a fixpoint.

// CHECK-LABEL: func.func @already_broadcast
// CHECK:       %[[TILE:.*]] = arith.constant dense<0> : tensor<1x8xi32>
// CHECK:       cnm.scatter %[[TILE]] into
func.func @already_broadcast() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<0> : tensor<1x8xi32>
  cnm.scatter %cst into %buf[#bc] of %wg
      : tensor<1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 16, d0 * 256 + d1 * 32 + d3 - (d0 floordiv 16) * 4096)>
#wg = #upmem.array<512x8, <type = v1A, dpus = 2048, tasklets = 24>>

// The split-reduction identity seed, as --convert-linalg-to-cnm leaves it: a
// pointwise map, and a host value that has the same rank as the buffer while
// being 4096 times its size. Rank alone says nothing about how much is being
// transferred.

// CHECK: #[[BC:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-LABEL: func.func @seed_same_rank_as_buffer
// CHECK:       %[[TILE:.*]] = arith.constant dense<0> : tensor<1x32xi32>
// CHECK:       cnm.scatter %[[TILE]] into %{{.*}}[#[[BC]]] of %{{.*}} : tensor<1x32xi32> into
// CHECK-NOT:   linalg.fill
func.func @seed_same_rank_as_buffer() {
  %c0 = arith.constant 0 : i32
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<1x32xi32 on #wg>
  %empty = tensor.empty() : tensor<32x4096xi32>
  %seed = linalg.fill ins(%c0 : i32) outs(%empty : tensor<32x4096xi32>) -> tensor<32x4096xi32>
  cnm.scatter %seed into %buf[#map] of %wg
      : tensor<32x4096xi32> into !cnm.buffer<1x32xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// The buffer feeds a launch, so the constant does not have to be transferred
// at all: the leaves write it themselves, in parallel, and the host keeps its
// bandwidth. The scatter disappears rather than shrinking to a broadcast.

// CHECK-LABEL: func.func @device_init
// CHECK-NOT:   cnm.scatter
// CHECK:       cnm.launch %{{.*}} ins(%[[A:.*]] = %{{.*}}) outs(%[[B:.*]] = %{{.*}})
// CHECK-NEXT:    %[[C:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    linalg.fill ins(%[[C]] : i32) outs(%[[B]] : memref<1x8xi32>)
// CHECK:         linalg.add
// CHECK-NOT:   cnm.scatter
func.func @device_init() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %acc = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<0> : tensor<4x2x1x8xi32>
  cnm.scatter %cst into %acc[#map] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.launch %wg ins(%a = %in : <1x8xi32>) outs(%b = %acc : <1x8xi32>) on !cnm.workgroup<#wg> {
    linalg.add ins(%a, %b : memref<1x8xi32>, memref<1x8xi32>) outs(%b : memref<1x8xi32>)
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// The scatter seeds an accumulator once and the launch runs many times, so
// filling in the body would reset it on every trip. Only the broadcast fires.

// CHECK-LABEL: func.func @seed_outside_loop
// CHECK:       cnm.scatter %{{.*}} : tensor<1x8xi32> into
// CHECK:       scf.for
// CHECK-NOT:     linalg.fill
// CHECK:         cnm.launch
func.func @seed_outside_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %acc = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<0> : tensor<4x2x1x8xi32>
  cnm.scatter %cst into %acc[#map] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  scf.for %i = %c0 to %c4 step %c1 {
    cnm.launch %wg ins(%a = %in : <1x8xi32>) outs(%b = %acc : <1x8xi32>) on !cnm.workgroup<#wg> {
      linalg.add ins(%a, %b : memref<1x8xi32>, memref<1x8xi32>) outs(%b : memref<1x8xi32>)
    }
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#gmap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// The host reads the buffer before the launch does, so it would observe an
// uninitialized buffer if the fill moved into the body.

// CHECK-LABEL: func.func @read_before_launch
// CHECK:       cnm.scatter %{{.*}} : tensor<1x8xi32> into
// CHECK:       cnm.gather
// CHECK:       cnm.launch
// CHECK-NOT:     linalg.fill
func.func @read_before_launch(%out: tensor<4x2x1x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %acc = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<0> : tensor<4x2x1x8xi32>
  cnm.scatter %cst into %acc[#map] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  %g = cnm.gather %acc[#gmap] of %wg into %out
      : !cnm.buffer<1x8xi32 on #wg> into tensor<4x2x1x8xi32>
  cnm.launch %wg ins(%a = %in : <1x8xi32>) outs(%b = %acc : <1x8xi32>) on !cnm.workgroup<#wg> {
    linalg.add ins(%a, %b : memref<1x8xi32>, memref<1x8xi32>) outs(%b : memref<1x8xi32>)
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// A fill spelled as a linalg.generic broadcasting a 0-d constant tensor --
// how the cinm lowering emits an accumulator's zero init. It is as uniform as
// a linalg.fill, so the same rewrite applies: the scatter disappears and the
// leaves initialize their share themselves.

// CHECK-LABEL: func.func @device_init_generic_fill
// CHECK-NOT:   cnm.scatter
// CHECK:       cnm.launch %{{.*}} ins(%[[A:.*]] = %{{.*}}) outs(%[[B:.*]] = %{{.*}})
// CHECK-NEXT:    %[[C:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    linalg.fill ins(%[[C]] : i32) outs(%[[B]] : memref<1x8xi32>)
// CHECK:         linalg.add
// CHECK-NOT:   cnm.scatter
func.func @device_init_generic_fill() {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %in = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %acc = cnm.declare_buffer() for %wg : !cnm.buffer<1x8xi32 on #wg>
  %cst = arith.constant dense<0> : tensor<i32>
  %empty = tensor.empty() : tensor<4x2x1x8xi32>
  %init = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%cst : tensor<i32>) outs(%empty : tensor<4x2x1x8xi32>) {
    ^bb0(%in_elem: i32, %out_elem: i32):
      linalg.yield %in_elem : i32
  } -> tensor<4x2x1x8xi32>
  cnm.scatter %init into %acc[affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>] of %wg
      : tensor<4x2x1x8xi32> into !cnm.buffer<1x8xi32 on #wg>
  cnm.launch %wg ins(%a = %in : <1x8xi32>) outs(%b = %acc : <1x8xi32>) on !cnm.workgroup<#wg> {
    linalg.add ins(%a, %b : memref<1x8xi32>, memref<1x8xi32>) outs(%b : memref<1x8xi32>)
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

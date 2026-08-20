// RUN: cinm-opt %s --cinm-expand-compute-scope --split-input-file | FileCheck %s

// Slicing ops are NOT absorbed: after loop unrolling their offsets are
// per-iteration constants, and baking those into the block body would make
// structurally identical blocks look distinct to the graph scheduler's
// signature-based deduplication. Insertions of the results stay outside for
// the same reason. The block keeps its sliced operand and its own result.

// CHECK-LABEL: func @slices_stay_out
// CHECK:       %[[E:.*]] = tensor.extract_slice %{{.*}}[8] [8] [1]
// CHECK:       %[[R:.*]] = cinm.compute_block (%{{.*}} = %[[E]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NOT:     tensor.extract_slice
// CHECK:       tensor.insert_slice %[[R]]
func.func @slices_stay_out(%s: tensor<64xf32>, %d: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.extract_slice %s[8] [8] [1] : tensor<64xf32> to tensor<8xf32>
  %r = cinm.compute_block (%x = %e : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.elementwise mul %x, %x: tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %i = tensor.insert_slice %r into %d[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %i : tensor<64xf32>
}

// -----

// Producers that only initialize data are pulled in transitively: the fill
// and the empty tensor it writes to move in as a whole, and only the
// splatted value crosses the block boundary.

// CHECK-LABEL: func @splat_and_fill
// CHECK-SAME:    (%[[A:.*]]: tensor<8x8xf32>, %[[S:.*]]: f32)
// CHECK-NOT:   tensor.splat
// CHECK-NOT:   linalg.fill
// CHECK:       %[[E:.*]] = tensor.empty()
// CHECK:       cinm.compute_block (%{{.*}} = %[[A]] : tensor<8x8xf32>, %[[BS:.*]] = %[[S]] : f32, %[[E2:.*]] = %[[E]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK:         %[[SP:.*]] = tensor.splat %[[BS]]
// CHECK:         %[[F:.*]] = linalg.fill {{.*}} outs(%[[E2]]
// CHECK:         cinm.op.gemv %{{.*}}, %[[SP]] into %[[F]]
func.func @splat_and_fill(%a: tensor<8x8xf32>, %s: f32) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %sp = tensor.splat %s : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %sp : tensor<8xf32>, %o = %f : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y into %o : tensor<8x8xf32>, tensor<8xf32> into tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}

// -----

// A linalg.generic in fill form holds the fill value in its body rather than in
// an operand. The block is isolated from above, so what the body captures has to
// move in as well.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @generic_fill
// CHECK-NOT:   linalg.generic
// CHECK:       %[[E:.*]] = tensor.empty()
// CHECK:       cinm.compute_block (%[[X:.*]] = %{{.*}} : tensor<8x8xf32>, %{{.*}} = %{{.*}} : tensor<8xf32>, %[[E2:.*]] = %[[E]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK:         %[[CST:.*]] = arith.constant 1.500000e+00 : f32
// CHECK:         %[[F:.*]] = linalg.generic {{.*}} outs(%[[E2]] : tensor<8xf32>)
// CHECK:           linalg.yield %[[CST]]
// CHECK:         cinm.op.gemv %[[X]], %{{.*}} into %[[F]]
func.func @generic_fill(%a: tensor<8x8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %cst = arith.constant 1.5 : f32
  %e = tensor.empty() : tensor<8xf32>
  %f = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%e : tensor<8xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  } -> tensor<8xf32>
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>, %o = %f : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y into %o : tensor<8x8xf32>, tensor<8xf32> into tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}

// -----

// A splat shared by two blocks is rematerialized in both: each block
// initializes its own copy on the device instead of receiving a transfer.

// CHECK-LABEL: func @shared_splat
// CHECK-COUNT-2: cinm.compute_block (%{{.*}} = %{{.*}} : f32) -> tensor<8xf32>
// CHECK:           tensor.splat
func.func @shared_splat(%s: f32) -> (tensor<8xf32>, tensor<8xf32>) {
  %sp = tensor.splat %s : tensor<8xf32>
  %r0 = cinm.compute_block (%x = %sp : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.elementwise mul %x, %x: tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %r1 = cinm.compute_block (%x = %sp : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.elementwise add %x, %x: tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  return %r0, %r1 : tensor<8xf32>, tensor<8xf32>
}

// -----

// A result that exists only to be unpacked into a scalar is unpacked inside the
// block, which then returns the scalar.

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

// CHECK-LABEL: func @extract_scalar
// CHECK:       %[[R:.*]] = cinm.compute_block ({{.*}}) -> f32
// CHECK:         %[[S:.*]] = linalg.generic
// CHECK:         %[[E:.*]] = tensor.extract %[[S]][] : tensor<f32>
// CHECK:         cinm.yield %[[E]] : f32
// CHECK:       return %[[R]] : f32
func.func @extract_scalar(%in: tensor<1024xf32>) -> f32 {
  %r = cinm.compute_block (%x = %in : tensor<1024xf32>) -> tensor<f32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<f32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<f32>) -> tensor<f32>
    %s = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%x : tensor<1024xf32>) outs(%f : tensor<f32>) {
    ^bb0(%in2: f32, %out: f32):
      %a = arith.addf %in2, %out : f32
      linalg.yield %a : f32
    } -> tensor<f32>
    cinm.yield %s : tensor<f32>
  }
  %extracted = tensor.extract %r[] : tensor<f32>
  return %extracted : f32
}

// -----

// Only a single-element result is worth unpacking: a wider one still has to
// leave the block as a tensor.

// CHECK-LABEL: func @extract_from_wide_result
// CHECK:       %[[R:.*]] = cinm.compute_block ({{.*}}) -> tensor<8xf32>
// CHECK:       tensor.extract %[[R]]
func.func @extract_from_wide_result(%a: tensor<8x8xf32>, %b: tensor<8xf32>, %i: index) -> f32 {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %extracted = tensor.extract %r[%i] : tensor<8xf32>
  return %extracted : f32
}

// -----

// A producer with side effects is never duplicated.

// CHECK-LABEL: func @side_effecting_producer
// CHECK:       linalg.fill
// CHECK:       %[[T:.*]] = bufferization.to_tensor
// CHECK:       cinm.compute_block (%{{.*}} = %{{.*}} : tensor<8x8xf32>, %{{.*}} = %[[T]] : tensor<8xf32>)
func.func @side_effecting_producer(%a: tensor<8x8xf32>, %m: memref<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%m : memref<8xf32>)
  %t = bufferization.to_tensor %m : memref<8xf32> to tensor<8xf32>
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %t : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}

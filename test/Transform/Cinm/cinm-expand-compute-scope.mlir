// RUN: cinm-opt %s --cinm-expand-compute-scope --split-input-file | FileCheck %s
// RUN: cinm-opt %s --cinm-expand-compute-scope=absorb-results=false --split-input-file | FileCheck %s --check-prefix=OPERANDS-ONLY

// Both the slicing of the input and the insertion of the result move inside the
// block, so the block sees the whole tensors and the loop index.

// CHECK-LABEL: func @gemv_loop
// CHECK:       scf.for %[[I:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %{{.*}})
// CHECK-NOT:     tensor.extract_slice
// CHECK:         %[[R:.*]] = cinm.compute_block (%[[W:.*]] = %{{.*}} : tensor<6x768x768xf32>, %[[BI:.*]] = %[[I]] : index, %[[X:.*]] = %{{.*}} : tensor<768xf32>, %[[D:.*]] = %[[ACC]] : tensor<6x1024x768xf32>) -> tensor<6x1024x768xf32>
// CHECK:           %[[S:.*]] = tensor.extract_slice %[[W]][%[[BI]], 0, 0] [1, 768, 768] [1, 1, 1]
// CHECK:           %[[V:.*]] = cinm.op.gemv %[[S]], %[[X]]
// CHECK:           %[[INS:.*]] = tensor.insert_slice %[[V]] into %[[D]][%[[BI]], 3, 0] [1, 1, 768] [1, 1, 1]
// CHECK:           cinm.yield %[[INS]]
// CHECK:         scf.yield %[[R]]
func.func @gemv_loop(%w: tensor<6x768x768xf32>, %x: tensor<768xf32>, %out: tensor<6x1024x768xf32>) -> tensor<6x1024x768xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %r = scf.for %i = %c0 to %c6 step %c1 iter_args(%acc = %out) -> tensor<6x1024x768xf32> {
    %slice = tensor.extract_slice %w[%i, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %v = cinm.compute_block (%a = %slice : tensor<768x768xf32>, %b = %x : tensor<768xf32>) -> tensor<768xf32> {
      %g = cinm.op.gemv %a, %b : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
      cinm.yield %g : tensor<768xf32>
    }
    %ins = tensor.insert_slice %v into %acc[%i, 3, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    scf.yield %ins : tensor<6x1024x768xf32>
  }
  return %r : tensor<6x1024x768xf32>
}

// -----

// Several results, each with its own insertion. The platform attribute is kept,
// and the constant offset is rematerialized rather than passed in.

// CHECK-LABEL: func @two_results
// CHECK:       %[[R:.*]]:2 = cinm.compute_block on platform #cinm.host_platform (%{{.*}} = %{{.*}} : tensor<8x8xf32>, %[[Y:.*]] = %{{.*}} : tensor<8xf32>, %[[D0:.*]] = %{{.*}} : tensor<64xf32>, %[[D1:.*]] = %{{.*}} : tensor<64xf32>) -> tensor<64xf32>, tensor<64xf32>
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[V:.*]] = cinm.op.gemv
// CHECK:         %[[I0:.*]] = tensor.insert_slice %[[V]] into %[[D0]][%[[C3]]] [8] [1]
// CHECK:         %[[I1:.*]] = tensor.insert_slice %[[Y]] into %[[D1]][0] [8] [1]
// CHECK:         cinm.yield %[[I0]], %[[I1]]
// CHECK:       return %[[R]]#0, %[[R]]#1
func.func @two_results(%a: tensor<8x8xf32>, %b: tensor<8xf32>, %d0: tensor<64xf32>, %d1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %c3 = arith.constant 3 : index
  %r:2 = cinm.compute_block on platform #cinm.host_platform (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32>, tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v, %y : tensor<8xf32>, tensor<8xf32>
  }
  %i0 = tensor.insert_slice %r#0 into %d0[%c3] [8] [1] : tensor<8xf32> into tensor<64xf32>
  %i1 = tensor.insert_slice %r#1 into %d1[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %i0, %i1 : tensor<64xf32>, tensor<64xf32>
}

// -----

// Producers that only shape or initialize data are pulled in transitively: the
// fill and the empty tensor it writes to move in as a whole, and only the
// splatted value crosses the block boundary.

// CHECK-LABEL: func @splat_and_fill
// CHECK-SAME:    (%[[A:.*]]: tensor<8x8xf32>, %[[S:.*]]: f32)
// CHECK-NOT:   tensor.splat
// CHECK-NOT:   linalg.fill
// CHECK:       cinm.compute_block (%{{.*}} = %[[A]] : tensor<8x8xf32>, %[[BS:.*]] = %[[S]] : f32) -> tensor<8xf32>
// CHECK:         %[[SP:.*]] = tensor.splat %[[BS]]
// CHECK:         %[[E:.*]] = tensor.empty()
// CHECK:         %[[F:.*]] = linalg.fill {{.*}} outs(%[[E]]
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
// CHECK:       cinm.compute_block (%[[X:.*]] = %{{.*}} : tensor<8x8xf32>, %{{.*}} = %{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK:         %[[CST:.*]] = arith.constant 1.500000e+00 : f32
// CHECK:         %[[E:.*]] = tensor.empty()
// CHECK:         %[[F:.*]] = linalg.generic {{.*}} outs(%[[E]] : tensor<8xf32>)
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

// A destination defined after the block is fine as long as it can be
// rematerialized inside it.

// CHECK-LABEL: func @dest_defined_later
// CHECK:       cinm.compute_block ({{.*}}) -> tensor<64xf32>
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:         tensor.insert_slice %{{.*}} into %[[E]][0] [8] [1]
func.func @dest_defined_later(%a: tensor<8x8xf32>, %b: tensor<8xf32>) -> tensor<64xf32> {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %dest = tensor.empty() : tensor<64xf32>
  %i = tensor.insert_slice %r into %dest[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %i : tensor<64xf32>
}

// -----

// The slice is shared: it is rematerialized in both blocks, so both take the
// full source tensor.

// CHECK-LABEL: func @shared_slice
// CHECK-COUNT-2: cinm.compute_block (%{{.*}} = %{{.*}} : tensor<?xf32>, %{{.*}} = %{{.*}} : index, %{{.*}} = %{{.*}} : index) -> tensor<?xf32>
// CHECK:           tensor.extract_slice
func.func @shared_slice(%s: tensor<?xf32>, %o: index, %n: index) -> (tensor<?xf32>, tensor<?xf32>) {
  %e = tensor.extract_slice %s[%o] [%n] [1] : tensor<?xf32> to tensor<?xf32>
  %r0 = cinm.compute_block (%x = %e : tensor<?xf32>) -> tensor<?xf32> {
    %v = cinm.op.elementwise mul %x, %x: tensor<?xf32>
    cinm.yield %v : tensor<?xf32>
  }
  %r1 = cinm.compute_block (%x = %e : tensor<?xf32>) -> tensor<?xf32> {
    %v = cinm.op.elementwise add %x, %x: tensor<?xf32>
    cinm.yield %v : tensor<?xf32>
  }
  return %r0, %r1 : tensor<?xf32>, tensor<?xf32>
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

// The result is used elsewhere: moving the insertion in would hide it.

// CHECK-LABEL: func @result_used_twice
// CHECK:       %[[R:.*]] = cinm.compute_block ({{.*}}) -> tensor<8xf32>
// CHECK:       %[[I:.*]] = tensor.insert_slice %[[R]]
// CHECK:       return %[[I]], %[[R]]
func.func @result_used_twice(%a: tensor<8x8xf32>, %b: tensor<8xf32>, %d: tensor<64xf32>) -> (tensor<64xf32>, tensor<8xf32>) {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %i = tensor.insert_slice %r into %d[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %i, %r : tensor<64xf32>, tensor<8xf32>
}

// -----

// The insertion is conditional, it must stay where it is.

// CHECK-LABEL: func @conditional_insert
// CHECK:       %[[R:.*]] = cinm.compute_block ({{.*}}) -> tensor<8xf32>
// CHECK:       scf.if
// CHECK:         tensor.insert_slice %[[R]]
func.func @conditional_insert(%a: tensor<8x8xf32>, %b: tensor<8xf32>, %d: tensor<64xf32>, %p: i1) -> tensor<64xf32> {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %i = scf.if %p -> tensor<64xf32> {
    %t = tensor.insert_slice %r into %d[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
    scf.yield %t : tensor<64xf32>
  } else {
    scf.yield %d : tensor<64xf32>
  }
  return %i : tensor<64xf32>
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

// -----

// Each direction can be disabled on its own.

// OPERANDS-ONLY-LABEL: func @one_direction
// OPERANDS-ONLY:       cinm.compute_block (%{{.*}} = %{{.*}} : tensor<64xf32>) -> tensor<8xf32>
// OPERANDS-ONLY:         tensor.extract_slice
// OPERANDS-ONLY:       tensor.insert_slice
func.func @one_direction(%s: tensor<64xf32>, %d: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.extract_slice %s[0] [8] [1] : tensor<64xf32> to tensor<8xf32>
  %r = cinm.compute_block (%x = %e : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.elementwise mul %x, %x: tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %i = tensor.insert_slice %r into %d[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %i : tensor<64xf32>
}

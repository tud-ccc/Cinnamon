// RUN: cinm-opt %s --split-input-file -verify-diagnostics

func.func @out_shape_mismatch(%a: memref<8x1024xi32>, %o: memref<1024xi32>) {
  // expected-error @below {{output buffer shape 1024 does not match the shape obtained by reducing dimension 1 of the input}}
  cinm.op.reduce add (%a) into %o : memref<8x1024xi32> into memref<1024xi32>
  return
}

// -----

func.func @out_elt_mismatch(%a: memref<8x1024xi32>, %o: memref<8xf32>) {
  // expected-error @below {{output buffer element type 'f32' does not match input element type 'i32'}}
  cinm.op.reduce add (%a) into %o : memref<8x1024xi32> into memref<8xf32>
  return
}

// -----

func.func @out_must_be_memref(%a: tensor<8x1024xi32>, %o: tensor<8xi32>) {
  // expected-error @below {{`into` output buffer is only supported in memref mode}}
  cinm.op.reduce add (%a) into %o : tensor<8x1024xi32> into tensor<8xi32>
  return
}

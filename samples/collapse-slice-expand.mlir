// Variant of reshape-slice-reshape using collapse_shape + expand_shape.
// collapse_shape and expand_shape bufferize as views (no copy), so the chain
// may fold more cleanly than tensor.reshape.
//
// Try:
//   mlir-opt %s --one-shot-bufferize="bufferize-function-boundaries=true" \
//     --canonicalize -o -

func.func @collapse_slice_expand(%arg0: tensor<2x8xi32>, %i: index) -> tensor<2x2xi32> {
  // Flatten: 2x8 -> 16
  %flat  = tensor.collapse_shape %arg0 [[0, 1]]
             : tensor<2x8xi32> into tensor<16xi32>

  // Extract a 4-element window at dynamic offset %i
  %slice = tensor.extract_slice %flat[%i][4][1]
             : tensor<16xi32> to tensor<4xi32>

  // Re-structure the window into 2x2
  %chunk = tensor.expand_shape %slice [[0, 1]] output_shape [2, 2]
             : tensor<4xi32> into tensor<2x2xi32>

  return %chunk : tensor<2x2xi32>
}

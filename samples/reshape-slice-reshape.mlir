// Minimal reproduction of the reshape -> extract_slice -> reshape chain that
// appears in the VA lowering:
//
//   %flat   = tensor.reshape %arg0(%shape_1d)        // flatten  2x8  ->  16
//   %slice  = tensor.extract_slice %flat[%i][4][1]   // tile window   ->   4
//   %chunk  = tensor.reshape %slice(%shape_2d)       // structure      ->  2x2
//
// Goal: bufferization should NOT materialise a full 16-element intermediate
// for the flatten reshape.  Ideally the whole chain collapses to a single
// memref.subview of the original buffer.
//
// Try:
//   mlir-opt %s --one-shot-bufferize="bufferize-function-boundaries=true" \
//     --canonicalize -o -

func.func @reshape_slice_reshape(%arg0: tensor<2x8xi32>, %i: index) -> tensor<2x2xi32> {
  %shape_1d = arith.constant dense<16>     : tensor<1xi64>
  %shape_2d = arith.constant dense<[2, 2]> : tensor<2xi64>

  // Flatten: 2x8 -> 16
  %flat  = tensor.reshape %arg0(%shape_1d)
             : (tensor<2x8xi32>, tensor<1xi64>) -> tensor<16xi32>

  // Extract a 4-element window at dynamic offset %i
  %slice = tensor.extract_slice %flat[%i][4][1]
             : tensor<16xi32> to tensor<4xi32>

  // Re-structure the window into 2x2
  %chunk = tensor.reshape %slice(%shape_2d)
             : (tensor<4xi32>, tensor<2xi64>) -> tensor<2x2xi32>

  return %chunk : tensor<2x2xi32>
}

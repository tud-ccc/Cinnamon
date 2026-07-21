// Minimal reproduction of the pattern:
//
//   tensor.empty() : tensor<16384x4xi32>
//   <op that writes into the empty as out buffer>   -> tensor<16384x4xi32>
//   tensor.collapse_shape [[0, 1]]                  -> tensor<65536xi32>
//   tensor.insert_slice into %acc[%i]               -> tensor<16777216xi32>
//
// Ideal bufferization: the fill writes directly into a contiguous slice of the
// accumulator — no intermediate 16384x4 allocation materialises.
//
// Try:
//   mlir-opt %s --one-shot-bufferize="bufferize-function-boundaries=true" \
//     --canonicalize -o -
//
// --- Research notes (not yet implemented) ------------------------------------
//
// WITHOUT collapse_shape, --eliminate-empty-tensors already handles this. It
// walks SubsetInsertionOpInterface (tensor.insert_slice) backward through the
// use-def chain to find tensor.empty ops and replaces them with
// tensor.extract_slice of the destination, enabling inplace bufferization.
//
// The relevant code lives in:
//   mlir/lib/Dialect/Bufferization/Transforms/EmptyTensorElimination.cpp
//     eliminateEmptyTensors()  (line ~110)
//
// The blocker is TraversalConfig::followSameTypeOrCastsOnly = true (line ~136),
// which stops the backward walk when the type changes — exactly what
// collapse_shape does. The upstream code even has a TODO for this case:
//
//   // TODO: This could be extended to support IR such as:
//   // %0 = tensor.empty() : tensor<128xf32>
//   // %1 = "some_op"(%0) : tensor<128xf32>
//   // %2 = tensor.expand_shape %1 ...
//   // %3 = tensor.insert_slice %2 into ...
//
// Proposed fix:
//   1. Set followSameTypeOrCastsOnly = false so the traversal crosses
//      collapse_shape.
//   2. When building the replacement for tensor.empty and the path went through
//      a collapse_shape with reassociation R, generate:
//        tensor.expand_shape (tensor.extract_slice dest[offsets][sizes][strides])
//                            R output_shape <original empty shape>
//      instead of a plain tensor.extract_slice.
//   3. After substitution, collapse_shape(expand_shape(x, R), R) folds to x via
//      the existing ComposeReassociativeReshapeOps canonicalization, making the
//      insert_slice a no-op (inplace).
//
// The path-through-collapse tracking would need to be threaded through
// findValueInReverseUseDefChain or done as a post-processing step that
// inspects visitedOpOperands for collapse_shape ops.
//
// Alternatively, a standalone OpRewritePattern<tensor::InsertSliceOp> could
// match the full chain and perform the substitution without touching the
// generic traversal infrastructure.
// -----------------------------------------------------------------------------

func.func @empty_fill_collapse_insert(
    %val: i32,
    %acc: tensor<16777216xi32>,
    %i: index) -> tensor<16777216xi32> {

  // Step 1: allocate a shaped scratch buffer and fill it.
  %empty   = tensor.empty() : tensor<16384x4xi32>
  %filled  = linalg.fill ins(%val : i32)
                         outs(%empty : tensor<16384x4xi32>)
             -> tensor<16384x4xi32>

  // Step 2: flatten the result.
  %collapsed = tensor.collapse_shape %filled [[0, 1]]
                 : tensor<16384x4xi32> into tensor<65536xi32>

  // Step 3: scatter the flat result into the accumulator.
  %result = tensor.insert_slice %collapsed into %acc[%i] [65536] [1]
              : tensor<65536xi32> into tensor<16777216xi32>

  return %result : tensor<16777216xi32>
}

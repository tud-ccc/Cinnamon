//===- CnmScatterMap.h - Interpreting cnm.scatter/gather maps ------------===//
//
// A scatter/gather map sends the elements of every leaf's `cnm.buffer` to
// elements of a host value. Its domain is the workgroup shape followed by the
// first `p` of the buffer's own dimensions, for any `p`:
//
//   (w_0..w_{n-1}, i_0..i_{p-1}) -> (h_0..h_{k-p'-1})     p' = bufferRank - p
//
// The `p'` buffer dimensions left out of the domain are transferred as a
// block, and the same number of host dimensions are left out of the results:
// they are the block's own shape, and must match extent for extent.
//
//   buf[w][i_0..i_{p-1}, rest] = host[map(w, i_0..i_{p-1}) ++ rest]
//
// `p = bufferRank` is the pointwise form, which names a host element for every
// buffer element and leaves nothing implicit. `p = 0` is one whole-buffer
// block per leaf. Whether a block is *contiguous* is a question about the host
// value's layout, answered where layouts exist rather than here.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include <mlir/IR/AffineMap.h>

#include <optional>

namespace mlir::cnm {

/// How many of `buffer`'s own dimensions `map` retains in its domain.
int64_t getNumRetainedBufferDims(AffineMap map, BufferType buffer);

/// Whether `map` names a host element for each buffer element individually.
bool isPointwiseScatterMap(AffineMap map, BufferType buffer);

/// Shape of the block a leaf receives at one point of the retained index
/// space: the buffer dimensions `map` leaves implicit. Empty for a pointwise
/// map.
ArrayRef<int64_t> getScatterBlockShape(AffineMap map, BufferType buffer);

/// Number of such blocks one leaf receives.
int64_t getScatterBlocksPerLeaf(AffineMap map, BufferType buffer);

/// Number of host dimensions `map` leaves implicit, i.e. how many results a
/// map over `buffer` has fewer than the host rank.
int64_t getNumImplicitHostDims(AffineMap map, BufferType buffer);

/// `map` with its implicit dimensions written out, so that it names one host
/// index per host dimension over the whole workgroup x buffer index space.
/// The identity on a map that is already pointwise.
AffineMap inflateScatterMapToPointwise(AffineMap map, BufferType buffer);

/// Extents of the inflated map's domain: the workgroup shape followed by the
/// buffer shape.
SmallVector<int64_t> getScatterIndexSpace(BufferType buffer);

//===----------------------------------------------------------------------===//
// Analysis of the linearized map
//
// Two obligations the old shape-suffix contract discharged for free -- that a
// transfer stays inside the host value, and that no two leaves gather to the
// same place -- become questions about one affine expression over a box.
//===----------------------------------------------------------------------===//

/// `map` composed with the row-major linearization of `hostShape`, giving one
/// expression for the element offset each buffer element reads or writes.
/// Fails if the result count does not match `hostShape`.
FailureOr<AffineExpr> linearizeScatterMap(AffineMap map,
                                          ArrayRef<int64_t> hostShape);

/// The exact largest value `expr` takes over the box `[0, extents)`, or
/// nullopt when that cannot be computed. Floordiv and mod by a constant are
/// handled, but no dimension may appear twice: interval arithmetic treats
/// occurrences as independent, and a bound that is merely an
/// over-approximation is no grounds for rejecting anything.
std::optional<int64_t> getAffineUpperBound(AffineExpr expr,
                                           ArrayRef<int64_t> extents);

/// Whether `expr` takes a different value at every point of the box
/// `[0, extents)`, or nullopt when that cannot be decided. Both answers are
/// conclusive; the undecided case is common.
std::optional<bool> isAffineExprInjective(AffineExpr expr,
                                          ArrayRef<int64_t> extents);

} // namespace mlir::cnm

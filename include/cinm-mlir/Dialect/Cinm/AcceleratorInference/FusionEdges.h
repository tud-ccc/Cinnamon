#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <optional>
#include <string>

namespace mlir::cinm {

/// What one distributed linalg op contributes to the search space, as the
/// fusion analysis needs to see it. A plugin fills this in while declaring the
/// op's own parameters; nothing here is target-specific.
struct DistributedOpInfo {
  linalg::LinalgOp op;
  /// Prefix its parameters are named after, e.g. `gemv`.
  std::string name;
  /// Loop extents, one per iteration dimension.
  SmallVector<int64_t> extents;
  /// Tile size per (level, iteration dimension), outermost level first.
  /// `tiles.front()` is the workgroup distribution, `tiles.back()` the leaf
  /// level.
  SmallVector<SmallVector<IntVar>> tiles;
  /// The op's tile-to-workgroup order parameter, absent when it has fewer than
  /// two iteration dimensions and so only one order.
  std::optional<PermVar> order;
};

/// Declare one `fuse.<producer>-><consumer>` variable per producer/consumer
/// edge between the given ops, and the constraints that make a positive value
/// of it mean what it says: at `fuse >= 1` the consumer finds its operand on
/// the leaf it is about to read it from, so the launches can be merged, and at
/// `fuse >= l + 1` the two ops additionally agree at memory level `l`, so their
/// loop nests at that level can be fused.
///
/// The variable is *not* stamped on any op and no pass reads it. The fusion
/// passes are opportunistic peepholes that fire iff the lowered IR matches, so
/// the variable's only jobs are to keep the two ops' tile sizes coupled and to
/// make the fusable corner of the joint space dense enough for a search to land
/// in. A configuration is never wrong because of what `fuse` says, only less
/// interesting than it claimed.
void declareFusionEdges(llvm::ArrayRef<DistributedOpInfo> ops, SpaceBuilder &b);

} // namespace mlir::cinm

//===- AbsorbStaticSlices.cpp - Slice static tensors inside compute ops ---===//
//
// A compute op inside a layer loop reads its weight as `weights[layer]`, a
// slice of the static tensor at the loop index. Taken outside the region,
// that slice would become the isolated block's operand and the block would
// know neither the tensor nor the index. Cloned into the region, the block
// takes the whole tensor and the index instead -- the signature under which
// all the slices can be kept resident and the launch selects its slot.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMABSORBSTATICSLICESPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// The slicing op `value` is the result of, when it is a static slice whose
/// index and source are both defined outside `region` -- the case where
/// cloning it inside changes what the region captures from (slice) to
/// (source, index).
Operation *absorbableSlice(Value value, Region &region) {
  std::optional<StaticSlice> slice = resolveStaticSlice(value);
  if (!slice)
    return nullptr;
  Operation *def = value.getDefiningOp();
  // Only the slicing op itself is moved, not a chain of views over it: the
  // resolver looks through casts and reshapes, this pass does not.
  if (!isa_and_nonnull<OffsetSizeAndStrideOpInterface>(def) ||
      def->getOperand(0) != slice->source)
    return nullptr;
  if (region.isAncestor(slice->source.getParentRegion()) ||
      region.isAncestor(slice->index.getParentRegion()))
    return nullptr;
  return def;
}

void absorbInto(ComputeOp compute, RewriterBase &rewriter) {
  Region &region = compute.getBody();
  SetVector<Value> captured;
  getUsedValuesDefinedAbove(region, captured);
  for (Value value : captured) {
    Operation *slice = absorbableSlice(value, region);
    if (!slice)
      continue;
    rewriter.setInsertionPointToStart(&region.front());
    Operation *inside = rewriter.clone(*slice);
    rewriter.replaceUsesWithIf(
        value, inside->getResult(0), [&](OpOperand &use) {
          return region.isAncestor(use.getOwner()->getParentRegion());
        });
  }
}

struct AbsorbStaticSlicesPass
    : public impl::CinmAbsorbStaticSlicesPassBase<AbsorbStaticSlicesPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<ComputeOp> computes;
    getOperation()->walk([&](ComputeOp op) { computes.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (ComputeOp compute : computes)
      absorbInto(compute, rewriter);
  }
};

} // namespace
} // namespace mlir::cinm

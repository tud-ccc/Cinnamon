//===- AbsorbStaticSlices.cpp - Slice static tensors inside compute ops ---===//
//
// A compute op inside a layer loop reads its weight as `weights[layer]`, a
// slice of the static tensor at the loop index. Taken outside the region,
// that slice would become the isolated block's operand and the block would
// know neither the tensor nor the index. Cloned into the region, the block
// takes the whole tensor and the index instead -- the signature under which
// all the slices can be kept resident and the launch selects its slot.
//
// Takes the isolated form: a block that takes such a slice as an operand is
// turned back into a cinm.compute for the time it takes to clone the slice
// in, then isolated again, with the slice's source and index as operands in
// its place. The other blocks are left alone.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
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

/// Of two reuse distances of the same operand, the one further from the
/// cache: -1 (not touched before) is the furthest.
int64_t colder(int64_t a, int64_t b) {
  return a < 0 || b < 0 ? -1 : std::max(a, b);
}

LogicalResult absorbInto(ComputeBlockOp block, RewriterBase &rewriter) {
  if (llvm::none_of(block.getOperands(), [&](Value operand) {
        return absorbableSlice(operand, block.getBody());
      }))
    return success();
  // The reuse distances (--cinm-annotate-operand-reuse), by value: a
  // slice's own operands take over its distance -- the block still reads
  // only the slice.
  auto reuse =
      block->getAttrOfType<DenseI64ArrayAttr>(CinmDialect::OPERAND_REUSE_NAME);
  DenseMap<Value, int64_t> reuseOf;
  if (reuse)
    for (auto [operand, entry] :
         llvm::zip(block.getOperands(), reuse.asArrayRef()))
      reuseOf[operand] = entry;

  ComputeOp compute = deisolateComputeBlock(block, rewriter);
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
    if (reuse)
      for (Value sliceOperand : slice->getOperands()) {
        auto [it, inserted] =
            reuseOf.try_emplace(sliceOperand, reuseOf.lookup(value));
        if (!inserted)
          it->second = colder(it->second, reuseOf.lookup(value));
      }
  }
  ComputeBlockOp isolated = isolateComputeBlock(compute, rewriter);
  if (!reuse)
    return success();
  SmallVector<int64_t> entries;
  for (Value operand : isolated.getOperands()) {
    auto it = reuseOf.find(operand);
    if (it == reuseOf.end())
      return isolated.emitError("absorbing static slices gave the block an "
                                "operand without a reuse distance");
    entries.push_back(it->second);
  }
  isolated->setAttr(CinmDialect::OPERAND_REUSE_NAME,
                    rewriter.getDenseI64ArrayAttr(entries));
  return success();
}

struct AbsorbStaticSlicesPass
    : public impl::CinmAbsorbStaticSlicesPassBase<AbsorbStaticSlicesPass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(requireIsolatedComputeOps(getOperation(), getArgument())))
      return signalPassFailure();
    SmallVector<ComputeBlockOp> blocks;
    getOperation()->walk([&](ComputeBlockOp op) { blocks.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (ComputeBlockOp block : blocks)
      if (failed(absorbInto(block, rewriter)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cinm

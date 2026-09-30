//===- MergeHostBlocks.cpp - Fuse host blocks the device does not split ---===//
//
// --cinm-complete-compute-graph cuts the host code at every view op, because
// for the graph solver slicing and reshaping are edges, not nodes. Handed to
// a host compiler, those cuts only cost: every host block becomes a separate
// function and a separate call, with nothing on the device in between. This
// pass fuses each run of host blocks that only views separate into one
// block, so the host compiler sees the code between two device blocks whole.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMMERGEHOSTBLOCKSPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// The ops --cinm-complete-compute-graph leaves between blocks as edges.
bool isView(Operation *op) {
  return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
             tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ReshapeOp,
             tensor::CastOp, bufferization::MaterializeInDestinationOp>(op);
}

/// Whether `op` writes into a block argument: a function argument or a
/// loop-carried value, i.e. state that outlives the call or the iteration
/// (the KV cache). Inside a host block, the host compiler would see it as a
/// value and copy the whole of it.
bool writesState(Operation *op) {
  Value dest;
  if (auto insert = dyn_cast<tensor::InsertSliceOp>(op))
    dest = insert.getDest();
  else if (auto pin = dyn_cast<bufferization::MaterializeInDestinationOp>(op))
    dest = pin.getDest();
  else
    return false;
  // Inserting into an insert writes into the same buffer; follow it.
  while (auto inner = dest.getDefiningOp<tensor::InsertSliceOp>())
    dest = inner.getDest();
  return isa<BlockArgument>(dest);
}

/// Free ops that stay outside, above the merged block: they have no
/// operands, so nothing in the run can feed them.
bool staysOutside(Operation *op) {
  return op->getNumOperands() == 0 &&
         (op->hasTrait<OpTrait::ConstantLike>() || isa<tensor::EmptyOp>(op));
}

struct MergeHostBlocksPass
    : public impl::CinmMergeHostBlocksPassBase<MergeHostBlocksPass> {
  using Base::Base;

  void runOnOperation() override {
    // The blocks the program's control flow runs through; the bodies of
    // compute ops are not among them.
    SmallVector<Block *> blocks;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<ComputeOpInterface>(op))
        return WalkResult::skip();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          blocks.push_back(&block);
      return WalkResult::advance();
    });

    IRRewriter rewriter(&getContext());
    for (Block *block : blocks)
      mergeRuns(*block, rewriter);
  }

private:
  void mergeRuns(Block &block, RewriterBase &rewriter) {
    SmallVector<Operation *> ops =
        llvm::map_to_vector(block, [](Operation &op) { return &op; });

    SmallVector<Operation *> run;     // host blocks and the views between
    SmallVector<Operation *> pending; // views since the last host block
    unsigned hosts = 0;
    auto flush = [&]() {
      if (hosts >= 2)
        merge(run, rewriter);
      run.clear();
      pending.clear();
      hosts = 0;
    };

    for (Operation *op : ops) {
      if (isHostComputeOp(op)) {
        run.append(pending);
        pending.clear();
        run.push_back(op);
        ++hosts;
        continue;
      }
      if (staysOutside(op))
        continue;
      if (isView(op) && !writesState(op)) {
        // Only worth holding once a run has started; a view before the
        // first host block is simply not part of it.
        if (hosts > 0)
          pending.push_back(op);
        continue;
      }
      flush();
    }
    flush();
  }

  /// Replaces the ops of `run` (in block order, starting and ending with a
  /// host block) with one host block placed where the last of them was.
  void merge(ArrayRef<Operation *> run, RewriterBase &rewriter) {
    SmallPtrSet<Operation *, 16> inRun(run.begin(), run.end());
    Block *block = run.front()->getBlock();

    SmallVector<Value> escaping;
    for (Operation *op : run)
      for (Value result : op->getResults())
        if (llvm::any_of(result.getUses(), [&](OpOperand &use) {
              Operation *user = block->findAncestorOpInBlock(*use.getOwner());
              return !user || !inRun.contains(user);
            }))
          escaping.push_back(result);

    rewriter.setInsertionPointAfter(run.back());
    auto merged = ComputeOp::create(rewriter, run.front()->getLoc(),
                                    ValueRange(escaping).getTypes());
    merged->setAttr(
        CinmDialect::AVAILABLE_PLATFORMS_NAME,
        run.front()->getAttr(CinmDialect::AVAILABLE_PLATFORMS_NAME));

    Block &body = merged.getBody().front();
    for (Operation *op : run)
      rewriter.moveOpBefore(op, &body, body.end());
    rewriter.setInsertionPointToEnd(&body);
    YieldOp::create(rewriter, merged.getLoc(), escaping);
    for (auto [value, result] : llvm::zip(escaping, merged->getResults()))
      rewriter.replaceUsesWithIf(value, result, [&](OpOperand &use) {
        return !merged->isProperAncestor(use.getOwner());
      });

    // The old host blocks dissolve into the new one.
    for (Operation *op : run) {
      if (auto compute = dyn_cast<ComputeOp>(op))
        unwrapComputeBlockOp(compute, rewriter);
      else if (auto isolated = dyn_cast<ComputeBlockOp>(op))
        unwrapComputeBlockOp(isolated, rewriter);
    }
  }
};

} // namespace
} // namespace mlir::cinm

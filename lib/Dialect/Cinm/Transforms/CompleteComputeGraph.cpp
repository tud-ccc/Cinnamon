//===- CompleteComputeGraph.cpp - Wrap host code in compute blocks --------===//
//
// Wraps the operations of a function that are not part of any cinm.compute
// region into compute blocks pinned to the host platform, so that the whole
// function body becomes a single connected dataflow graph of compute blocks.
// The graph scheduling algorithm can then see the host portions of the
// application as nodes with their own cost, instead of invisible gaps
// between the offloadable blocks.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMCOMPLETECOMPUTEGRAPHPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// Whether a compute op is nested under `root` (`root` itself not counted).
bool containsComputeOp(Operation *root) {
  auto result = root->walk([&](ComputeOpInterface op) {
    return op.getOperation() == root ? WalkResult::advance()
                                     : WalkResult::interrupt();
  });
  return result.wasInterrupted();
}

/// The function `call` targets, if it can be resolved statically.
Operation *resolveCallee(CallOpInterface call) {
  auto callable = llvm::dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  if (!callable)
    return nullptr;
  return SymbolTable::lookupNearestSymbolFrom(call, callable);
}

/// Dissolves every compute op nested under `root` back into plain host code,
/// so that `root` can be wrapped whole into a host block.
void demoteNestedComputeOps(Operation *root, RewriterBase &rewriter) {
  // Unwrapping clones the body ops, which invalidates any other compute op
  // collected under it, so ops are found and dissolved one at a time.
  while (true) {
    Operation *nested = nullptr;
    root->walk([&](Operation *op) {
      if (op != root && isa<ComputeOp, ComputeBlockOp>(op)) {
        nested = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!nested)
      return;
    nested->emitRemark(
        "demoting compute op to host code, so that the enclosing control "
        "flow op can be wrapped into a host compute block");
    if (auto compute = llvm::dyn_cast<ComputeOp>(nested))
      unwrapComputeBlockOp(compute, rewriter);
    else
      unwrapComputeBlockOp(llvm::cast<ComputeBlockOp>(nested), rewriter);
  }
}

struct CompleteComputeGraphPass
    : public impl::CinmCompleteComputeGraphPassBase<CompleteComputeGraphPass> {
  using Base::Base;

  void runOnOperation() override {
    // Collected up front: wrapping restructures the IR under the functions.
    SmallVector<func::FuncOp> funcs;
    getOperation()->walk([&](func::FuncOp func) { funcs.push_back(func); });

    IRRewriter rewriter(&getContext());
    for (func::FuncOp func : funcs) {
      // Only complete a graph that exists: a function without compute ops
      // has nothing to connect.
      if (func.isExternal() || !containsComputeOp(func))
        continue;
      for (Block &block : func.getBody())
        completeBlock(block, rewriter);
    }
  }

private:
  /// Wraps every maximal run of consecutive non-compute ops of `block` into
  /// one host compute block.
  void completeBlock(Block &block, RewriterBase &rewriter) {
    SmallVector<Operation *> ops = llvm::map_to_vector(
        block.without_terminator(), [](Operation &op) { return &op; });

    SmallVector<Operation *> group;
    SmallPtrSet<Operation *, 8> groupSet;
    auto flush = [&]() {
      wrapGroup(group, rewriter);
      groupSet.clear();
    };
    // Whether `op` consumes a value the current group produces. Such an op,
    // when left outside, must come after the group's block: flushing first
    // places the block before it. Skipped ops have no regions, so operands
    // are all there is to check.
    auto dependsOnGroup = [&](Operation *op) {
      return llvm::any_of(op->getOperands(), [&](Value v) {
        Operation *def = v.getDefiningOp();
        return def && groupSet.contains(def);
      });
    };
    for (Operation *op : ops) {
      // An existing compute op is already a node of the graph.
      if (isa<ComputeOpInterface>(op)) {
        flush();
        continue;
      }
      // Constants stay outside: they are free, and they are typically used
      // from many blocks, so capturing one would only force its value
      // through a yield. --cinm-isolate-compute-blocks clones them into the
      // blocks that need them. tensor.empty stays outside for the same
      // reason: yielding an undefined tensor out of an isolated block leaves
      // bufferization no sane place for the allocation, while an empty
      // passed *into* a block is a plain operand that
      // --eliminate-empty-tensors knows how to fold into a destination.
      if (op->hasTrait<OpTrait::ConstantLike>() || isa<tensor::EmptyOp>(op))
        continue;
      // View and routing ops stay outside as well: slicing, reshaping and
      // destination-pinning are how values travel between nodes, not
      // computation. The graph collection sees through them -- they unite
      // the blocks they touch into one component and dependency edges trace
      // through them -- so they are the edges of the compute graph, and a
      // node holding only edges would distort the schedule. (A single-element
      // tensor.extract never reaches this pass: --cinm-expand-compute-scope
      // absorbs it into the producing block, which yields the scalar.)
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
              tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ReshapeOp,
              tensor::CastOp, bufferization::MaterializeInDestinationOp>(op)) {
        if (dependsOnGroup(op))
          flush();
        continue;
      }
      // An op with compute ops nested inside (a loop around an offloaded
      // block) cannot become part of a host block without hiding those
      // nodes. Either demote them to host code, or leave the op in place as
      // a hole in the graph.
      if (op->getNumRegions() != 0 && containsComputeOp(op)) {
        if (demoteNestedCompute) {
          demoteNestedComputeOps(op, rewriter);
        } else {
          op->emitWarning(
              "op contains compute ops and cannot be wrapped into a host "
              "compute block; the compute graph is disconnected here "
              "(demote-nested-compute would dissolve them into host code)");
          flush();
          continue;
        }
      }
      // Same for a call: the callee's compute ops would be hidden. Unlike a
      // control flow op the callee cannot be demoted, other callers share it.
      if (auto call = llvm::dyn_cast<CallOpInterface>(op)) {
        Operation *callee = resolveCallee(call);
        if (callee && containsComputeOp(callee)) {
          op->emitWarning(
              "callee contains compute ops, so the call cannot be wrapped "
              "into a host compute block; the compute graph is disconnected "
              "here (inline the callee first)");
          flush();
          continue;
        }
      }
      group.push_back(op);
      groupSet.insert(op);
    }
    flush();
  }

  /// Moves the ops of `group` into a new host compute block inserted in their
  /// place, and clears `group`. The values the group ops define that are used
  /// from outside the group become the results of the block.
  void wrapGroup(SmallVectorImpl<Operation *> &group, RewriterBase &rewriter) {
    if (group.empty())
      return;
    SmallPtrSet<Operation *, 8> inGroup(group.begin(), group.end());
    Block *block = group.front()->getBlock();

    SmallVector<Value> escaping;
    for (Operation *op : group)
      for (Value result : op->getResults())
        if (llvm::any_of(result.getUses(), [&](OpOperand &use) {
              Operation *user = block->findAncestorOpInBlock(*use.getOwner());
              return !user || !inGroup.contains(user);
            }))
          escaping.push_back(result);

    // The ops between the first and the last op of the group that are not
    // part of it (constants, at most) stay behind, above the new block.
    rewriter.setInsertionPointAfter(group.back());
    auto compute = ComputeOp::create(rewriter, group.front()->getLoc(),
                                     ValueRange(escaping).getTypes());
    auto host = HostPlatformAttr::get(rewriter.getContext());
    compute.setPlatformAttr(host);
    // Shadows the cinm.available_platforms attribute of the enclosing
    // function, which advertises the offloading platforms.
    compute->setAttr(CinmDialect::AVAILABLE_PLATFORMS_NAME,
                     rewriter.getArrayAttr({host}));

    Block &body = compute.getBody().front();
    for (Operation *op : group)
      rewriter.moveOpBefore(op, &body, body.end());
    rewriter.setInsertionPointToEnd(&body);
    YieldOp::create(rewriter, compute.getLoc(), escaping);

    for (auto [value, result] : llvm::zip(escaping, compute->getResults()))
      rewriter.replaceUsesWithIf(value, result, [&](OpOperand &use) {
        return !compute->isProperAncestor(use.getOwner());
      });
    group.clear();
  }
};

} // namespace
} // namespace mlir::cinm

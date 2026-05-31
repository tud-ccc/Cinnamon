//===- HoistWriteWeights.cpp - Reduce redundant crossbar uploads ----------===//
//
// Hoist alpine.write_weights (and the associated allocation / cast /
// quantize chain) out of inner column loops so that each GEMV row tile gets
// written to the device exactly once per reduction chunk.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Alpine/Transforms/Passes.h"

#include "cinm-mlir/Dialect/Alpine/IR/AlpineOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_ALPINEHOISTWRITEWEIGHTSPASS
#include "cinm-mlir/Dialect/Alpine/Transforms/Passes.h.inc"

namespace mlir::alpine {
namespace {

static bool isMovableAlloc(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp>(op);
}

/// Collect the simple alloc/cast chain that produces `memref`. The resulting
/// operations are returned in dominance order (alloc before cast).
static void collectDefChain(Value memref,
                            llvm::SmallVectorImpl<Operation *> &chain) {
  llvm::SmallVector<Operation *, 4> reverse;
  Value cur = memref;
  while (Operation *def = cur.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      reverse.push_back(cast);
      cur = cast.getSource();
      continue;
    }
    if (isMovableAlloc(def)) {
      reverse.push_back(def);
    }
    break;
  }
  if (reverse.empty())
    return;
  chain.assign(reverse.rbegin(), reverse.rend());
}

/// Find the quantize op (Alpine or Cinm bufferization variant) that populates
/// `weights`. Returns null if no such producer exists.
static Operation *findQuantizeProducer(Value weights) {
  Operation *found = nullptr;
  for (Operation *user : weights.getUsers()) {
    if (auto quant = dyn_cast<alpine::QuantizeOp>(user)) {
      if (quant.getOut() == weights) {
        if (!found || quant->isBeforeInBlock(found))
          found = quant;
      }
      continue;
    }
    if (auto quant = dyn_cast<cinm::QuantizeMemRefOp>(user)) {
      if (quant.getOut() == weights) {
        if (!found || quant->isBeforeInBlock(found))
          found = quant;
      }
    }
  }
  return found;
}

/// Check whether all operands of the ops we plan to hoist are defined outside
/// `loop`, discounting operands produced by other hoisted ops.
static bool operandsLoopInvariant(
    scf::ForOp loop, llvm::ArrayRef<Operation *> opsToMove,
    const llvm::DenseSet<Operation *> &movable) {
  for (Operation *op : opsToMove) {
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        if (movable.contains(def))
          continue;
      }
      if (loop.isDefinedOutsideOfLoop(operand))
        continue;
      return false;
    }
  }
  return true;
}

static bool tryHoist(alpine::WriteWeightsOp writeOp) {
  Value weights = writeOp.getW();

  llvm::SmallVector<Operation *, 4> opsToMove;
  collectDefChain(weights, opsToMove);

  if (Operation *quant = findQuantizeProducer(weights))
    opsToMove.push_back(quant);
  opsToMove.push_back(writeOp);

  if (opsToMove.size() <= 1)
    return false; // nothing to hoist

  llvm::DenseSet<Operation *> movable(opsToMove.begin(), opsToMove.end());

  llvm::SmallVector<scf::ForOp, 4> loops;
  for (Operation *parent = writeOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(parent))
      loops.push_back(loop);
  }

  Operation *insertBefore = nullptr;
  for (scf::ForOp loop : loops) {
    if (!operandsLoopInvariant(loop, opsToMove, movable))
      break;
    insertBefore = loop.getOperation();
  }

  if (!insertBefore)
    return false;

  bool changed = false;
  for (Operation *op : opsToMove) {
    if (op->getBlock() == insertBefore->getBlock() &&
        op->isBeforeInBlock(insertBefore))
      continue;
    op->moveBefore(insertBefore);
    changed = true;
  }
  return changed;
}

struct AlpineHoistWriteWeightsPass
    : public ::impl::AlpineHoistWriteWeightsPassBase<
          AlpineHoistWriteWeightsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](alpine::WriteWeightsOp op) { (void)tryHoist(op); });
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createAlpineHoistWriteWeightsPass() {
  return std::make_unique<AlpineHoistWriteWeightsPass>();
}

} // namespace mlir::alpine

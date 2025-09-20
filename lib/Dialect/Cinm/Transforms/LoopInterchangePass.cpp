#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::cinm {


#define GEN_PASS_DEF_CINMLOOPINTERCHANGEPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"


namespace {
static bool isHoistablePrologueOp(Operation *op) {
  return isa<arith::ConstantOp, tensor::EmptyOp>(op);
}

static void hoistLeadingPrologueOps(affine::AffineForOp forOp) {
  Block *body = forOp.getBody();
  if (body->empty())
    return;

  for (Operation *op = &body->front(); op && !isa<affine::AffineForOp>(op);
       op = op->getNextNode()) {
    if (!isHoistablePrologueOp(op))
      break;
    op->moveBefore(forOp.getOperation());
  }
}

static void
collectHeadNestedLoopsRelaxed(affine::AffineForOp root, unsigned maxDepth,
                              SmallVectorImpl<affine::AffineForOp> &out) {
  out.clear();
  affine::AffineForOp cur = root;
  for (unsigned depth = 0; depth < maxDepth && cur; ++depth) {
    out.push_back(cur);

    affine::AffineForOp inner;
    for (Operation &op : *cur.getBody()) {
      if (auto maybeFor = dyn_cast<affine::AffineForOp>(&op)) {
        inner = maybeFor;
        break;
      }
      if (!isHoistablePrologueOp(&op))
        break;
    }
    cur = inner;
  }
}
static LogicalResult parsePermutationString(StringRef s,
                                            SmallVectorImpl<unsigned> &out) {
  out.clear();
  if (s.empty())
    return failure();

  SmallVector<StringRef, 8> parts;
  s.split(parts, ',');
  if (parts.empty())
    return failure();

  out.reserve(parts.size());
  for (StringRef p : parts) {
    auto t = p.trim();
    unsigned v = 0;
    if (t.getAsInteger(10, v))
      return failure();
    out.push_back(v);
  }
  return success();
}

static void
collectForRootsAtDepth(Block *blk, unsigned wantDepth, unsigned curDepth,
                       SmallVectorImpl<affine::AffineForOp> &roots) {
  for (Operation &op : *blk) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      if (curDepth == wantDepth) {
        roots.push_back(forOp);
      } else {
        collectForRootsAtDepth(forOp.getBody(), wantDepth, curDepth + 1, roots);
      }
      continue;
    }
    for (Region &r : op.getRegions())
      for (Block &b : r)
        collectForRootsAtDepth(&b, wantDepth, curDepth, roots);
  }
}
static bool bandContainsOp(ArrayRef<affine::AffineForOp> band,
                           StringRef fqOpName) {
  if (band.empty())
    return false;
  affine::AffineForOp inner = band.back();
  Operation *innermost = inner.getOperation();
  bool found = false;
  innermost->walk([&](Operation *op) {
    if (op->getName().getStringRef() == fqOpName) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool isValidPermutation(ArrayRef<unsigned> perm, unsigned numLoops) {
  if (perm.size() != numLoops)
    return false;
  llvm::SmallVector<bool, 8> seen(numLoops, false);
  for (unsigned v : perm) {
    if (v >= numLoops || seen[v])
      return false;
    seen[v] = true;
  }
  return true;
}

static LogicalResult
interchangeTwoLoopBandWithSingleIterArg(affine::AffineForOp outer,
                                        affine::AffineForOp inner) {
  if (inner->getParentOp() != outer.getOperation())
    return failure();
  if (outer.getNumIterOperands() != 1 || inner.getNumIterOperands() != 1)
    return failure();

  Location loc = outer.getLoc();

  auto oLbMap = outer.getLowerBoundMap();
  auto oUbMap = outer.getUpperBoundMap();
  ValueRange oLbOps = outer.getLowerBoundOperands();
  ValueRange oUbOps = outer.getUpperBoundOperands();
  int64_t oStep = outer.getStep().getSExtValue();

  auto iLbMap = inner.getLowerBoundMap();
  auto iUbMap = inner.getUpperBoundMap();
  ValueRange iLbOps = inner.getLowerBoundOperands();
  ValueRange iUbOps = inner.getUpperBoundOperands();
  int64_t iStep = inner.getStep().getSExtValue();

  ValueRange initArgs = outer.getInits();
  if (initArgs.size() != 1)
    return failure();

  OpBuilder topBuilder(outer);
  topBuilder.setInsertionPoint(outer);
  auto newOuter = topBuilder.create<affine::AffineForOp>(
      loc, iLbOps, iLbMap, iUbOps, iUbMap, iStep, initArgs);

  Block *newOuterBody = newOuter.getBody();
  Value newOuterIV = newOuter.getInductionVar();
  Value newOuterAcc = newOuter.getRegionIterArgs()[0];

  OpBuilder innerBuilder = OpBuilder::atBlockBegin(newOuterBody);
  auto newInner = innerBuilder.create<affine::AffineForOp>(
      loc, oLbOps, oLbMap, oUbOps, oUbMap, oStep, ValueRange{newOuterAcc});

  Block *newInnerBody = newInner.getBody();
  Value newInnerIV = newInner.getInductionVar();
  Value newInnerAcc = newInner.getRegionIterArgs()[0];

  IRMapping map;
  map.map(inner.getInductionVar(), newOuterIV);
  map.map(outer.getInductionVar(), newInnerIV);
  map.map(outer.getRegionIterArgs()[0], newOuterAcc);
  map.map(inner.getRegionIterArgs()[0], newInnerAcc);

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newInnerBody);
  for (Operation &op :
       llvm::make_early_inc_range(inner.getBody()->without_terminator()))
    bodyBuilder.clone(op, map);

  auto oldInnerYield =
      cast<affine::AffineYieldOp>(inner.getBody()->getTerminator());
  Value yielded = map.lookup(oldInnerYield.getOperand(0));
  OpBuilder(newInnerBody, newInnerBody->end())
      .create<affine::AffineYieldOp>(loc, yielded);

  OpBuilder(newOuterBody, newOuterBody->end())
      .create<affine::AffineYieldOp>(loc, newInner.getResult(0));

  outer.getResult(0).replaceAllUsesWith(newOuter.getResult(0));

  outer.erase();

  return success();
}

struct CinmLoopInterchangePass
    : public impl::CinmLoopInterchangePassBase<CinmLoopInterchangePass> {
  using Base::Base;

  void runOnOperation() final {
    if (numLoops == 0) {
      getOperation()->emitError()
          << "cinm-loop-interchange: num-loops must be > 0";
      signalPassFailure();
      return;
    }
    if (permOpt.empty()) {
      getOperation()->emitError()
          << "cinm-loop-interchange: missing required --perm (e.g. 1,0)";
      signalPassFailure();
      return;
    }
    if (targetInnerOp.empty()) {
      getOperation()->emitError() << "cinm-loop-interchange: missing required "
                                     "--target-op (e.g. cinm.op.gemm)";
      signalPassFailure();
      return;
    }

    SmallVector<unsigned, 8> perm;
    if (failed(parsePermutationString(StringRef(permOpt), perm))) {
      getOperation()->emitError()
          << "cinm-loop-interchange: unable to parse --perm='" << permOpt
          << "' (expected comma-separated unsigned integers)";
      signalPassFailure();
      return;
    }
    if (!isValidPermutation(perm, numLoops)) {
      getOperation()->emitError()
          << "cinm-loop-interchange: --perm must be a permutation of 0.."
          << (numLoops - 1);
      signalPassFailure();
      return;
    }

    bool changed = false;

    getOperation()->walk([&](cinm::ComputeOp compute) {
      Region &body = compute.getBody();
      if (body.empty())
        return;

      SmallVector<affine::AffineForOp, 8> rootsAtDepth;
      collectForRootsAtDepth(&body.front(), bandDepth,
                             0, rootsAtDepth);

      for (affine::AffineForOp root : rootsAtDepth) {
        affine::AffineForOp cur = root;
        for (unsigned i = 0; i < numLoops && cur; ++i) {
          hoistLeadingPrologueOps(cur);
          affine::AffineForOp next;
          for (Operation &op : *cur.getBody()) {
            if (auto f = dyn_cast<affine::AffineForOp>(&op)) {
              next = f;
              break;
            }
            if (!isHoistablePrologueOp(&op))
              break;
          }
          cur = next;
        }

        SmallVector<affine::AffineForOp, 8> subBand;
        collectHeadNestedLoopsRelaxed(root, numLoops, subBand);
        if (subBand.size() < numLoops)
          continue;

        if (!bandContainsOp(subBand, StringRef(targetInnerOp)))
          continue;

        bool hasIter = llvm::any_of(subBand, [](affine::AffineForOp f) {
          return f.getNumIterOperands() != 0 || f.getNumResults() != 0;
        });
        if (hasIter) {
          if (subBand.size() == 2) {
            auto outer = subBand[0];
            auto inner = subBand[1];
            if (succeeded(
                    interchangeTwoLoopBandWithSingleIterArg(outer, inner))) {
              changed = true;
              continue;
            }
          }
          continue;
        }

        SmallVector<unsigned, 8> perm;
        if (failed(parsePermutationString(StringRef(permOpt), perm)) ||
            !isValidPermutation(perm, numLoops) ||
            perm.size() != subBand.size()) {
          compute.emitError()
              << "invalid permutation for band of size " << subBand.size();
          signalPassFailure();
          return;
        }

        SmallVector<affine::AffineForOp, 8> target(subBand.size());
        for (unsigned oldIdx = 0; oldIdx < subBand.size(); ++oldIdx)
          target[perm[oldIdx]] = subBand[oldIdx];

        SmallVector<affine::AffineForOp, 8> curr = subBand;
        auto bubbleTo = [&](unsigned from, unsigned to) {
          for (unsigned i = from; i > to; --i) {
            mlir::affine::interchangeLoops(curr[i - 1], curr[i]);
            std::swap(curr[i - 1], curr[i]);
          }
        };

        for (unsigned i = 0; i < target.size(); ++i) {
          unsigned pos = 0;
          while (pos < curr.size() &&
                 curr[pos].getOperation() != target[i].getOperation())
            ++pos;
          if (pos == curr.size()) {
            compute.emitError() << "internal error: band loop not found";
            signalPassFailure();
            return;
          }
          if (pos != i)
            bubbleTo(pos, i);
        }

        changed = true;
      }
    });

    (void)changed;
  }
};

}
}
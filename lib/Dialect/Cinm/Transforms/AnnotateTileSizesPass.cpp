#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMANNOTATETILESIZESPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

static LogicalResult parseTileSizesString(StringRef s,
                                          SmallVectorImpl<int64_t> &out) {
  out.clear();
  if (s.empty())
    return failure();

  SmallVector<StringRef, 8> parts;
  if (s.contains('x') || s.contains('X')) {
    s.split(parts, 'x');
    if (parts.size() == 1) {
      parts.clear();
      s.split(parts, 'X');
    }
  } else {
    s.split(parts, ',');
  }
  if (parts.empty())
    return failure();

  out.reserve(parts.size());
  for (StringRef p : parts) {
    int64_t v = 0;
    if (p.trim().getAsInteger(10, v))
      return failure();
    out.push_back(v);
  }
  return success();
}

static cinm::ComputeOp
wrapArbitraryCinmTensorOpInCompute(Operation *op, DenseI64ArrayAttr tileSizes,
                                   IRRewriter &rewriter) {
  using cinm::ComputeOp;
  using cinm::YieldOp;

  if (auto parent = op->getParentOfType<ComputeOp>()) {
    if (tileSizes)
      parent->setAttr("tileSizes", tileSizes);
    return parent;
  }

  if (op->getNumResults() != 1)
    return {};
  auto rty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!rty)
    return {};

  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op->getLoc();

  rewriter.setInsertionPoint(op);
  auto compute = rewriter.create<ComputeOp>(loc, op->getOperands(), rty);
  if (tileSizes)
    compute->setAttr("tileSizes", tileSizes);

  // remap original values to block argument
  IRMapping mapping;
  for (auto [arg, opnd] : compute.zipArgsWithOperands()) {
    mapping.map(opnd, arg);
  }

  Region &region = compute.getBody();
  Block &entry = region.front();

  rewriter.setInsertionPointToStart(&entry);
  Operation *inner = rewriter.clone(*op, mapping);
  rewriter.create<YieldOp>(loc, inner->getResult(0));

  rewriter.setInsertionPointAfter(compute);
  rewriter.replaceOp(op, compute.getResult(0));
  return compute;
}

template <typename OpT> static bool regionContainsOp(Region &r) {
  bool found = false;
  auto res = r.walk([&](OpT) -> WalkResult {
    found = true;
    return WalkResult::interrupt();
  });
  (void)res;
  return found;
}

struct CinmAnnotateTileSizesPass
    : public impl::CinmAnnotateTileSizesPassBase<CinmAnnotateTileSizesPass> {
  using Base::Base;
  void runOnOperation() final {
    Operation *root = getOperation();
    IRRewriter rewriter(&getContext());
    Builder b(&getContext());

    if (opsOpt.empty())
      opsOpt = "compute";
    llvm::StringSet<> kinds;
    {
      StringRef ops = StringRef(opsOpt);
      SmallVector<StringRef, 4> parts;
      ops.split(parts, ',');
      for (StringRef p : parts)
        kinds.insert(p.trim().lower());
    }
    const bool wantCompute = kinds.contains("compute");
    const bool wantGemm = kinds.contains("gemm");
    const bool wantGemv = kinds.contains("gemv");
    const bool wantBatchGemm = kinds.contains("batch_gemm");
    const bool wantBatchGemv = kinds.contains("batch_gemv");
    const bool wantActivate = kinds.contains("activate");

    DenseI64ArrayAttr tileSizesAttr;
    if (wantCompute || wantGemm || wantGemv || wantBatchGemm || wantBatchGemv ||
        wantActivate) {
      if (tileSizesOpt.empty()) {
        root->emitError()
            << "cinm-annotate-tiles: missing --tile-sizes for ops=" << opsOpt;
        signalPassFailure();
        return;
      }
      SmallVector<int64_t, 8> sizes;
      if (failed(parseTileSizesString(StringRef(tileSizesOpt), sizes))) {
        root->emitError()
            << "cinm-annotate-tiles: unable to parse --tile-sizes='"
            << tileSizesOpt << "' (use 'x'/'X' or ',' separators)";
        signalPassFailure();
        return;
      }
      if (sizes.empty() || sizes.size() > 4) {
        root->emitError() << "cinm-annotate-tiles: expected 1..4 integers; got "
                          << sizes.size();
        signalPassFailure();
        return;
      }
      for (int64_t v : sizes) {
        if (v <= 0) {
          root->emitError()
              << "cinm-annotate-tiles: tile sizes must be positive";
          signalPassFailure();
          return;
        }
      }
      tileSizesAttr = b.getDenseI64ArrayAttr(ArrayRef<int64_t>(sizes));
    }

    if (wantCompute) {
      root->walk([&](cinm::ComputeOp compute) {
        compute->setAttr("tileSizes", tileSizesAttr);
      });
    }

    if (wantGemm || wantGemv || wantBatchGemm || wantBatchGemv) {
      SmallVector<Operation *, 16> toWrap;
      root->walk([&](Operation *o) {
        if (!o->getDialect() || o->getDialect()->getNamespace() !=
                                    cinm::CinmDialect::getDialectNamespace())
          return;
        if (isa<cinm::ComputeOp, cinm::YieldOp>(o))
          return;
        if (o->getNumResults() != 1 ||
            !isa<RankedTensorType>(o->getResult(0).getType()))
          return;
        if (o->getParentOfType<cinm::ComputeOp>())
          return;

        if ((wantGemm && isa<cinm::GemmOp>(o)) ||
            (wantGemv && isa<cinm::GemvOp>(o)) ||
            (wantBatchGemm && isa<cinm::BatchGemmOp>(o)) ||
            (wantBatchGemv && isa<cinm::BatchGemvOp>(o)))
          toWrap.push_back(o);
      });
      for (Operation *o : toWrap)
        (void)wrapArbitraryCinmTensorOpInCompute(o, tileSizesAttr, rewriter);

      root->walk([&](cinm::ComputeOp compute) {
        bool match = false;
        if (wantGemm)
          match |= regionContainsOp<cinm::GemmOp>(compute.getRegion());
        if (wantGemv)
          match |= regionContainsOp<cinm::GemvOp>(compute.getRegion());
        if (wantBatchGemm)
          match |= regionContainsOp<cinm::BatchGemmOp>(compute.getRegion());
        if (wantBatchGemv)
          match |= regionContainsOp<cinm::BatchGemvOp>(compute.getRegion());
        if (match)
          compute->setAttr("tileSizes", tileSizesAttr);
      });
    }

    if (wantActivate) {
      SmallVector<Operation *, 16> toWrap;
      root->walk([&](cinm::ActivateOp act) {
        if (!act->getParentOfType<cinm::ComputeOp>())
          toWrap.push_back(act.getOperation());
      });
      for (Operation *o : toWrap)
        (void)wrapArbitraryCinmTensorOpInCompute(o, tileSizesAttr, rewriter);

      root->walk([&](cinm::ComputeOp compute) {
        if (regionContainsOp<cinm::ActivateOp>(compute.getRegion()))
          compute->setAttr("tileSizes", tileSizesAttr);
      });
    }
  }
};

} // namespace
} // namespace mlir::cinm

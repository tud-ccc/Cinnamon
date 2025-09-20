#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

struct RewriteScfTensorIterArgsToMemref final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp oldFor,
                                PatternRewriter &rewriter) const override {
    ValueRange oldInits = oldFor.getInitArgs();
    if (oldInits.empty())
      return failure();

    bool anyTensor = llvm::any_of(oldInits, [](Value v) {
      return isa<RankedTensorType>(v.getType());
    });
    if (!anyTensor)
      return failure();

    Location loc = oldFor.getLoc();

    SmallVector<Value> memInitArgs;
    memInitArgs.reserve(oldInits.size());
    SmallVector<Type> memIterTypes;
    memIterTypes.reserve(oldInits.size());
    for (Value init : oldInits) {
      if (auto mt = dyn_cast<MemRefType>(init.getType())) {
        memInitArgs.push_back(init);
        memIterTypes.push_back(mt);
        continue;
      }
      auto tt = dyn_cast<RankedTensorType>(init.getType());
      if (!tt)
        return failure();
      BaseMemRefType mr = bufferization::getMemRefTypeWithFullyDynamicLayout(tt);
      Value mem = rewriter.create<bufferization::ToMemrefOp>(loc, mr, init,
                                                             false);
      memInitArgs.push_back(mem);
      memIterTypes.push_back(mem.getType());
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(oldFor);
    scf::ForOp newFor = rewriter.create<scf::ForOp>(
        loc, oldFor.getLowerBound(), oldFor.getUpperBound(), oldFor.getStep(),
        memInitArgs);

    Block *oldBody = oldFor.getBody();
    Block *newBody = newFor.getBody();

    IRMapping mapper;
    mapper.map(oldBody->getArgument(0), newBody->getArgument(0));

    {
      OpBuilder::InsertionGuard bg(rewriter);
      rewriter.setInsertionPointToStart(newBody);
      for (unsigned i = 0, e = oldInits.size(); i < e; ++i) {
        BlockArgument newMem = newBody->getArgument(1 + i);
        BlockArgument oldArg = oldBody->getArgument(1 + i);
        if (isa<MemRefType>(oldArg.getType())) {
          mapper.map(oldArg, newMem);
        } else {
          auto tt = cast<RankedTensorType>(oldArg.getType());
          Value tview = rewriter.create<bufferization::ToTensorOp>(
              loc, tt, newMem, true, true);
          mapper.map(oldArg, tview);
        }
      }

      for (Operation &nested :
           llvm::make_early_inc_range(oldBody->without_terminator()))
        rewriter.clone(nested, mapper);

      auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
      SmallVector<Value> newYields;
      newYields.reserve(oldYield.getNumOperands());
      for (unsigned i = 0, e = oldYield.getNumOperands(); i < e; ++i) {
        Value mapped = mapper.lookup(oldYield.getOperand(i));
        auto expectTy = cast<MemRefType>(newBody->getArgument(1 + i).getType());
        Value y = mapped;
        if (!isa<MemRefType>(y.getType())) {
          y = rewriter.create<bufferization::ToMemrefOp>(loc, expectTy, y,
                                                         false);
        } else if (y.getType() != Type(expectTy)) {
          y = rewriter.create<memref::CastOp>(loc, expectTy, y);
        }
        newYields.push_back(y);
      }

      Operation *maybeTerm = nullptr;
      if (!newBody->empty()) {
        Operation &last = newBody->back();
        if (last.hasTrait<OpTrait::IsTerminator>())
          maybeTerm = &last;
      }
      if (maybeTerm) {
        rewriter.setInsertionPoint(maybeTerm);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(maybeTerm, newYields);
      } else {
        rewriter.setInsertionPointToEnd(newBody);
        rewriter.create<scf::YieldOp>(loc, newYields);
      }
    }

    SmallVector<Value> repls;
    repls.reserve(newFor.getNumResults());
    rewriter.setInsertionPointAfter(newFor);
    for (auto it : llvm::enumerate(newFor.getResults())) {
      Value res = it.value();
      Type oldTy = oldFor.getResult(it.index()).getType();
      if (isa<RankedTensorType>(oldTy)) {
        Value t = rewriter.create<bufferization::ToTensorOp>(
            loc, oldTy, res, true, true);
        repls.push_back(t);
      } else {
        repls.push_back(res);
      }
    }

    rewriter.replaceOp(oldFor, repls);
    return success();
  }
};


struct LowerDynamicShapeCopyAnyRank final : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
    auto dstTy = dyn_cast<MemRefType>(op.getTarget().getType());
    if (!srcTy || !dstTy)
      return failure();

    if (srcTy.hasStaticShape() && dstTy.hasStaticShape())
      return failure();

    if (srcTy.getRank() != dstTy.getRank())
      return op.emitOpError("rank mismatch in memref.copy");

    Location loc = op.getLoc();
    const int64_t rank = srcTy.getRank();

    SmallVector<Value> extents;
    extents.reserve(rank);
    for (int64_t d = 0; d < rank; ++d) {
      if (!srcTy.isDynamicDim(d)) {
        extents.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, srcTy.getDimSize(d)));
      } else if (!dstTy.isDynamicDim(d)) {
        extents.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, dstTy.getDimSize(d)));
      } else {
        extents.push_back(
            rewriter.create<memref::DimOp>(loc, op.getSource(), d));
      }
    }

    SmallVector<Value> ivs;
    ivs.reserve(rank);

    std::function<void(int64_t)> buildLoop = [&](int64_t depth) {
      if (depth == rank) {
        Value v = rewriter.create<memref::LoadOp>(loc, op.getSource(), ivs);
        rewriter.create<memref::StoreOp>(loc, v, op.getTarget(), ivs);
        return;
      }
      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = rewriter.create<scf::ForOp>(loc, c0, extents[depth], c1);

      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      buildLoop(depth + 1);
      ivs.pop_back();

      rewriter.setInsertionPointAfter(loop);
    };

    rewriter.setInsertionPoint(op);
    buildLoop(0);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerRank1ContiguousCopy final : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
    auto dstTy = dyn_cast<MemRefType>(op.getTarget().getType());
    if (!srcTy || !dstTy)
      return failure();

    if (srcTy.getRank() != 1 || dstTy.getRank() != 1)
      return failure();
    if (srcTy.hasStaticShape() && dstTy.hasStaticShape())
      return failure();

    auto isContiguousRank1 = [](MemRefType ty) -> bool {
      if (ty.getRank() != 1)
        return false;
      Attribute layout = ty.getLayout();
      if (!layout)
        return true;
      if (auto strided = dyn_cast<StridedLayoutAttr>(layout)) {
        auto strides = strided.getStrides();
        auto off = strided.getOffset();
        return off == 0 && strides.size() == 1 && strides[0] == 1;
      }
      return false;
    };

    if (!isContiguousRank1(srcTy) || !isContiguousRank1(dstTy))
      return failure();

    Location loc = op.getLoc();
    Value extent;
    if (!srcTy.isDynamicDim(0)) {
      extent =
          rewriter.create<arith::ConstantIndexOp>(loc, srcTy.getDimSize(0));
    } else if (!dstTy.isDynamicDim(0)) {
      extent =
          rewriter.create<arith::ConstantIndexOp>(loc, dstTy.getDimSize(0));
    } else {
      extent = rewriter.create<memref::DimOp>(loc, op.getSource(), 0);
    }

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, c0, extent, c1);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    Value v = rewriter.create<memref::LoadOp>(loc, op.getSource(), i);
    rewriter.create<memref::StoreOp>(loc, v, op.getTarget(), i);
    rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReplaceRank1CopyLoopWithMemRefCopy final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto c0 = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto c1 = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!c0 || !c1 || c0.value() != 0 || c1.value() != 1)
      return failure();

    memref::LoadOp load;
    memref::StoreOp store;
    for (Operation &op : *forOp.getBody()) {
      if (isa<scf::YieldOp>(op))
        continue;
      if (auto l = dyn_cast<memref::LoadOp>(op)) {
        if (load)
          return failure();
        load = l;
        continue;
      }
      if (auto s = dyn_cast<memref::StoreOp>(op)) {
        if (store)
          return failure();
        store = s;
        continue;
      }
      return failure();
    }
    if (!load || !store)
      return failure();

    if (load.getIndices().size() != 1 || store.getIndices().size() != 1)
      return failure();
    if (load.getIndices().front() != forOp.getInductionVar())
      return failure();
    if (store.getIndices().front() != forOp.getInductionVar())
      return failure();

    auto srcTy = dyn_cast<MemRefType>(load.getMemref().getType());
    auto dstTy = dyn_cast<MemRefType>(store.getMemref().getType());
    if (!srcTy || !dstTy || srcTy.getRank() != 1 || dstTy.getRank() != 1)
      return failure();
    if (srcTy.getElementType() != dstTy.getElementType())
      return failure();

    Value ub = forOp.getUpperBound();
    bool ubMatches = false;
    if (auto dim = ub.getDefiningOp<memref::DimOp>()) {
      if (dim.getSource() == load.getMemref() &&
          dim.getIndex() == forOp.getInductionVar())
        ubMatches = true;
    }
    (void)ubMatches;

    rewriter.setInsertionPoint(forOp);
    rewriter.create<memref::CopyOp>(forOp.getLoc(), load.getMemref(),
                                    store.getMemref());
    rewriter.eraseOp(forOp);
    return success();
  }
};

struct ReplacePerfectCopyNestWithMemRefCopy final
    : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp->getParentOfType<scf::ForOp>())
      return failure();

    SmallVector<scf::ForOp> nest;
    nest.push_back(forOp);
    scf::ForOp cur = forOp;
    while (true) {
      scf::ForOp next = nullptr;
      for (Operation &op : *cur.getBody()) {
        if (auto inner = dyn_cast<scf::ForOp>(op)) {
          if (next)
            return failure();
          next = inner;
        } else if (isa<scf::YieldOp>(op)) {
          continue;
        } else {
        }
      }
      if (!next)
        break;
      nest.push_back(next);
      cur = next;
    }

    memref::LoadOp load;
    memref::StoreOp store;
    for (Operation &op : *cur.getBody()) {
      if (isa<scf::YieldOp>(op))
        continue;
      if (auto l = dyn_cast<memref::LoadOp>(op)) {
        if (load)
          return failure();
        load = l;
        continue;
      }
      if (auto s = dyn_cast<memref::StoreOp>(op)) {
        if (store)
          return failure();
        store = s;
        continue;
      }
      return failure();
    }
    if (!load || !store)
      return failure();
    if (store.getValue() != load.getResult())
      return failure();

    SmallVector<Value> ivs;
    ivs.reserve(nest.size());
    for (scf::ForOp f : nest)
      ivs.push_back(f.getInductionVar());

    auto matchIndices = [&](ValueRange idxs) -> bool {
      if (idxs.size() != ivs.size())
        return false;
      for (auto it : llvm::zip(idxs, ivs))
        if (std::get<0>(it) != std::get<1>(it))
          return false;
      return true;
    };

    if (!matchIndices(load.getIndices()) || !matchIndices(store.getIndices()))
      return failure();

    auto srcTy = dyn_cast<MemRefType>(load.getMemref().getType());
    auto dstTy = dyn_cast<MemRefType>(store.getMemref().getType());
    if (!srcTy || !dstTy)
      return failure();
    if (srcTy.getElementType() != dstTy.getElementType())
      return failure();

    rewriter.setInsertionPoint(forOp);
    rewriter.create<memref::CopyOp>(forOp.getLoc(), load.getMemref(),
                                    store.getMemref());
    rewriter.eraseOp(forOp);
    return success();
  }
};

struct ReplacePerfectAffineCopyNestWithMemRefCopy final
    : OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp->getParentOfType<affine::AffineForOp>())
      return failure();

    SmallVector<affine::AffineForOp> nest;
    nest.push_back(forOp);
    affine::AffineForOp cur = forOp;
    while (true) {
      affine::AffineForOp next = nullptr;
      for (Operation &op : *cur.getBody()) {
        if (auto inner = dyn_cast<affine::AffineForOp>(op)) {
          if (next)
            return failure();
          next = inner;
        } else if (isa<affine::AffineYieldOp>(op)) {
          continue;
        } else {
        }
      }
      if (!next)
        break;
      nest.push_back(next);
      cur = next;
    }

    memref::LoadOp load;
    memref::StoreOp store;
    for (Operation &op : *cur.getBody()) {
      if (isa<affine::AffineYieldOp>(op))
        continue;
      if (auto l = dyn_cast<memref::LoadOp>(op)) {
        if (load)
          return failure();
        load = l;
        continue;
      }
      if (auto s = dyn_cast<memref::StoreOp>(op)) {
        if (store)
          return failure();
        store = s;
        continue;
      }
      return failure();
    }
    if (!load || !store)
      return failure();
    if (store.getValue() != load.getResult())
      return failure();

    SmallVector<Value> ivs;
    ivs.reserve(nest.size());
    for (affine::AffineForOp f : nest)
      ivs.push_back(f.getInductionVar());

    auto matchIndices = [&](ValueRange idxs) -> bool {
      if (idxs.size() != ivs.size())
        return false;
      for (auto it : llvm::zip(idxs, ivs))
        if (std::get<0>(it) != std::get<1>(it))
          return false;
      return true;
    };

    if (!matchIndices(load.getIndices()) || !matchIndices(store.getIndices()))
      return failure();

    auto srcTy = dyn_cast<MemRefType>(load.getMemref().getType());
    auto dstTy = dyn_cast<MemRefType>(store.getMemref().getType());
    if (!srcTy || !dstTy)
      return failure();
    if (srcTy.getElementType() != dstTy.getElementType())
      return failure();

    rewriter.setInsertionPoint(forOp);
    rewriter.create<memref::CopyOp>(forOp.getLoc(), load.getMemref(),
                                    store.getMemref());
    rewriter.eraseOp(forOp);
    return success();
  }
};

struct ReplaceSimpleAffineForToScf final : OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isConstZeroMap(AffineMap m) {
    if (m.getNumResults() != 1 || m.getNumDims() != 0 || m.getNumSymbols() != 0)
      return false;
    if (auto cexpr = llvm::dyn_cast<AffineConstantExpr>(m.getResult(0)))
      return cexpr.getValue() == 0;
    return false;
  }

  static bool isDim0Map(AffineMap m) {
    if (m.getNumResults() != 1 || m.getNumDims() != 1 || m.getNumSymbols() != 0)
      return false;
    if (auto dexpr = llvm::dyn_cast<AffineDimExpr>(m.getResult(0)))
      return dexpr.getPosition() == 0;
    return false;
  }

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp.getStep() != 1)
      return failure();
    if (!isConstZeroMap(forOp.getLowerBoundMap()))
      return failure();
    if (!isDim0Map(forOp.getUpperBoundMap()))
      return failure();
    if (forOp.getUpperBoundOperands().size() != 1)
      return failure();

    for (Operation &op : *forOp.getBody()) {
      if (isa<affine::AffineYieldOp>(op))
        continue;
      if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
        if (!isDim0Map(st.getAffineMap()))
          return failure();
        continue;
      }
      return failure();
    }

    Location loc = forOp.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value ub = forOp.getUpperBoundOperands().front();
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, c0, ub, c1);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();

    for (Operation &op : llvm::make_early_inc_range(*forOp.getBody())) {
      if (isa<affine::AffineYieldOp>(op))
        continue;
      if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
        rewriter.create<memref::StoreOp>(loc, st.getValue(), st.getMemRef(), iv);
        continue;
      }
    }
    rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(forOp);
    return success();
  }
};

struct ReplaceLinalgGenericPassthroughWithCopy
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    if (b.empty())
      return failure();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();

    if (b.getNumArguments() < 1)
      return failure();
    if (yield.getOperand(0) != b.getArgument(0))
      return failure();

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity())
      return failure();

    auto stripCast = [](Value v) -> Value {
      if (auto c = v.getDefiningOp<memref::CastOp>())
        return c.getSource();
      return v;
    };
    Value in = stripCast(op.getInputs()[0]);
    Value out = stripCast(op.getOutputs()[0]);
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, in, out);
    return success();
  }
};

struct ReplaceLinalgGenericSameMapPassthroughWithCopy
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    for (auto it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();

    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    if (maps.size() != 2)
      return failure();
    if (maps[0] != maps[1])
      return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    if (b.getNumArguments() < 1)
      return failure();
    if (yield.getOperand(0) != b.getArgument(0))
      return failure();

    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getInputs()[0],
                                                op.getOutputs()[0]);
    return success();
  }
};

struct ReplaceLinalgTransposeIdentityWithCopy
    : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto perm = llvm::to_vector(op.getPermutation());
    for (unsigned i = 0, e = perm.size(); i < e; ++i)
      if (perm[i] != i)
        return failure();
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getInput(),
                                                op.getInit());
    return success();
  }
};

struct ReplaceLinalgGenericConstantWithFill
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInputs().empty() || op.getOutputs().size() != 1)
      return failure();
    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    Value v = yield.getOperand(0);
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return failure();
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, cst.getResult(),
                                                op.getOutputs()[0]);
    return success();
  }
};

struct EraseLinalgGenericNoOp : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOutputs().size() != 1)
      return failure();
    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    if (maps.size() != 1 + op.getInputs().size())
      return failure();
    for (AffineMap m : maps)
      if (!m.isIdentity())
        return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    if (b.getNumArguments() == 0)
      return failure();
    if (yield.getOperand(0) != b.getArguments().back())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReplaceLinalgGenericIndexRange1DWithLoop
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInputs().empty() || op.getOutputs().size() != 1)
      return failure();
    if (op.getIteratorTypes().size() != 1)
      return failure();
    if (op.getIteratorTypesArray()[0] != utils::IteratorType::parallel)
      return failure();
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    if (maps.size() != 1 || !maps[0].isIdentity())
      return failure();

    auto outTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!outTy || outTy.getRank() != 1 || !outTy.getElementType().isSignlessInteger(64))
      return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();

    Value yv = yield.getOperand(0);
    auto cast = yv.getDefiningOp<arith::IndexCastOp>();
    if (!cast)
      return failure();
    auto idx = cast.getIn().getDefiningOp<linalg::IndexOp>();
    if (!idx || idx.getDim() != 0)
      return failure();

    Location loc = op.getLoc();
    Value out = op.getOutputs()[0];
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value ub = rewriter.create<memref::DimOp>(loc, out, 0);
    auto loop = rewriter.create<scf::ForOp>(loc, c0, ub, c1);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value iv64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), iv);
    rewriter.create<memref::StoreOp>(loc, iv64, out, iv);
    rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReplaceLinalgGenericIndexRangeNDWithLoops
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInputs().empty() || op.getOutputs().size() != 1)
      return failure();
    if (!op.getIteratorTypes().size())
      return failure();
    for (auto it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    if (maps.size() != 1 || !maps[0].isIdentity())
      return failure();

    auto outTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!outTy || !outTy.getElementType().isSignlessInteger(64))
      return failure();
    unsigned rank = static_cast<unsigned>(outTy.getRank());
    if (rank == 0 || rank != op.getIteratorTypes().size())
      return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    Value yv = yield.getOperand(0);
    auto cast = yv.getDefiningOp<arith::IndexCastOp>();
    if (!cast)
      return failure();
    auto idx = cast.getIn().getDefiningOp<linalg::IndexOp>();
    if (!idx)
      return failure();
    unsigned dimK = static_cast<unsigned>(idx.getDim());
    if (dimK >= rank)
      return failure();

    Location loc = op.getLoc();
    Value out = op.getOutputs()[0];
    SmallVector<Value> lbs(rank), ubs(rank), steps(rank);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (unsigned d = 0; d < rank; ++d) {
      lbs[d] = c0;
      ubs[d] = rewriter.create<memref::DimOp>(loc, out, d);
      steps[d] = c1;
    }

    auto createNest = [&](auto &&self, unsigned d, SmallVector<Value> &ivs) -> void {
      if (d == rank) {
        Value which = ivs[dimK];
        Value as64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), which);
        rewriter.create<memref::StoreOp>(loc, as64, out, ivs);
        return;
      }
      auto loop = rewriter.create<scf::ForOp>(loc, lbs[d], ubs[d], steps[d]);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      self(self, d + 1, ivs);
      ivs.pop_back();
      rewriter.setInsertionPointAfter(loop);
    };

    rewriter.setInsertionPoint(op);
    SmallVector<Value> ivs;
    createNest(createNest, 0, ivs);
    rewriter.eraseOp(op);
    return success();
  }
};
struct ReplaceLinalgGenericForwardToLoops
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isAllParallel(linalg::GenericOp op) {
    for (auto it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return false;
    return true;
  }

  static bool isIdentity(AffineMap m) { return m.isIdentity(); }

  static LogicalResult getDimPositions(AffineMap m, SmallVectorImpl<unsigned> &pos) {
    if (m.getNumSymbols() != 0)
      return failure();
    pos.clear();
    pos.reserve(m.getNumResults());
    for (AffineExpr e : m.getResults()) {
      if (auto d = llvm::dyn_cast<AffineDimExpr>(e)) {
        pos.push_back(static_cast<unsigned>(d.getPosition()));
      } else {
        return failure();
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics())
      return failure();
    if (op.getOutputs().size() != 1)
      return failure();
    if (!isAllParallel(op))
      return failure();

    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    if (maps.empty() || !isIdentity(maps.back()))
      return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();

    unsigned numInputs = op.getInputs().size();
    unsigned numOutputs = op.getOutputs().size();
    if (b.getNumArguments() != numInputs + numOutputs)
      return failure();

    Value yv = yield.getOperand(0);
    int forwardedInputIdx = -1;
    for (unsigned i = 0; i < numInputs; ++i) {
      if (yv == b.getArgument(i)) {
        forwardedInputIdx = static_cast<int>(i);
        break;
      }
    }
    if (forwardedInputIdx < 0)
      return failure();

    Value inView = op.getInputs()[forwardedInputIdx];
    Value outView = op.getOutputs()[0];
    auto inTy = dyn_cast<MemRefType>(inView.getType());
    auto outTy = dyn_cast<MemRefType>(outView.getType());
    if (!inTy || !outTy)
      return failure();

    SmallVector<unsigned, 4> inPos;
    if (failed(getDimPositions(maps[forwardedInputIdx], inPos)))
      return failure();

    Location loc = op.getLoc();
    unsigned rank = outTy.getRank();
    SmallVector<Value> lbs(rank), ubs(rank), steps(rank);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (unsigned d = 0; d < rank; ++d) {
      lbs[d] = c0;
      ubs[d] = rewriter.create<memref::DimOp>(loc, outView, d);
      steps[d] = c1;
    }

    auto createNest = [&](auto &&self, unsigned d, SmallVector<Value> &ivs) -> void {
      if (d == rank) {
        SmallVector<Value, 4> inIdx;
        inIdx.reserve(inPos.size());
        for (unsigned p : inPos) {
          if (p >= ivs.size()) return;
          inIdx.push_back(ivs[p]);
        }
        Value val = rewriter.create<memref::LoadOp>(loc, inView, inIdx);
        rewriter.create<memref::StoreOp>(loc, val, outView, ivs);
        return;
      }
      auto loop = rewriter.create<scf::ForOp>(loc, lbs[d], ubs[d], steps[d]);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      self(self, d + 1, ivs);
      ivs.pop_back();
      rewriter.setInsertionPointAfter(loop);
    };

    rewriter.setInsertionPoint(op);
    SmallVector<Value> ivs;
    createNest(createNest, 0, ivs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReplaceLinalgGenericIndexYieldWithLoops
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics())
      return failure();
    if (op.getOutputs().size() != 1)
      return failure();
    for (auto it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    if (maps.empty() || !maps.back().isIdentity())
      return failure();

    auto outTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!outTy || !outTy.getElementType().isSignlessInteger(64))
      return failure();
    unsigned rank = static_cast<unsigned>(outTy.getRank());
    if (rank == 0 || rank != op.getIteratorTypes().size())
      return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    Value yv = yield.getOperand(0);
    auto cast = yv.getDefiningOp<arith::IndexCastOp>();
    if (!cast)
      return failure();
    auto idx = cast.getIn().getDefiningOp<linalg::IndexOp>();
    if (!idx)
      return failure();
    unsigned dimK = static_cast<unsigned>(idx.getDim());
    if (dimK >= rank)
      return failure();

    Location loc = op.getLoc();
    Value out = op.getOutputs()[0];
    SmallVector<Value> lbs(rank), ubs(rank), steps(rank);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (unsigned d = 0; d < rank; ++d) {
      lbs[d] = c0;
      ubs[d] = rewriter.create<memref::DimOp>(loc, out, d);
      steps[d] = c1;
    }

    auto createNest = [&](auto &&self, unsigned d, SmallVector<Value> &ivs) -> void {
      if (d == rank) {
        Value which = ivs[dimK];
        Value as64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), which);
        rewriter.create<memref::StoreOp>(loc, as64, out, ivs);
        return;
      }
      auto loop = rewriter.create<scf::ForOp>(loc, lbs[d], ubs[d], steps[d]);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      self(self, d + 1, ivs);
      ivs.pop_back();
      rewriter.setInsertionPointAfter(loop);
    };

    rewriter.setInsertionPoint(op);
    SmallVector<Value> ivs;
    createNest(createNest, 0, ivs);
    rewriter.eraseOp(op);
    return success();
  }
};
struct FuseActivateTmpCopyPattern final : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    Value tmp = copy.getSource();
    Value dst = copy.getTarget();

    auto act = tmp.getDefiningOp<cinm::ActivateMemRefOp>();
    if (!act)
      return failure();

    if (act->getNumOperands() != 2 || act->getOperand(1) != tmp)
      return failure();

    Operation *tmpDef = tmp.getDefiningOp();
    if (!tmpDef || !isa<memref::AllocOp, memref::AllocaOp>(tmpDef))
      return failure();

    SmallVector<memref::DeallocOp, 1> deallocs;
    for (OpOperand &use : tmp.getUses()) {
      Operation *user = use.getOwner();
      if (user == act)
        continue;
      if (auto c = dyn_cast<memref::CopyOp>(user)) {
        if (c != copy || use.getOperandNumber() != 0)
          return failure();
        continue;
      }
      if (auto d = dyn_cast<memref::DeallocOp>(user)) {
        deallocs.push_back(d);
        continue;
      }
      return failure();
    }

    Location loc = act.getLoc();
    Value src = act->getOperand(0);

    OperationState st(loc, cinm::ActivateMemRefOp::getOperationName());
    st.addOperands({src, dst});
    st.addAttribute("kind", cinm::ActivationKindAttr::get(rewriter.getContext(),
                                                          act.getKind()));
    (void)rewriter.create(st);

    rewriter.eraseOp(copy);
    rewriter.eraseOp(act);

    bool onlyDeallocsRemain = llvm::all_of(tmp.getUses(), [](OpOperand &u) {
      return isa<memref::DeallocOp>(u.getOwner());
    });
    if (onlyDeallocsRemain) {
      for (memref::DeallocOp d : deallocs)
        rewriter.eraseOp(d);
      rewriter.eraseOp(tmpDef);
    }
    return success();
  }
};

struct ReplaceLinalgGenericCapturedValueWithFill
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics())
      return failure();
    if (op.getOutputs().size() != 1)
      return failure();
    for (auto it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    if (maps.empty() || !maps.back().isIdentity())
      return failure();

    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &b = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    Value yv = yield.getOperand(0);
    for (BlockArgument barg : b.getArguments())
      if (yv == barg)
        return failure();
    auto outTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!outTy)
      return failure();
    if (yv.getType() != outTy.getElementType())
      return failure();

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, yv, op.getOutputs()[0]);
    return success();
  }
};
struct CinmMemoryCleanupPass
    : public CinmMemoryCleanupPassBase<CinmMemoryCleanupPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);

    if (fuseActivateTmpCopy)
      patterns.add<FuseActivateTmpCopyPattern>(&ctx);

    if (lowerAnyDynamicCopy) {
      patterns.add<LowerDynamicShapeCopyAnyRank>(&ctx);
    } else if (lowerDynRank1Copy) {
      patterns.add<LowerRank1ContiguousCopy>(&ctx);
    }

    patterns.add<RewriteScfTensorIterArgsToMemref>(&ctx);

    patterns.add<ReplaceRank1CopyLoopWithMemRefCopy>(&ctx);
    patterns.add<ReplacePerfectCopyNestWithMemRefCopy>(&ctx);
    patterns.add<ReplacePerfectAffineCopyNestWithMemRefCopy>(&ctx);

    patterns.add<ReplaceLinalgGenericPassthroughWithCopy>(&ctx);
    patterns.add<ReplaceLinalgGenericSameMapPassthroughWithCopy>(&ctx);
    patterns.add<ReplaceLinalgGenericConstantWithFill>(&ctx);
    patterns.add<ReplaceLinalgTransposeIdentityWithCopy>(&ctx);
    patterns.add<EraseLinalgGenericNoOp>(&ctx);
    patterns.add<ReplaceLinalgGenericIndexRange1DWithLoop>(&ctx);
    patterns.add<ReplaceLinalgGenericIndexRangeNDWithLoops>(&ctx);
    patterns.add<ReplaceLinalgGenericIndexYieldWithLoops>(&ctx);
    patterns.add<ReplaceLinalgGenericForwardToLoops>(&ctx);
    patterns.add<ReplaceSimpleAffineForToScf>(&ctx);

    GreedyRewriteConfig cfg;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), cfg)))
      return signalPassFailure();
  }
};

}

namespace mlir::cinm {
std::unique_ptr<Pass> createCinmMemoryCleanupPass() {
  return std::make_unique<CinmMemoryCleanupPass>();
}
}

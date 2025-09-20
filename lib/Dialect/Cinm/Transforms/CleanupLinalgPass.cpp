#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::cinm;

namespace mlir::cinm {
namespace {

static Value createFullSliceInsert(IRRewriter &rewriter, Location loc,
                                   Value src, Value dest) {
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  auto destType = dyn_cast<RankedTensorType>(dest.getType());
  if (!srcType || !destType || srcType != destType)
    return Value();

  int64_t rank = srcType.getRank();
  SmallVector<int64_t> staticOffsets(rank, 0);
  SmallVector<int64_t> staticStrides(rank, 1);
  SmallVector<int64_t> staticSizes(rank, ShapedType::kDynamic);
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (srcType.isDynamicDim(i))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, src, i));
    else
      staticSizes[i] = srcType.getDimSize(i);
  }

  return rewriter.create<tensor::InsertSliceOp>(
      loc, src, dest, ValueRange{}, ValueRange(dynamicSizes), ValueRange{},
      rewriter.getDenseI64ArrayAttr(staticOffsets),
      rewriter.getDenseI64ArrayAttr(staticSizes),
      rewriter.getDenseI64ArrayAttr(staticStrides));
}

struct CinmCleanupLinalgPass
    : PassWrapper<CinmCleanupLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CinmCleanupLinalgPass)

  StringRef getArgument() const final { return "cinm-cleanup-linalg"; }
  StringRef getDescription() const final {
    return "Rewrite scf.for tensor yields to insert-slice form for bufferization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    bool changed = false;
    func.walk([&](scf::ForOp forOp) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      bool localChange = false;
      for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        Value yielded = yield.getOperand(idx);
        auto tensorType = dyn_cast<RankedTensorType>(yielded.getType());
        if (!tensorType)
          continue;

        if (auto insert = yielded.getDefiningOp<tensor::InsertSliceOp>()) {
          if (insert.getDest() == iterArg)
            continue;
        }

        rewriter.setInsertionPoint(yield);
        Value inserted = createFullSliceInsert(rewriter, yielded.getLoc(),
                                               yielded, iterArg);
        if (!inserted)
          continue;
        yield.setOperand(idx, inserted);
        localChange = true;
      }
      changed |= localChange;
    });

    if (!changed)
      return;
  }
};

}

std::unique_ptr<mlir::Pass> createCinmCleanupLinalgPass() {
  return std::make_unique<CinmCleanupLinalgPass>();
}

}


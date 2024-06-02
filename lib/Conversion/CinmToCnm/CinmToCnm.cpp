
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include <cinm-mlir/Conversion/CinmPasses.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>

using namespace mlir;
#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/CinmPasses.h.inc>

namespace {

llvm::SmallVector<int64_t, 3>
newShapeFittingWorkgroup(llvm::ArrayRef<int64_t> shape,
                         cnm::WorkgroupType wgTy) {
  auto wgShape = wgTy.getShape();

  auto numWgItems =
      std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
  auto numBufItems =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());
  assert(numBufItems % numWgItems == 0);
  auto remainder = numBufItems / numWgItems;

  llvm::SmallVector<int64_t, 3> newShapeBuf(wgShape);
  newShapeBuf.push_back(remainder);
  return newShapeBuf;
}

AffineMap computeAffineMap(cnm::WorkgroupType wgTy) {
  auto wgShape =
      mlir::getAffineConstantExprs(wgTy.getShape(), wgTy.getContext());

  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(wgShape.size() * 2);
  for (size_t i = 0; i < wgShape.size(); i++) {
    auto index = mlir::getAffineDimExpr(i, wgTy.getContext());
    dimExprs.emplace_back(index.floorDiv(wgShape[i]));
  }
  for (size_t i = 0; i < wgShape.size(); i++) {
    auto index = mlir::getAffineDimExpr(i, wgTy.getContext());
    dimExprs.emplace_back(index % wgShape[i]);
  }
  return AffineMap::get(dimExprs.size(), 0, dimExprs, wgTy.getContext());
}

Value convertInputIntoAlloc(Value inputBuf, Value workGroup,
                            cnm::WorkgroupType wgTy, AffineMap &scatterMap,
                            ImplicitLocOpBuilder &rewriter) {
  // For each input of the reduce, we need to

  auto inputTy = inputBuf.getType().cast<RankedTensorType>();
  cnm::BufferType bufTy = cnm::BufferType::get(
      rewriter.getContext(), inputTy.getShape(), inputTy.getElementType(), 0);

  // 1. Allocate a cinm buffer
  Value alloc = rewriter.create<cnm::AllocOp>(bufTy, workGroup);

  // 2. Reshape original tensor
  auto newShape = newShapeFittingWorkgroup(inputTy.getShape(), wgTy);
  Value shapeReified =
      rewriter.create<arith::ConstantOp>(rewriter.getI64TensorAttr(newShape));
  Value reshaped = rewriter.create<tensor::ReshapeOp>(
      inputTy.cloneWith(newShape, inputTy.getElementType()), inputBuf,
      shapeReified);
  // 2. Scatter into buffer
  scatterMap = computeAffineMap(wgTy);
  rewriter.create<cnm::ScatterOp>(reshaped, alloc, workGroup, scatterMap);

  return alloc;
}

void convertLinalgReduceIntoLaunch(ImplicitLocOpBuilder builder,
                                   linalg::ReduceOp reduction, Value workgroup,
                                   cnm::WorkgroupType wgTy) {

  llvm::SmallVector<Value, 2> launchOperands;
  llvm::SmallVector<AffineMap, 2> affineMaps;
  for (auto input : reduction->getOperands()) {
    launchOperands.emplace_back(convertInputIntoAlloc(
        input, workgroup, wgTy, affineMaps.emplace_back(), builder));
  }

  cnm::LaunchOp launchOp =
      builder.create<cnm::LaunchOp>(workgroup, launchOperands);

  {
    builder.setInsertionPointToStart(&launchOp.getBody().front());
    // arguments are memrefs with same shape as inputs
    auto args = launchOp.getBody().getArguments();
    auto firstOutput = args.begin() + reduction.getNumDpsInits();
    llvm::SmallVector<Value, 2> reduceInpts(args.begin(), firstOutput);
    llvm::SmallVector<Value, 1> reduceInits(firstOutput, args.end());

    auto innerReduce = builder.create<linalg::ReduceOp>(
        // no results bc memref
        TypeRange{}, reduceInpts, reduceInits, reduction.getDimensions());
    IRMapping irMapping;
    reduction.getRegion().cloneInto(&innerReduce.getRegion(), irMapping);
  }
  builder.setInsertionPointAfter(launchOp);
  for (auto [alloc, map, result] :
       llvm::zip(launchOperands, affineMaps, reduction.getResults())) {
    auto res = builder.create<cnm::GatherOp>(
        result.getType(), cnm::GatherTokenType::get(builder.getContext()),
        workgroup, alloc, map);
    result.replaceAllUsesWith(res.getOutput());
  }

  reduction->remove();
}

bool fitsIntoWorkgroup(Value value, cnm::WorkgroupType wgTy) {
  if (auto shaped = value.getType().dyn_cast<RankedTensorType>()) {
    return shaped.getNumElements() % wgTy.getNumElements() == 0;
  }
  return false;
}

struct ConvertLinalgReduceIntoLaunch
    : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    cnm::WorkgroupType wgTy = cnm::WorkgroupType::get(getContext(), {4, 16});
    if (!llvm::all_of(op->getOperands(),
                      [&](Value v) { return fitsIntoWorkgroup(v, wgTy); }))
      return failure();

    Value workgroup = builder.create<cnm::WorkgroupOp>(wgTy);

    convertLinalgReduceIntoLaunch(builder, op, workgroup, wgTy);

    return success();
  }
};

struct ConvertTiledCinmToCnm
    : public ConvertTiledCinmToCnmBase<ConvertTiledCinmToCnm> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertLinalgReduceIntoLaunch>(&getContext());
    ConversionTarget target(getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm::createConvertTiledCinmToCnmPass() {
  return std::make_unique<ConvertTiledCinmToCnm>();
}

void mlir::cinm::registerCinmToCnmPipeline() {
  // todo
}
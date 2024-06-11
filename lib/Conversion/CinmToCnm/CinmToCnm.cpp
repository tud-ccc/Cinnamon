
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include <cinm-mlir/Conversion/CinmPasses.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/InliningUtils.h>
#include <numeric>

using namespace mlir;
#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/CinmPasses.h.inc>

namespace {

struct ReduceToCnmRewriteParams {
  cnm::WorkgroupType wgTy;
  //  llvm::SmallVector<, unsigned int N>
};

LogicalResult
computeShapeOfTensors(llvm::ArrayRef<int64_t> shape, cnm::WorkgroupType wgTy,
                      int64_t maxBlockSize,
                      // if empty then all dims are parallel
                      // otherwise those dims are reductions. They are
                      // used to select the size of the buffer. The rest of
                      // the dimensions are used to create a scattermap
                      llvm::ArrayRef<int64_t> reductionDims,
                      AffineMap &scatterMap, AffineMap &gatherMap,
                      llvm::SmallVectorImpl<int64_t> &shapeOfBuffer) {
  auto wgShape = wgTy.getShape();

  auto numWgItems =
      std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
  auto numBufItems =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());

  llvm::SmallVector<int64_t> parallelDims;
  int64_t numParallelElts = 1;
  int64_t numReductionElts = 1;

  for (size_t i = 0, j = 0; i < shape.size(); i++) {
    if (j >= reductionDims.size() || i != reductionDims[j]) {
      // this is a parallel dim
      parallelDims.push_back(shape[i]);
      numParallelElts *= shape[i];
    } else {
      // this is a reduction dim
      j++;
      shapeOfBuffer.push_back(shape[i]);
      numReductionElts *= shape[i];
    }
  }
  if (numReductionElts > maxBlockSize)
    return failure();

  // Now we support two cases: either
  // 1. tensor has shape of WG
  if (parallelDims == wgShape) {
    scatterMap = AffineMap::getMultiDimIdentityMap(
        wgShape.size() + reductionDims.size(), wgTy.getContext());
    gatherMap = scatterMap;
    return success();
  }

  // or 2. numParallelItems == k * numWgItems
  if (numParallelElts % numWgItems != 0) {
    return failure();
  }

  // Affine map has form the form (with a working group that has m dimensions):
  // if k = 1:
  //     (t, R1,..., Rn) -> (W1,...,Wm,R1,...,Rn)
  // if k * numReductionItems <= maxBlockSize:
  //     (t, R1,..., Rn) -> (W1,...,Wm,ki,R1,...,Rn)
  //    where ki ranges from 0 to k
  // otherwise:
  //     (t, R1,..., Rn) -> (ki,W1,...,Wm,b,R1,...,Rn)
  //    where ki ranges from 0 to k/maxBlockSize,
  //          b ranges from 0 to maxBlockSize.
  llvm::SmallVector<AffineExpr> affineDims;
  affineDims.reserve(wgShape.size() + reductionDims.size());

  SmallVector<int64_t, 6> fullWgShape(wgShape);
  int64_t k = numBufItems / numWgItems;
  if (k != 1) {
    if (k * numReductionElts <= maxBlockSize) {
      fullWgShape.push_back(k);
      // The buffer that will be passed to the CINM launch has this dimension.
      shapeOfBuffer.push_back(k);
    } else {
      // probably the op hasn't been tiled properly
      return failure();
    }
  }

  AffineExpr index = mlir::getAffineDimExpr(0, wgTy.getContext());
  if (parallelDims.size() > 1) {
    if (reductionDims.size() > 0)
      return failure();
    index = 0;
    // then we "flatten" the original tensor expression
    for (int i = 0; i < parallelDims.size() - 1; i++) {
      index = index +
              parallelDims[i] * mlir::getAffineDimExpr(i, wgTy.getContext());
    }
    index = index +
            mlir::getAffineDimExpr(parallelDims.size() - 1, wgTy.getContext());
  }
  int64_t sizeOfTrailing = numParallelElts / fullWgShape[0];
  affineDims.push_back(index.floorDiv(sizeOfTrailing));

  AffineExpr gatherExpr = index * sizeOfTrailing;
  size_t i = 1;

  for (auto dim : llvm::drop_begin(fullWgShape, 1)) {
    index = index % sizeOfTrailing;
    sizeOfTrailing /= dim;
    affineDims.push_back(index.floorDiv(sizeOfTrailing));
    gatherExpr = gatherExpr +
                 mlir::getAffineDimExpr(i, wgTy.getContext()) * sizeOfTrailing;
    i++;
  }
  // reduction dims are identity
  for (unsigned i = 0; i < reductionDims.size(); i++) {
    affineDims.push_back(
        mlir::getAffineDimExpr(i + parallelDims.size(), wgTy.getContext()));
  }
  scatterMap = AffineMap::get(parallelDims.size() + reductionDims.size(), 0,
                              affineDims, wgTy.getContext());

  llvm::SmallVector<AffineExpr> gatherMapResults;
  gatherMapResults.reserve(1 + reductionDims.size());

  gatherMapResults.push_back(gatherExpr);
  for (unsigned i = 1; i < reductionDims.size(); i++) {
    gatherMapResults.push_back(mlir::getAffineDimExpr(i, wgTy.getContext()));
  }

  gatherMap =
      AffineMap::get(affineDims.size(), 0, gatherMapResults, wgTy.getContext());
  return success();
}

LogicalResult convertInputIntoAlloc(Value inputBuf, Value workGroup,
                                    cnm::WorkgroupType wgTy,
                                    int64_t maxBlockSize,
                                    ArrayRef<int64_t> reduceDims,
                                    AffineMap &gatherMap, Value &result,
                                    ImplicitLocOpBuilder &rewriter) {
  // For each input of the reduce, we need to

  AffineMap scatterMap;
  auto inputTy = inputBuf.getType().cast<RankedTensorType>();
  llvm::SmallVector<int64_t, 1> shapeOfBuffer;
  if (computeShapeOfTensors(inputTy.getShape(), wgTy, maxBlockSize, reduceDims,
                            scatterMap, gatherMap, shapeOfBuffer)
          .failed())
    return failure();

  // Allocate a cinm buffer
  cnm::BufferType bufTy =
      cnm::BufferType::get(rewriter.getContext(), shapeOfBuffer,
                           inputTy.getElementType(), wgTy.getShape(),
                           0); // todo level is hardcoded

  Value alloc = rewriter.create<cnm::AllocOp>(bufTy, workGroup);

  // Scatter into buffer
  rewriter.create<cnm::ScatterOp>(inputBuf, alloc, workGroup, scatterMap);
  result = alloc;

  return success();
}

LogicalResult convertCinmToCnm(
    ImplicitLocOpBuilder builder, Operation *operation,
    TypedValue<cnm::WorkgroupType> workgroup, int64_t maxDpuMemory,
    ArrayRef<int64_t> reductionDimensionsSorted, ValueRange operands,
    ValueRange outputInitializers, ValueRange results,
    llvm::SmallVectorImpl<Value> &resultValues,
    function_ref<void(ImplicitLocOpBuilder &, ValueRange, ValueRange)>
        createCnmLaunchBlock) {

  auto wgTy = workgroup.getType();

  llvm::SmallVector<Value, 3> launchInputs;
  llvm::SmallVector<Value, 3> launchOutputs;
  llvm::SmallVector<AffineMap, 3> gatherMaps;
  llvm::SmallVector<Type, 3> mappedArgTypes;

  int maxBlockSize = maxDpuMemory; // simplification

  builder.setInsertionPointAfter(operation);

  for (auto input : operands) {
    if (convertInputIntoAlloc(
            input, workgroup, wgTy, maxBlockSize, reductionDimensionsSorted,
            gatherMaps.emplace_back(), launchInputs.emplace_back(), builder)
            .failed()) {
      return failure();
    }
  }
  for (auto output : outputInitializers) {
    if (convertInputIntoAlloc(output, workgroup, wgTy, maxBlockSize, {},
                              gatherMaps.emplace_back(),
                              launchOutputs.emplace_back(), builder)
            .failed()) {
      return failure();
    }
  }

  cnm::LaunchOp launchOp =
      builder.create<cnm::LaunchOp>(workgroup, launchInputs, launchOutputs);

  {
    auto &launchBlock = launchOp.getBody().emplaceBlock();
    // arguments are memrefs with same shape as inputs
    for (auto input : launchOp.getParams()) {
      if (auto inputTy = input.getType().cast<cnm::BufferType>()) {
        auto mappedTy =
            MemRefType::get(inputTy.getShape(), inputTy.getElementType());
        launchBlock.addArgument(mappedTy, input.getLoc());
      } else {
        launchBlock.addArgument(input.getType(), input.getLoc());
      }
    }
    auto args = launchBlock.getArguments();
    auto firstOutput = args.begin() + operands.size();
    llvm::SmallVector<Value, 2> reduceInpts(args.begin(), firstOutput);
    llvm::SmallVector<Value, 1> reduceInits(firstOutput, args.end());

    builder.setInsertionPointToStart(&launchBlock);
    createCnmLaunchBlock(builder, reduceInpts, reduceInits);
    builder.create<cnm::TerminatorOp>();
  }
  builder.setInsertionPointAfter(launchOp);

  // gather the results (only the out buffers)

  for (auto [i, result, alloc] : llvm::enumerate(results, launchOutputs)) {
    auto map = gatherMaps[launchInputs.size() + i];
    auto res = builder.create<cnm::GatherOp>(
        result.getType(), cnm::GatherTokenType::get(builder.getContext()),
        alloc, workgroup, map);
    resultValues.push_back(res.getOutput());
  }
  return success();
}

// todo change that into
struct ConvertLinalgReduceIntoLaunch
    : public OpConversionPattern<linalg::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto computeOp = mlir::cinm::getEnclosingComputeBlock(op);

    cnm::WorkgroupOp workgroup =
        builder.create<cnm::WorkgroupOp>(computeOp.getCnmWorkgroupType());

    llvm::SmallVector<Value, 1> newResults;
    if (convertCinmToCnm(
            builder, op, workgroup.getResult(), computeOp.getMaxDpuBufferSize(),
            op.getDimensions(), adaptor.getInputs(), adaptor.getInits(),
            op->getResults(), newResults,
            [&](ImplicitLocOpBuilder &builder, ValueRange memrefInputs,
                ValueRange memrefOutputs) {
              // Here we are copying the original reduce into the launch,
              // except it's now operating on memrefs provided by cinm.
              // This can be lowered to affine or whatever afterwards.
              auto innerReduce = builder.create<linalg::ReduceOp>(
                  // no results bc memref
                  TypeRange{}, memrefInputs, memrefOutputs,
                  // todo we are hardcoding the dimensions
                  // This is because we flatten everything. This does not
                  // support custom reduction dimensions.
                  ArrayRef<int64_t>{0});

              IRMapping irMapping;
              op.getRegion().cloneInto(&innerReduce.getRegion(), irMapping);
            })
            .failed())
      return failure();
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

template <typename CinmOp, typename LinalgOp>
struct ConvertElementWiseToCnm : public OpConversionPattern<CinmOp> {
  using OpConversionPattern<CinmOp>::OpConversionPattern;
  ConvertElementWiseToCnm<CinmOp>(MLIRContext *ctx)
      : mlir::OpConversionPattern<CinmOp>(ctx) {
    this->setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(CinmOp op, OpConversionPattern<CinmOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    cinm::ComputeOp computeBlock = mlir::cinm::getEnclosingComputeBlock(op);
    cnm::WorkgroupOp workgroup =
        builder.create<cnm::WorkgroupOp>(computeBlock.getCnmWorkgroupType());
    auto outputInit = builder.create<arith::ConstantOp>(
        op.getResult().getType(),
        builder.getZeroAttr(op.getResult().getType()));

    llvm::SmallVector<Value, 1> newResults;
    if (convertCinmToCnm(builder, op, workgroup.getResult(),
                         computeBlock.getMaxDpuBufferSize(), {},
                         adaptor.getOperands(), ValueRange{outputInit},
                         op->getResults(), newResults,
                         [&](ImplicitLocOpBuilder &builder, ValueRange inputs,
                             ValueRange outputs) {
                           builder.create<LinalgOp>(inputs, outputs);
                         })
            .failed()) {
      return failure();
    }

    rewriter.replaceOp(op, newResults);

    return success();
  }
};

struct DeleteCinmCompute : public OpConversionPattern<cinm::ComputeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::ComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.setInsertionPointAfter(op);
    IRMapping mapper;
    for (auto &toCopy : op.getBody().front().without_terminator()) {
      rewriter.clone(toCopy, mapper);
    }
    auto term = op->getBlock()->getTerminator();
    for (auto [result, termOperand] :
         llvm::zip(op->getResults(), term->getOperands())) {
      rewriter.replaceAllUsesWith(result, mapper.lookup(termOperand));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void populateCinmRewritePatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  patterns.insert<ConvertLinalgReduceIntoLaunch>(ctx);
  // elementwise
  patterns.insert<ConvertElementWiseToCnm<cinm::AddOp, linalg::AddOp>>(ctx);
  patterns.insert<ConvertElementWiseToCnm<cinm::MulOp, linalg::MulOp>>(ctx);
  patterns.insert<ConvertElementWiseToCnm<cinm::SubOp, linalg::SubOp>>(ctx);
  patterns.insert<ConvertElementWiseToCnm<cinm::DivOp, linalg::DivOp>>(ctx);
}
struct ConvertTiledCinmToCnm
    : public ConvertTiledCinmToCnmBase<ConvertTiledCinmToCnm> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCinmRewritePatterns(patterns, &getContext());
    ConversionTarget target(getContext());

    //  target.addIllegalDialect<linalg::ReduceOp>();
    //    target.addDynamicallyLegalOp<linalg::ReduceOp>(
    //      [](linalg::ReduceOp op) -> bool {
    //          return
    //          !WorkGroupMakerStrategy::determineWorkGroupTypeForRewrite(op);
    //      });
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addIllegalDialect<cinm::CinmDialect>();
    target.addLegalDialect<cnm::CnmDialect>();
    target.addLegalOp<cnm::LaunchOp>();
    target.addLegalOp<cinm::ComputeOp>();
    target.addLegalOp<cinm::YieldOp>();
    target.markOpRecursivelyLegal<cnm::LaunchOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();

    // in a second phase we remove cinm compute blocks

    target.addIllegalOp<cinm::ComputeOp>();
    target.addIllegalOp<cinm::YieldOp>();
    RewritePatternSet patterns2 = RewritePatternSet(&getContext());
    patterns2.insert<DeleteCinmCompute>(&getContext());
    if (applyFullConversion(getOperation(), target, std::move(patterns2))
            .failed())
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

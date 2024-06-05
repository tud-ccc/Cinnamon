
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
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
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
#include <optional>

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
  if (numBufItems % numWgItems != 0)
    return failure();

  llvm::SmallVector<int64_t> parallelDims;
  int64_t numParallelElts = 1;

  for (size_t i = 0, j = 0; i < shape.size(); i++) {
    if (j >= reductionDims.size() || i != reductionDims[j]) {
      // this is a parallel dim
      parallelDims.push_back(shape[i]);
      numParallelElts *= shape[i];
    } else {
      // this is a reduction dim
      j++;
      shapeOfBuffer.push_back(shape[i]);
    }
  }
  // Now we support two cases: either
  // 1. tensor has shape of WG
  if (parallelDims == wgShape) {
    scatterMap = AffineMap::getMultiDimIdentityMap(
        wgShape.size() + reductionDims.size(), wgTy.getContext());
    gatherMap = scatterMap;
    return success();
  }
  // or 2. tensor is flat, and numBufItems == numWgItems
  //   (could be extended to handle numBufItems = k * numWgItems)
  if (parallelDims.size() != 1 || numParallelElts != numWgItems)
    return failure();

  // Affine map has form (t, R1,..., Rn) -> (W1,...,Wm,R1,...,Rn)
  // Where the working group has m dimensions
  llvm::SmallVector<AffineExpr> affineDims;
  affineDims.reserve(wgShape.size() + reductionDims.size());

  AffineExpr index = mlir::getAffineDimExpr(0, wgTy.getContext());
  int64_t sizeOfTrailing = numWgItems / wgShape[0];
  affineDims.push_back(index.floorDiv(sizeOfTrailing));

  AffineExpr gatherExpr = index * sizeOfTrailing;
  size_t i = 1;

  for (auto dim : wgShape.drop_front(1)) {
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
  scatterMap = AffineMap::get(1 + reductionDims.size(), 0, affineDims,
                              wgTy.getContext());

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
                                    ArrayRef<int64_t> reduceDims,
                                    AffineMap &gatherMap, Value &result,
                                    ImplicitLocOpBuilder &rewriter) {
  // For each input of the reduce, we need to

  AffineMap scatterMap;
  auto inputTy = inputBuf.getType().cast<RankedTensorType>();
  llvm::SmallVector<int64_t, 1> shapeOfBuffer;
  if (computeShapeOfTensors(inputTy.getShape(), wgTy, reduceDims, scatterMap,
                            gatherMap, shapeOfBuffer)
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

LogicalResult convertLinalgReduceIntoLaunch(
    ImplicitLocOpBuilder builder, linalg::ReduceOp reduction,
    linalg::ReduceOp::Adaptor adaptor, Value workgroup,
    llvm::SmallVectorImpl<Value> &resultValues,
    ReduceToCnmRewriteParams parms) {
  cnm::WorkgroupType wgTy = parms.wgTy;

  llvm::SmallVector<Value, 3> launchOperands;
  llvm::SmallVector<AffineMap, 3> gatherMaps;
  llvm::SmallVector<Type, 3> mappedArgTypes;

  builder.setInsertionPointAfter(reduction);
  int i = 0;
  for (auto input : adaptor.getOperands()) {
    auto reduceDims = reduction.getDimensions();
    if (i >= reduction.getNumDpsInputs()) {
      // this is an output
      reduceDims = {};
    }
    if (convertInputIntoAlloc(input, workgroup, wgTy, reduceDims,
                              gatherMaps.emplace_back(),
                              launchOperands.emplace_back(), builder)
            .failed())
      return failure();
    i++;
  }

  cnm::LaunchOp launchOp = builder.create<cnm::LaunchOp>(
      workgroup, ValueRange{launchOperands}.take_front(2),
      ValueRange{launchOperands}.take_back());

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
    auto firstOutput = args.begin() + reduction.getNumDpsInputs();
    llvm::SmallVector<Value, 2> reduceInpts(args.begin(), firstOutput);
    llvm::SmallVector<Value, 1> reduceInits(firstOutput, args.end());

    // Here we are copying the original reduce into the launch,
    // except it's now operating on memrefs provided by cinm.
    // This can be lowered to affine or whatever afterwards.
    builder.setInsertionPointToStart(&launchBlock);
    auto innerReduce = builder.create<linalg::ReduceOp>(
        // no results bc memref
        TypeRange{}, reduceInpts, reduceInits,
        // todo we are hardcoding the dimensions
        // This is because we flatten everything. This does not support
        // custom reduction dimensions.
        ArrayRef<int64_t>{0});

    IRMapping irMapping;
    reduction.getRegion().cloneInto(&innerReduce.getRegion(), irMapping);
    builder.create<cnm::TerminatorOp>();
  }
  builder.setInsertionPointAfter(launchOp);

  // gather the results (only the out buffers)

  for (size_t i = 0; i < reduction->getNumResults(); i++) {
    auto result = reduction.getResult(i);
    auto alloc = launchOperands[reduction.getNumDpsInputs() + i];
    auto map = gatherMaps[reduction.getNumDpsInputs() + i];
    auto res = builder.create<cnm::GatherOp>(
        result.getType(), cnm::GatherTokenType::get(builder.getContext()),
        alloc, workgroup, map);
    resultValues.push_back(res.getOutput());
  }
  return success();
}

bool workgroupFitsParallelDims(Type ty, cnm::WorkgroupType wgTy,
                               ArrayRef<int64_t> reductionDims) {
  // reductionDims is sorted
  if (auto tensorTy = ty.dyn_cast<RankedTensorType>()) {
    auto shape = tensorTy.getShape();
    int64_t numNonReduceElts = 1;
    for (size_t i = 0, j = 0; i < shape.size(); i++) {
      if (j >= reductionDims.size() || i != reductionDims[j]) {
        numNonReduceElts *= shape[i];
      } else {
        j++;
      }
    }
    return numNonReduceElts % wgTy.getNumElements() == 0;
  }
  return false;
}

struct WorkGroupMakerStrategy {
  static std::optional<cnm::WorkgroupType>
  determineWorkGroupTypeForRewrite(linalg::ReduceOp op);
};

// this strategy just tries to use a static workgroup shape
template <unsigned... Shape> struct StaticWorkGroup {
  static std::optional<ReduceToCnmRewriteParams>
  determineWorkGroupTypeForRewrite(linalg::ReduceOp op) {
    cnm::WorkgroupType wgTy =
        cnm::WorkgroupType::get(op.getContext(), {Shape...});

    // output ops need to be big enough to be dispatchable on the WG
    if (llvm::any_of(op.getDpsInits(), [&](Value v) {
          return !workgroupFitsParallelDims(v.getType(), wgTy, {});
        }))
      return std::nullopt;

    // in particular the input operands WITHOUT the reduction dimensions should
    // fit
    if (llvm::any_of(op.getDpsInputs(), [&](Value v) {
          return !workgroupFitsParallelDims(v.getType(), wgTy,
                                            op.getDimensions());
        }))
      return std::nullopt;
    return ReduceToCnmRewriteParams{.wgTy = wgTy};
  }
};

template <typename WGStrat>
struct ConvertLinalgReduceIntoLaunch
    : public OpConversionPattern<linalg::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto wgTyOpt = WGStrat::determineWorkGroupTypeForRewrite(op);
    if (!wgTyOpt)
      return failure();
    ReduceToCnmRewriteParams parms = *wgTyOpt;
    Value workgroup = builder.create<cnm::WorkgroupOp>(parms.wgTy);

    llvm::SmallVector<Value, 1> newResults;
    if (convertLinalgReduceIntoLaunch(builder, op, adaptor, workgroup,
                                      newResults, parms)
            .failed())
      return failure();
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

struct ConvertTiledCinmToCnm
    : public ConvertTiledCinmToCnmBase<ConvertTiledCinmToCnm> {

  using WorkGroupMakerStrategy = StaticWorkGroup<4, 8, 2>;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertLinalgReduceIntoLaunch<WorkGroupMakerStrategy>>(
        &getContext());
    ConversionTarget target(getContext());

    //  target.addIllegalDialect<linalg::ReduceOp>();
    //    target.addDynamicallyLegalOp<linalg::ReduceOp>(
    //      [](linalg::ReduceOp op) -> bool {
    //          return
    //          !WorkGroupMakerStrategy::determineWorkGroupTypeForRewrite(op);
    //      });
    //    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<cnm::CnmDialect>();
    target.addLegalOp<cnm::LaunchOp>();
    target.markOpRecursivelyLegal<cnm::LaunchOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
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

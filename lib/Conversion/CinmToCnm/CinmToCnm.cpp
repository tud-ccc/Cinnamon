
#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Utils/CinmUtils.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
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

using namespace mlir;
#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/CinmPasses.h.inc>

namespace {

// Turn an index in the index space of the given shape into a linear index.
AffineExpr linearizeIndices(MLIRContext *ctx, ArrayRef<int64_t> shape) {

  AffineExpr index = getAffineConstantExpr(0, ctx);
  int64_t dimIndex = shape.size() - 1;
  int64_t trailing = 1;
  for (auto it = shape.rbegin(); it != shape.rend(); it++) {
    auto dim = *it;
    if (dim != 1) {
      // otherwise the only index in this space is zero, so we simplify it.
      index = trailing * getAffineDimExpr(dimIndex, ctx) + index;
      trailing *= dim;
    }
    dimIndex--;
  }
  return index;
}

// inflate a linear index into the given shape
void structureIndex(AffineExpr index, ArrayRef<int64_t> shape,
                    SmallVectorImpl<AffineExpr> &map) {

  int64_t sizeOfTrailing = computeProduct(shape) / shape[0];
  map.push_back(index.floorDiv(sizeOfTrailing));

  AffineExpr gatherExpr = index * sizeOfTrailing;
  size_t i = 1;

  for (auto dim : llvm::drop_begin(shape, 1)) {
    index = index % sizeOfTrailing;
    sizeOfTrailing /= dim;
    map.push_back(index.floorDiv(sizeOfTrailing));
    gatherExpr = gatherExpr +
                 mlir::getAffineDimExpr(i, index.getContext()) * sizeOfTrailing;
    i++;
  }
}

LogicalResult
computeShapeOfTensors(Location loc, llvm::ArrayRef<int64_t> shape,
                      cnm::WorkgroupType wgTy, int64_t maxBlockSize,
                      // if empty then all dims are parallel
                      // otherwise those dims are reductions. They are
                      // used to select the size of the buffer. The rest of
                      // the dimensions are used to create a scattermap
                      llvm::ArrayRef<int64_t> reductionDims,
                      AffineMap &scatterMap,
                      llvm::SmallVectorImpl<int64_t> &shapeOfBuffer,

                      std::optional<llvm::SmallVector<int64_t>> &reshapeInputTo,
                      bool scatterScalar) {
  auto wgShape = wgTy.getShape();

  auto numWgItems =
      std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
  auto numBufItems =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());

  llvm::SmallVector<int64_t> parallelDims;
  int64_t numParallelElts = 1;
  int64_t numReductionElts = 1;

  for (size_t i = 0, j = 0; i < shape.size(); i++) {
    if (j >= reductionDims.size() ||
        i != static_cast<size_t>(reductionDims[j])) {
      // this is a parallel dim
      parallelDims.push_back(shape[i]);
      numParallelElts *= shape[i];
      if (numReductionElts > 1) {
        // This means the reduction dims are not all at the end
        // Todo revisit, needs a transpose
        return failure();
      }
    } else {
      // this is a reduction dim
      j++;
      shapeOfBuffer.push_back(shape[i]);
      numReductionElts *= shape[i];
    }
  }

  if (numReductionElts > maxBlockSize) {
    emitError(loc, "can't compute shape of tensors: numReductionElts (" +
                       std::to_string(numReductionElts) + ") > maxBlockSize (" +
                       std::to_string(maxBlockSize) + ")");
    return failure();
  }

  // Now we support 3 cases: either
  // 0. scattering a single element
  if (scatterScalar) {
    const size_t numDims = wgShape.size() + reductionDims.size();
    scatterMap = AffineMap::get(
        numDims, 0,
        SmallVector<AffineExpr>(1, getAffineConstantExpr(0, wgTy.getContext())),
        wgTy.getContext());
    return success();
  }

  // 1. tensor has shape of WG
  if (parallelDims == wgShape) {
    scatterMap = AffineMap::getMultiDimIdentityMap(
        wgShape.size() + reductionDims.size(), wgTy.getContext());
    return success();
  }

  // or 2. numParallelItems == k * numWgItems
  if (numParallelElts % numWgItems != 0) {
    return emitError(loc, "can't compute shape of tensors: numParallelElts (" +
                              std::to_string(numParallelElts) +
                              ") % numWgItems (" + std::to_string(numWgItems) +
                              ") != 0");
  }

  // Say we have p parallel dims, m working group dimensions, and r reduction
  // dimensions. Affine map has the form

  // In case we have 0 reduction dimensions:
  // Affine map has the form
  //   F: (W1, ..., Wm) -> (T1, ..., Tp)
  // For all w=(w1, ..., wm), F(w) must be in range,
  // that is to say, for each dim i, 0 <= F(w)_i < |T_i|,
  // and

  // For example consider
  //    gemm: (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
  //    WG: <1x128>
  //    bufsize: 32
  //
  // The first tensor is <1x32>. Num par elements is 1. Broadcast:
  //    (w0, w1) -> (0, 0)
  // The second tensor is <32x128>. Num par elements is 128 = |WG|.
  //    (w0, w1) -> (0, lin(w0, w1))
  // Here we see that the elements are not contiguous though.
  //    Transpose: <32x128> -> <128x32>
  //    (w0, w1) -> (lin(w0, w1), 0)
  // The output tensor is <1x128>. It matches WG shape:
  //    (w0, w1) -> (w0, w1)

  // IMPLNOTE: the transpose is not done in this routine. It must be
  //   done in the caller. This rountine will fail if the reduction
  //   dims are not at the end.

  // Another example. Elementwise operators have no reduction dims.
  //    add: tensor<16384xi32>
  //    WG: <8x128>
  //    bufsize: 16
  //
  // Both input tensors and output have same shape.
  // Num par elements is 16384. Break this down into
  // 16384/16=1024 buffers of 16 elements and expand shape:
  //   T': <16384> -> <1024x16>
  // Then scatter map is
  //   (w0, w1) -> (lin(w0, w1), 0)

  // What if I want to do this without expand_shape?
  // (w0, w1) -> (lin(w0, w1) * 16)

  /*
  Summary:
  We need to support the following cases:
  - Tensor has exactly 1 parallel element. Then broadcast. (only if it is an
  input)
  - Tensor is flat and needs to be chunked.
  - Tensor has parallel elts = |WG| but chunks are not contiguous.
    Need a linalg.transpose.

  */

  // if k = 1:
  //     (t, R1,..., Rn) -> (W1,...,Wm,R1,...,Rn)
  // if k * numReductionItems <= maxBlockSize:
  //     (t, R1,..., Rn) -> (W1,...,Wm,ki,R1,...,Rn)
  //    where ki ranges from 0 to k
  //

  int64_t k = numBufItems / numWgItems;
  if (k != 1) {
    if (k * numReductionElts <= maxBlockSize) {
      // In this branch we handle the case where there are no reduction
      // dimensions, in that case we do some parallel work on the DPU, and
      // therefore push these extra parallel elts into the buffer.
      shapeOfBuffer.push_back(k);

      // expand dimension
      int trailing = 1;
      int numFlattened = 0;
      for (auto it = parallelDims.rbegin(); it != parallelDims.rend(); it++) {
        trailing *= *it;
        numFlattened++;
        if (trailing >= k && trailing % k == 0) {
          break;
        }
      }
      SmallVector<int64_t> newShape;
      for (auto [i, dim] : llvm::enumerate(parallelDims)) {
        if (i < parallelDims.size() - numFlattened) {
          // not flattened
          newShape.push_back(dim);
        } else {
          newShape.push_back(trailing / k);
          newShape.push_back(k);
          break;
        }
      }
      parallelDims = newShape; // our parallel dims have changed
      parallelDims.pop_back();

      for (auto dim : reductionDims)
        newShape.push_back(dim);

      reshapeInputTo = std::make_optional(std::move(newShape));

    } else {
      // probably the op hasn't been tiled properly
      emitError(loc, "can't compute shape of tensors: k (" + std::to_string(k) +
                         ") * numReductionElts (" +
                         std::to_string(numReductionElts) +
                         ") > "
                         "maxBlockSize (" +
                         std::to_string(maxBlockSize) + ")");
      return failure();
    }
  }

  AffineExpr index = linearizeIndices(wgTy.getContext(), wgShape);

  llvm::SmallVector<AffineExpr> scatterResults;
  scatterResults.reserve(parallelDims.size());
  structureIndex(index, parallelDims, scatterResults);

  scatterMap =
      AffineMap::get(wgShape.size(), 0, scatterResults, wgTy.getContext());
  return success();
}

LogicalResult convertInputIntoAlloc(Value &inputBuf, Value workGroup,
                                    cnm::WorkgroupType wgTy,
                                    int64_t maxBlockSizeBytes,
                                    ArrayRef<int64_t> reduceDims,
                                    AffineMap &scatterMap, Value &result,
                                    ImplicitLocOpBuilder &rewriter) {
  // For each input of the reduce, we need to

  // convert single element to tensor<numTasklets x leafSize x ElementTy>
  bool scatterScalar = false;
  if (!llvm::isa<ShapedType>(inputBuf.getType())) {
    scatterScalar = true;
    inputBuf = rewriter.create<tensor::FromElementsOp>(
        RankedTensorType::get({wgTy.getShape()[2]}, inputBuf.getType()),
        SmallVector<Value>(wgTy.getShape()[2], inputBuf));
  }

  auto inputType = cast<ShapedType>(inputBuf.getType());

  llvm::SmallVector<int64_t, 1> shapeOfBuffer;
  std::optional<SmallVector<int64_t>> reshapeInto;
  auto maxBlockSizeItems =
      maxBlockSizeBytes * 8 / inputType.getElementTypeBitWidth();
  if (computeShapeOfTensors(inputBuf.getLoc(), inputType.getShape(), wgTy,
                            maxBlockSizeItems, reduceDims, scatterMap,
                            shapeOfBuffer, reshapeInto, scatterScalar)
          .failed())
    return failure();

  if (reshapeInto) {
    inputBuf =
        cinm::reshapeStatic(rewriter, rewriter.getLoc(), inputBuf,
                            cast<ShapedType>(inputType), *reshapeInto);
  }

  // Allocate a cinm buffer
  cnm::BufferType bufTy = cnm::BufferType::get(
      shapeOfBuffer, inputType.getElementType(), wgTy.getShape(),
      0); // todo level is hardcoded

  Value alloc = rewriter.create<cnm::AllocOp>(bufTy, workGroup);

  // Scatter into buffer
  rewriter.create<cnm::ScatterOp>(inputBuf, alloc, workGroup, scatterMap);
  result = alloc;

  return success();
}

cnm::LaunchOp createLaunchOp(
    ImplicitLocOpBuilder &builder, Value workgroup, ValueRange inputs,
    ValueRange outputs,
    function_ref<void(ImplicitLocOpBuilder &, ValueRange, ValueRange)>
        createCnmLaunchBlock) {

  cnm::LaunchOp launchOp =
      cnm::LaunchOp::create(builder, workgroup, inputs, outputs);

  {
    auto &launchBlock = launchOp.getBody().emplaceBlock();
    // arguments are memrefs with same shape as inputs
    for (auto input : launchOp.getParams()) {
      if (auto inputTy = dyn_cast<cnm::BufferType>(input.getType())) {
        auto mappedTy =
            MemRefType::get(inputTy.getShape(), inputTy.getElementType());
        launchBlock.addArgument(mappedTy, input.getLoc());
      } else {
        launchBlock.addArgument(input.getType(), input.getLoc());
      }
    }

    auto args = launchBlock.getArguments();
    auto firstOutput = args.begin() + inputs.size();
    llvm::SmallVector<Value, 2> reduceInpts(args.begin(), firstOutput);
    llvm::SmallVector<Value, 1> reduceInits(firstOutput, args.end());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&launchBlock);
    createCnmLaunchBlock(builder, reduceInpts, reduceInits);
    cnm::TerminatorOp::create(builder);
  }
  return launchOp;
}

LogicalResult convertCinmToCnm(
    ImplicitLocOpBuilder builder, Operation *operation,
    TypedValue<cnm::WorkgroupType> workgroup, cinm::ComputeOp computeOp,
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

  auto tilingParms = cinm::TilingParameters::fromComputeBlock(computeOp);
  int maxBlockSizeBytes = tilingParms.bufferSizeOfLeaf() / operands.size();

  builder.setInsertionPointAfter(operation);

  for (auto input : operands) {
    if (convertInputIntoAlloc(input, workgroup, wgTy, maxBlockSizeBytes,
                              reductionDimensionsSorted,
                              gatherMaps.emplace_back(),
                              launchInputs.emplace_back(), builder)
            .failed()) {
      return failure();
    }
  }

  // output values, may have been reshaped
  llvm::SmallVector<Value, 1> reshapedOutputs;
  for (auto output : outputInitializers) {
    if (convertInputIntoAlloc(output, workgroup, wgTy, maxBlockSizeBytes, {},
                              gatherMaps.emplace_back(),
                              launchOutputs.emplace_back(), builder)
            .failed()) {
      return failure();
    }
    reshapedOutputs.push_back(output);
  }

  createLaunchOp(builder, workgroup, launchInputs, launchOutputs,
                 createCnmLaunchBlock);

  // gather the results (only the out buffers)

  for (auto [i, reshaped, result, alloc] :
       llvm::enumerate(reshapedOutputs, results, launchOutputs)) {
    auto map = gatherMaps[launchInputs.size() + i];
    auto outBuf =
        tensor::EmptyOp::create(builder, reshaped.getType(), ValueRange{});
    auto res = cnm::GatherOp::create(builder, alloc, workgroup, map, outBuf);
    auto shapedBack = cinm::reshapeStatic(
        builder, builder.getLoc(),
        cast<TypedValue<ShapedType>>(res.getOutput()),
        cast<ShapedType>(result.getType()).getShape());

    resultValues.push_back(shapedBack);
  }

  cnm::FreeWorkgroupOp::create(builder, workgroup);
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
        cnm::WorkgroupOp::create(builder, computeOp.getCnmWorkgroupType());

    llvm::SmallVector<Value, 1> newResults;
    if (convertCinmToCnm(
            builder, op, workgroup.getResult(), computeOp, op.getDimensions(),
            adaptor.getInputs(), adaptor.getInits(), op->getResults(),
            newResults,
            [&](ImplicitLocOpBuilder &builder, ValueRange memrefInputs,
                ValueRange memrefOutputs) {
              // Here we are copying the original reduce into the launch,
              // except it's now operating on memrefs provided by cinm.
              // This can be lowered to affine or whatever afterwards.
              auto innerReduce = linalg::ReduceOp::create(
                  builder,
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

struct ConvertElementwiseOpToCnm : OpConversionPattern<cinm::ElementwiseOp> {
  explicit ConvertElementwiseOpToCnm(MLIRContext *ctx)
      : OpConversionPattern(ctx) {
    this->setHasBoundedRewriteRecursion();
  }

  static std::optional<linalg::ElementwiseKind>
  toLinalgUnaryFn(cinm::ElementwiseKind kind) {
    switch (kind) {
    case cinm::ElementwiseKind::Neg:
      return linalg::ElementwiseKind::negf;
    case cinm::ElementwiseKind::Abs:
      return linalg::ElementwiseKind::abs;
    case cinm::ElementwiseKind::Ceil:
      return linalg::ElementwiseKind::ceil;
    case cinm::ElementwiseKind::Erf:
      return linalg::ElementwiseKind::erf;
    case cinm::ElementwiseKind::Exp:
      return linalg::ElementwiseKind::exp;
    case cinm::ElementwiseKind::Floor:
      return linalg::ElementwiseKind::floor;
    case cinm::ElementwiseKind::Log:
      return linalg::ElementwiseKind::log;
    case cinm::ElementwiseKind::Reciprocal:
      return linalg::ElementwiseKind::reciprocal;
    case cinm::ElementwiseKind::Round:
      return linalg::ElementwiseKind::round;
    case cinm::ElementwiseKind::Rsqrt:
      return linalg::ElementwiseKind::rsqrt;
    case cinm::ElementwiseKind::Sqrt:
      return linalg::ElementwiseKind::sqrt;
    case cinm::ElementwiseKind::Square:
      return linalg::ElementwiseKind::square;
    case cinm::ElementwiseKind::Tanh:
      return linalg::ElementwiseKind::tanh;
    default:
      return std::nullopt;
    }
  }

  LogicalResult
  matchAndRewrite(cinm::ElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    cinm::ComputeOp computeBlock = getEnclosingComputeBlock(op);
    auto workgroup =
        cnm::WorkgroupOp::create(builder, computeBlock.getCnmWorkgroupType());

    // Initialize output for linalg.generic
    auto outputInit = tensor::EmptyOp::create(builder, op.getResult().getType(),
                                              ValueRange{});
    const Type elementType =
        dyn_cast_or_null<ShapedType>(op.getLhs().getType()).getElementType();
    bool isScalarOp = op.getRhs() && op.getRhs().getType() == elementType;

    SmallVector<Value, 1> newResults;
    const auto conversionResult = convertCinmToCnm(
        builder, op, workgroup.getResult(), computeBlock, {},
        adaptor.getOperands(), ValueRange{outputInit}, op->getResults(),
        newResults,
        [&](ImplicitLocOpBuilder &builder, ValueRange inputs,
            ValueRange outputs) {
          SmallVector<AffineMap> affineMaps;
          for (const auto &i : inputs) {
            MemRefType t = cast<MemRefType>(i.getType());
            affineMaps.push_back(AffineMap::getMultiDimIdentityMap(
                t.getRank(), op.getContext()));

            if (isScalarOp) {
              // for scalar ops only the first parameter is
              // passed to the linalg::generic op
              break;
            }
          }

          affineMaps.push_back(AffineMap::getMultiDimIdentityMap(
              cast<MemRefType>(outputs[0u].getType()).getRank(),
              op.getContext()));

          SmallVector<utils::IteratorType> iteratorTypes(
              cast<MemRefType>(inputs[0u].getType()).getRank(),
              utils::IteratorType::parallel);

          if (auto fn = toLinalgUnaryFn(op.getKind())) {
            linalg::ElementwiseOp::create(
                builder, ValueRange(inputs), ValueRange(outputs),
                builder.getAttr<linalg::ElementwiseKindAttr>(*fn),
                builder.getAffineMapArrayAttr(affineMaps));
          } else {
            linalg::GenericOp::create(
                builder, isScalarOp ? inputs.drop_back() : inputs, outputs,
                affineMaps, iteratorTypes,
                [&](OpBuilder &builder, Location loc, ValueRange args) {
                  Value lhs = args[0u];
                  Value rhs = op.getRhs() ? (isScalarOp ? inputs[1u] : args[1u])
                                          : Value();
                  if (isScalarOp) {
                    if (const auto memrefType =
                            dyn_cast<MemRefType>(rhs.getType())) {
                      const Value zero =
                          arith::ConstantIndexOp::create(builder, loc, 0);
                      rhs = memref::LoadOp::create(
                          builder, loc, rhs,
                          SmallVector<Value>(memrefType.getRank(), zero));
                    }
                  }

                  Value result;
                  bool isFloatOp = dyn_cast<FloatType>(elementType) != nullptr;
                  switch (op.getKind()) {
                  case cinm::ElementwiseKind::Add:
                    result = isFloatOp
                                 ? arith::AddFOp::create(builder, loc, lhs, rhs)
                                       .getResult()
                                 : arith::AddIOp::create(builder, loc, lhs, rhs)
                                       .getResult();
                    break;
                  case cinm::ElementwiseKind::Sub:
                    result = isFloatOp
                                 ? arith::SubFOp::create(builder, loc, lhs, rhs)
                                       .getResult()
                                 : arith::SubIOp::create(builder, loc, lhs, rhs)
                                       .getResult();
                    break;
                  case cinm::ElementwiseKind::Mul:
                    result = isFloatOp
                                 ? arith::MulFOp::create(builder, loc, lhs, rhs)
                                       .getResult()
                                 : arith::MulIOp::create(builder, loc, lhs, rhs)
                                       .getResult();
                    break;
                  case cinm::ElementwiseKind::Div:
                    result =
                        isFloatOp
                            ? arith::DivFOp::create(builder, loc, lhs, rhs)
                                  .getResult()
                            : arith::DivSIOp::create(builder, loc, lhs, rhs)
                                  .getResult();
                    break;
                  case cinm::ElementwiseKind::Mod:
                    result =
                        isFloatOp
                            ? arith::RemFOp::create(builder, loc, lhs, rhs)
                                  .getResult()
                            : arith::RemSIOp::create(builder, loc, lhs, rhs)
                                  .getResult();
                    break;
                  case cinm::ElementwiseKind::And:
                    result = arith::AndIOp::create(builder, loc, lhs, rhs)
                                 .getResult();
                    break;
                  case cinm::ElementwiseKind::Or:
                    result = arith::OrIOp::create(builder, loc, lhs, rhs)
                                 .getResult();
                    break;
                  case cinm::ElementwiseKind::Xor:
                    result = arith::XOrIOp::create(builder, loc, lhs, rhs)
                                 .getResult();
                    break;
                  case cinm::ElementwiseKind::Not: // ~a = a xor 0b111111111
                    result =
                        builder
                            .create<arith::XOrIOp>(
                                loc, lhs,
                                arith::ConstantOp::create(
                                    builder, loc, lhs.getType(),
                                    builder.getIntegerAttr(lhs.getType(), -1)))
                            .getResult();
                    break;
                  default:
                    break;
                  }

                  linalg::YieldOp::create(builder, loc, result);
                });
          }
        });

    if (conversionResult.failed()) {
      return failure();
    }

    rewriter.replaceOp(op, newResults);

    return success();
  }
};

LogicalResult computeScatterMapForGemm(cnm::BufferType bufferTyAB,
                                       int64_t rowsA, int64_t colsB,
                                       AffineMap &scatterA, AffineMap &scatterB,
                                       AffineMap &scatterGatherC) {

  auto wgElts = 1;
  SmallVector<int64_t, 4> wgShapeWithoutUnits; // only non unit dims
  for (auto dim : bufferTyAB.getWorkgroupShape()) {
    if (dim != 1) {
      wgElts *= dim;
      wgShapeWithoutUnits.push_back(dim);
    }
  }

  // this assumption relies on the tiling phase
  if (wgElts != rowsA * colsB)
    return failure();

  // couple of situations we know work
  if (rowsA == 1 || colsB == 1 ||
      wgShapeWithoutUnits == ArrayRef<int64_t>{rowsA, colsB} ||
      wgShapeWithoutUnits == ArrayRef<int64_t>{colsB, rowsA}) {

    auto ctx = bufferTyAB.getContext();
    auto numInputs = bufferTyAB.getWorkgroupShape().size();
    const auto linearInput =
        linearizeIndices(ctx, bufferTyAB.getWorkgroupShape());

    scatterA = mlir::simplifyAffineMapWithBounds(
        AffineMap::get(numInputs, 0, linearInput % rowsA),
        bufferTyAB.getWorkgroupShape());

    scatterB = mlir::simplifyAffineMapWithBounds(
        AffineMap::get(numInputs, 0, linearInput % colsB),
        bufferTyAB.getWorkgroupShape());

    SmallVector<AffineExpr> results;
    structureIndex(linearInput, {rowsA, colsB}, results);
    scatterGatherC = mlir::simplifyAffineMapWithBounds(
        AffineMap::get(numInputs, 0, std::move(results), ctx),
        bufferTyAB.getWorkgroupShape());

    return success();
  }

  return failure();
}

struct ConvertCinmGemmToCnm : public OpConversionPattern<cinm::GemmOp> {
  using OpConversionPattern<cinm::GemmOp>::OpConversionPattern;

  static Value transpose(ImplicitLocOpBuilder &builder, Value tensor) {
    auto inTy = cast<ShapedType>(tensor.getType());
    auto shape = inTy.getShape();

    SmallVector<int64_t, 2> newShape{shape[1], shape[0]};
    SmallVector<int64_t, 2> perms{1, 0};
    Value output;
    bool tensorOutput;
    if (llvm::isa<TensorType>(tensor.getType())) {
      output =
          tensor::EmptyOp::create(builder, newShape, inTy.getElementType());
      tensorOutput = true;
    } else {
      output = memref::AllocOp::create(
          builder, MemRefType::get(newShape, inTy.getElementType()));
      tensorOutput = false;
    }
    auto transposeRight =
        linalg::TransposeOp::create(builder, tensor, output, perms);

    if (tensorOutput)
      return transposeRight->getResult(0);
    else
      return output;
  }

  LogicalResult
  matchAndRewrite(cinm::GemmOp op, OpConversionPattern<cinm::GemmOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TypedValue<ShapedType> lhs =
        llvm::cast<TypedValue<ShapedType>>(op.getLhs());
    TypedValue<ShapedType> rhs =
        llvm::cast<TypedValue<ShapedType>>(op.getRhs());

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    cinm::ComputeOp computeBlock = mlir::cinm::getEnclosingComputeBlock(op);
    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, computeBlock.getCnmWorkgroupType());
    auto wgShape = computeBlock.getWorkgroupShape();

    auto transposeRight = transpose(builder, rhs);

    auto tilingParms = cinm::TilingParameters::fromComputeBlock(computeBlock);
    auto elTyBytes = lhs.getType().getElementTypeBitWidth() / 8;

    // Check that the tiling pass chose a fitting reduction size.
    auto reductionSize = lhs.getType().getDimSize(1);
    if (reductionSize * 2 * elTyBytes >
        tilingParms.bufferSizeOfLeaf() - elTyBytes) {
      return op->emitOpError(
          "cannot be converted to CINM, reduction size is too large");
    }
    auto eltTy = lhs.getType().getElementType();
    // buffer type for A and B
    cnm::BufferType bufferType =
        cnm::BufferType::get({reductionSize}, eltTy, wgShape);
    Value bufferA = cnm::AllocOp::create(builder, bufferType, workgroup);
    Value bufferB = cnm::AllocOp::create(builder, bufferType, workgroup);

    // C has a single element and no dimensions
    cnm::BufferType bufferCType = cnm::BufferType::get({}, eltTy, wgShape);
    Value bufferC = cnm::AllocOp::create(builder, bufferCType, workgroup);

    //::mlir::Value input, ::mlir::Value buffer, ::mlir::Value wg,
    //:::mlir::AffineMap scatterMap);
    AffineMap scatterA;
    AffineMap scatterB;
    AffineMap scatterGatherC;
    if (computeScatterMapForGemm(bufferType, lhs.getType().getDimSize(0),
                                 rhs.getType().getDimSize(1), scatterA,
                                 scatterB, scatterGatherC)
            .failed()) {
      return op->emitOpError("Cannot be converted to CINM, parallel dims "
                             "cannot be mapped onto workgroup (")
             << wgShape << ")";
    }
    cnm::ScatterOp::create(builder, op.getLhs(), bufferA, workgroup,
                           std::move(scatterA));
    cnm::ScatterOp::create(builder, transposeRight, bufferB, workgroup,
                           std::move(scatterB));

    // the bias is the initializer for the out buffer
    // since it has same shape as output we can use same gather map
    Value outputInit;
    if (op.getBias()) {
      outputInit = op.getBias();
    } else if (op.getResult()) {
      outputInit = arith::ConstantOp::create(
          builder, op.getResult().getType(),
          builder.getZeroAttr(op.getResult().getType()));
    } else {
      // memref op with initializer
      outputInit = op.getOut();
    }
    assert(outputInit);
    cnm::ScatterOp::create(builder, outputInit, bufferC, workgroup,
                           scatterGatherC);

    SmallVector<AffineMap, 2> indexingMaps{
        AffineMap::getMultiDimIdentityMap(1, getContext()),
        AffineMap::getMultiDimIdentityMap(1, getContext()),
        AffineMap::get(1, 0, getContext()),
    };

    createLaunchOp(
        builder, workgroup, ValueRange{bufferA, bufferB}, ValueRange{bufferC},
        [&](ImplicitLocOpBuilder &builder, ValueRange ins, ValueRange outs) {
          linalg::ContractOp::create(
              builder, TypeRange{}, ins, outs,
              builder.getAffineMapArrayAttr(indexingMaps));
        });

    Value outbuf;
    if (op.getBias()) {
      outbuf = op.getBias();
    } else if (op.getResult()) {
      outbuf = tensor::EmptyOp::create(builder, op.getResult().getType(),
                                       ValueRange{});
    } else {
      outbuf = op.getOut();
    }
    auto gather = cnm::GatherOp::create(builder, bufferC, workgroup,
                                        scatterGatherC, outbuf);

    if (op.getResult()) {
      rewriter.replaceOp(op, ValueRange{gather.getOutput()});
    } else {
      rewriter.eraseOp(op);
    }
    cnm::FreeWorkgroupOp::create(builder, workgroup);
    return success();
  }
};

struct ConvertCinmGemvToCnm : public OpConversionPattern<cinm::GemvOp> {
  using OpConversionPattern<cinm::GemvOp>::OpConversionPattern;
  ConvertCinmGemvToCnm(MLIRContext *ctx)
      : mlir::OpConversionPattern<cinm::GemvOp>(ctx) {
    this->setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(cinm::GemvOp op,
                  OpConversionPattern<cinm::GemvOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    cinm::ComputeOp computeBlock = mlir::cinm::getEnclosingComputeBlock(op);
    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, computeBlock.getCnmWorkgroupType());
    Value outputInit;
    if (op.getResult()) {
      outputInit = arith::ConstantOp::create(
          builder, op.getResult().getType(),
          builder.getZeroAttr(op.getResult().getType()));
    } else {
      outputInit = op.getOut();
    }

    llvm::SmallVector<Value, 1> newResults;
    if (convertCinmToCnm(builder, op, workgroup.getResult(), computeBlock, {1},
                         adaptor.getOperands(), ValueRange{outputInit},
                         op->getResults(), newResults,
                         [&](ImplicitLocOpBuilder &builder, ValueRange inputs,
                             ValueRange outputs) {
                           linalg::MatvecOp::create(builder, inputs, outputs);
                         })
            .failed()) {
      return failure();
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct ConvertCinmReduceToCnm : public OpConversionPattern<cinm::ReduceOp> {
  using OpConversionPattern<cinm::ReduceOp>::OpConversionPattern;
  ConvertCinmReduceToCnm(MLIRContext *ctx)
      : mlir::OpConversionPattern<cinm::ReduceOp>(ctx) {
    this->setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(cinm::ReduceOp op,
                  OpConversionPattern<cinm::ReduceOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    cinm::ComputeOp computeBlock = mlir::cinm::getEnclosingComputeBlock(op);
    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, computeBlock.getCnmWorkgroupType());
    auto outputInit = arith::ConstantOp::create(
        builder, op.getResult().getType(),
        builder.getZeroAttr(op.getResult().getType()));

    const bool isFloatOp =
        isa<FloatType>(cast<ShapedType>(op.getType()).getElementType());

    llvm::SmallVector<Value, 1> newResults;
    if (convertCinmToCnm(
            builder, op, workgroup.getResult(), computeBlock, {1},
            adaptor.getOperands(), ValueRange{outputInit}, op->getResults(),
            newResults,
            [&](ImplicitLocOpBuilder &builder, ValueRange inputs,
                ValueRange outputs) {
              linalg::ReduceOp::create(
                  builder, inputs, outputs, ArrayRef<int64_t>{0},
                  [&](OpBuilder &builder, Location loc,
                      ValueRange inputs) -> void {
                    Value result;
                    switch (op.getMethod()) {
                    case mlir::cinm::ReduceMethod::ADD: {
                      if (isFloatOp) {
                        result = arith::AddFOp::create(builder, loc, inputs[0],
                                                       inputs[1]);
                      } else {
                        result = arith::AddIOp::create(builder, loc, inputs[0],
                                                       inputs[1]);
                      }
                    } break;
                    case mlir::cinm::ReduceMethod::MUL: {
                      if (isFloatOp) {
                        result = arith::MulFOp::create(builder, loc, inputs[0],
                                                       inputs[1]);
                      } else {
                        result = arith::MulIOp::create(builder, loc, inputs[0],
                                                       inputs[1]);
                      }
                    } break;
                    case mlir::cinm::ReduceMethod::MAX: {
                      if (isFloatOp) {
                        result = arith::MaximumFOp::create(
                            builder, loc, inputs[0], inputs[1]);
                      } else {
                        result = arith::MaxSIOp::create(builder, loc, inputs[0],
                                                        inputs[1]);
                      }
                    } break;
                    case mlir::cinm::ReduceMethod::MIN: {
                      if (isFloatOp) {
                        result = arith::MinimumFOp::create(
                            builder, loc, inputs[0], inputs[1]);
                      } else {
                        result = arith::MinSIOp::create(builder, loc, inputs[0],
                                                        inputs[1]);
                      }
                    } break;
                    }
                    linalg::YieldOp::create(builder, loc, result);
                  });
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
    for (auto &toCopy : adaptor.getBody().front().without_terminator()) {
      rewriter.clone(toCopy, mapper);
    }
    auto term = op.getBody().front().getTerminator();
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
  patterns.insert<ConvertElementwiseOpToCnm>(ctx);
  // matmul
  patterns.insert<ConvertCinmGemmToCnm>(ctx);
  patterns.insert<ConvertCinmGemvToCnm>(ctx);
  // reduce
  patterns.insert<ConvertCinmReduceToCnm>(ctx);
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

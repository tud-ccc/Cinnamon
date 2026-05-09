#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <array>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/TilingInterface.h>

using namespace mlir;
using namespace mlir::cinm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static SmallVector<Value> createNestedScfForLoops(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> tripCounts,
    ArrayRef<int64_t> steps, ValueRange iterArgs,
    std::function<SmallVector<Value>(OpBuilder &, Location, ValueRange,
                                     ValueRange)>
        bodyBuilder) {
  assert(tripCounts.size() == steps.size());
  const unsigned rank = tripCounts.size();

  SmallVector<Value> lbs(rank), ubs(rank), stepVals(rank);
  for (unsigned d = 0; d < rank; ++d) {
    lbs[d] = arith::ConstantIndexOp::create(builder, loc, 0);
    ubs[d] = arith::ConstantIndexOp::create(builder, loc, tripCounts[d]);
    stepVals[d] = arith::ConstantIndexOp::create(builder, loc, steps[d]);
  }

  SmallVector<Value> ivs;
  ivs.reserve(rank);
  std::function<SmallVector<Value>(unsigned, ValueRange)> build;
  build = [&](unsigned depth, ValueRange carried) -> SmallVector<Value> {
    if (depth == rank) {
      return bodyBuilder(builder, loc, ivs, carried);
    }
    auto loop = scf::ForOp::create(builder, loc, lbs[depth], ubs[depth],
                                   stepVals[depth], carried);
    builder.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
    SmallVector<Value> yielded = build(depth + 1, loop.getRegionIterArgs());
    scf::YieldOp::create(builder, loc, yielded);
    ivs.pop_back();
    builder.setInsertionPointAfter(loop);
    return SmallVector<Value>(loop.getResults().begin(),
                              loop.getResults().end());
  };

  return build(0, iterArgs);
}

static constexpr std::array<int64_t, 1> noStaticOffsets1{ShapedType::kDynamic};
static constexpr std::array<int64_t, 1> unitStrides1{1};

static Value extractSliceND(OpBuilder &builder, Location loc,
                            TypedValue<ShapedType> tensorOrMemref,
                            ArrayRef<int64_t> sizes, ValueRange offsets) {
  assert(offsets.size() == sizes.size());

  const ShapedType sliceTy = tensorOrMemref.getType().clone(sizes);
  if (llvm::isa<RankedTensorType>(tensorOrMemref.getType())) {
    llvm::SmallVector<int64_t> unitStrides(offsets.size(), 1);
    llvm::SmallVector<int64_t> noStaticOffsets(offsets.size(),
                                               ShapedType::kDynamic);
    return tensor::ExtractSliceOp::create(
        builder, loc, sliceTy, tensorOrMemref, offsets, ValueRange{},
        ValueRange{}, noStaticOffsets, sliceTy.getShape(), unitStrides);
  } else if (llvm::isa<MemRefType>(tensorOrMemref.getType())) {
    llvm::SmallVector<OpFoldResult> offsetsFoldRes(offsets.begin(),
                                                   offsets.end());
    auto one = builder.getI64IntegerAttr(1);
    llvm::SmallVector<OpFoldResult> stridesFoldRes(offsets.size(), one);
    llvm::SmallVector<OpFoldResult> sizesFoldRes;
    sizesFoldRes.reserve(offsets.size());
    for (auto i : sizes) {
      auto attr = builder.getI64IntegerAttr(i);
      sizesFoldRes.push_back(attr);
    }

    return memref::SubViewOp::create(builder, loc, tensorOrMemref,
                                     offsetsFoldRes, sizesFoldRes,
                                     stridesFoldRes);
  }
  assert(false && "type not handled");
}
static Value extractSlice1D(OpBuilder &builder, Location loc,
                            TypedValue<ShapedType> tensorOrMemref, int64_t size,
                            Value offset) {
  return extractSliceND(builder, loc, tensorOrMemref, {size}, {offset});
}

static Value extractSlice(OpBuilder &builder, Location loc,
                          TypedValue<ShapedType> tensorOrMemref, int64_t a,
                          int64_t b, Value ia, Value ib) {
  return extractSliceND(builder, loc, tensorOrMemref, {a, b}, {ia, ib});
}

static Value insertSliceND(OpBuilder &builder, Location loc, Value slice,
                           TypedValue<ShapedType> tensorOrMemref,
                           ArrayRef<int64_t> sizes, ValueRange offsets) {
  assert(offsets.size() == sizes.size());

  const ShapedType sliceTy = tensorOrMemref.getType().clone(sizes);
  if (llvm::isa<RankedTensorType>(tensorOrMemref.getType())) {
    llvm::SmallVector<int64_t> unitStrides(offsets.size(), 1);
    llvm::SmallVector<int64_t> noStaticOffsets(offsets.size(),
                                               ShapedType::kDynamic);
    return tensor::InsertSliceOp::create(
        builder, loc, slice, tensorOrMemref, offsets, ValueRange{},
        ValueRange{}, noStaticOffsets, sliceTy.getShape(), unitStrides);
  } else if (llvm::isa<MemRefType>(tensorOrMemref.getType())) {
    auto dest = extractSliceND(builder, loc, tensorOrMemref, sizes, offsets);
    memref::CopyOp::create(builder, loc, slice, dest);
    return {};
  }
  assert(false && "type not handled");
}

static constexpr std::array<int64_t, 2> noStaticOffsets2D{ShapedType::kDynamic,
                                                          ShapedType::kDynamic};
static constexpr std::array<int64_t, 2> unitStrides2D{1, 1};

// ---------------------------------------------------------------------------
// getTilableDimSizes implementations
// ---------------------------------------------------------------------------

void ReduceOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto inputType = cast<ShapedType>(getOperand().getType());
  auto shape = inputType.getShape();
  dimSizes.append(shape.begin(), shape.end());
}

void ElementwiseOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  dimSizes.push_back(cast<ShapedType>(getLhs().getType()).getNumElements());
}

void GemmOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  auto rhsType = cast<ShapedType>(getRhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // M
  dimSizes.push_back(rhsType.getDimSize(1)); // N
  dimSizes.push_back(lhsType.getDimSize(1)); // K
}

void BatchGemmOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  auto rhsType = cast<ShapedType>(getRhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // B
  dimSizes.push_back(lhsType.getDimSize(1)); // M
  dimSizes.push_back(rhsType.getDimSize(2)); // N
  dimSizes.push_back(lhsType.getDimSize(2)); // K
}

void BatchGemvOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // B
  dimSizes.push_back(lhsType.getDimSize(1)); // M
  dimSizes.push_back(lhsType.getDimSize(2)); // K
}

void GemvOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // M
  dimSizes.push_back(lhsType.getDimSize(1)); // K
}

void ActivateOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  dimSizes.push_back(cast<ShapedType>(getInput().getType()).getNumElements());
}

// ---------------------------------------------------------------------------
// convertToTiledOps implementations
// ---------------------------------------------------------------------------

static TypedAttr getNeutralElement(ReduceMethod r, Type ty, OpBuilder &builder,
                                   Location loc) {
  return arith::getIdentityValueAttr(getArithConstant(r, ty), ty, builder, loc);
}

static Value materializeReduction(OpBuilder &builder, Location loc,
                                  ReduceMethod method, Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType());
  return arith::getReductionOp(getArithConstant(method, lhs.getType()), builder,
                               loc, lhs, rhs);
}

DiagnosedSilenceableFailure
ReduceOp::convertToTiledOps(RewriterBase &builder, ArrayRef<int64_t> tileSizes,
                            SmallVectorImpl<Value> &results) {
  auto inputType = getInput().getType();
  if (static_cast<int64_t>(tileSizes.size()) != inputType.getRank())
    return emitSilenceableFailure(getLoc())
           << "expected " << inputType.getRank()
           << " tiling factors for reduce, got " << tileSizes.size();

  auto method = getMethod();

  // To tile the reduction two different templates may be used:
  //   tensor<NxM> -> tensor<N>
  // - tile the parallel part (N) and concatenate the results
  // - tile the reduction part (M) and add the partial results

  // Assume you have been given tile sizes for both (for all dimensions
  // basically). Then:

  int64_t reductionDim = getDimensionAttr().getInt();
  if (reductionDim < 0)
    reductionDim += inputType.getRank();
  // auto reductionExtent = inputType.getDimSize(dim);

  auto neutral =
      getNeutralElement(method, inputType.getElementType(), builder, getLoc());

  auto resultType = getResult().getType();

  Value result;
  if (isa<TensorType>(resultType))
    result = tensor::EmptyOp::create(builder, getLoc(), resultType, {});
  else if (resultType.isIntOrFloat())
    result = arith::ConstantOp::create(builder, getLoc(), neutral);
  else
    // memref not supported
    return emitSilenceableFailure(getLoc(), "Cannot tile reduction on type ")
           << resultType;

  SmallVector<Value> loopResult = createNestedAffineForLoops(
      builder, getLoc(), inputType.getShape(), tileSizes, {result},
      [&](OpBuilder &b, Location loc, ValueRange tileIndex,
          ValueRange iterArgs) -> SmallVector<Value> {
        auto acc = iterArgs[0];
        Value sliceIn =
            extractSliceND(b, loc, getInput(), tileSizes, tileIndex);

        SmallVector<int64_t> resultTileSize(tileSizes);
        resultTileSize.erase(resultTileSize.begin() + reductionDim);
        SmallVector<Value> resultTileIndex(tileIndex);
        resultTileIndex.erase(resultTileIndex.begin() + reductionDim);

        Type resultTy = resultTileSize.size() == 0
                            ? inputType.getElementType()
                            : inputType.cloneWith(resultTileSize,
                                                  inputType.getElementType());

        auto smaller =
            ReduceOp::create(b, loc, resultTy, method, sliceIn, reductionDim);

        auto shapedResultTile =
            llvm::dyn_cast_or_null<TypedValue<ShapedType>>(smaller.getResult());
        auto shapedResult = llvm::dyn_cast_or_null<TypedValue<ShapedType>>(acc);

        if (shapedResult && shapedResultTile) {
          return {insertSliceND(b, loc, shapedResultTile, shapedResult,
                                resultTileSize, resultTileIndex)};
        } else if (smaller.getResult().getType().isIntOrFloat()) {
          if (isa<TensorType>(acc.getType())) {
            auto accElt =
                tensor::ExtractOp::create(b, loc, acc, resultTileIndex);
            auto red = materializeReduction(b, loc, method, accElt,
                                            smaller.getResult());
            return {
                tensor::InsertOp::create(b, loc, red, acc, resultTileIndex)};
          } else if (acc.getType().isIntOrFloat()) {
            // both are scalars
            return {
                materializeReduction(b, loc, method, acc, smaller.getResult())};
          }
        }
        assert(false && "unhandled type");
      });

  results.append(loopResult.begin(), loopResult.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
ElementwiseOp::convertToTiledOps(RewriterBase &rewriter,
                                 ArrayRef<int64_t> tilingFactors,
                                 SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 1)
    return emitSilenceableFailure(getLoc())
           << "expected 1 tiling factor for elementwise, got "
           << tilingFactors.size();

  ImplicitLocOpBuilder builder(getLoc(), rewriter);

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();
  const bool isUnaryOp = !rhs;

  ShapedType tensorTy = cast<ShapedType>(lhs.getType());
  auto shape = tensorTy.getShape();
  const ShapedType originalType = tensorTy;
  Value originalShapeValue;

  TypedValue<ShapedType> memrefOut =
      llvm::dyn_cast_or_null<TypedValue<ShapedType>>(getOut());
  if (shape.size() > 1) {
    originalShapeValue = arith::ConstantOp::create(
        builder,
        RankedTensorType::get({static_cast<int64_t>(shape.size())},
                              builder.getI64Type()),
        builder.getI64TensorAttr(shape));
    lhs = cinm::reshapeStatic(builder, builder.getLoc(), lhs,
                              {tensorTy.getNumElements()});
    if (!isUnaryOp) {
      rhs = cinm::reshapeStatic(builder, builder.getLoc(), rhs,
                                {tensorTy.getNumElements()});
    }
    if (memrefOut) {
      memrefOut = cinm::reshapeStatic(builder, builder.getLoc(), memrefOut,
                                      {tensorTy.getNumElements()});
    }
    tensorTy = lhs.getType();
  }

  const int64_t numElements = tensorTy.getNumElements();
  int64_t tileSize =
      std::max<int64_t>(1, std::min<int64_t>(tilingFactors[0], numElements));

  ValueRange resultInit{};
  if (getResult()) {
    resultInit =
        tensor::EmptyOp::create(builder, tensorTy, ValueRange{})->getResults();
  } else {
    assert(memrefOut);
  }

  SmallVector<Value> loopResult = createNestedAffineForLoops(
      builder, getLoc(), {numElements}, {tileSize}, resultInit,
      [&](OpBuilder &b, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value base = indices[0];
        SmallVector<OpFoldResult, 1> off{base};

        Value lhsSlice = extractSlice1D(b, loc, lhs, tileSize, base);

        Value rhsSlice;
        if (!isUnaryOp)
          rhsSlice = extractSlice1D(b, loc, rhs, tileSize, base);

        Value sliceOut;
        if (memrefOut)
          sliceOut = extractSlice1D(b, loc, memrefOut, tileSize, base);

        ElementwiseOp smaller = ElementwiseOp::create(
            b, loc, getKind(), lhsSlice, rhsSlice, sliceOut);

        if (smaller.getResult()) {
          SmallVector<OpFoldResult, 1> siz{b.getIndexAttr(tileSize)};
          SmallVector<OpFoldResult, 1> str{b.getI64IntegerAttr(1)};
          Value subResult = tensor::InsertSliceOp::create(
              b, loc, smaller.getResult(), iterArgs[0], off, siz, str);
          return {subResult};
        }
        return {};
      });

  if (originalType.getRank() > 1) {
    loopResult[0] = tensor::ReshapeOp::create(
        builder, originalType, loopResult[0], originalShapeValue);
  }
  results.append(loopResult.begin(), loopResult.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
GemmOp::convertToTiledOps(RewriterBase &rewriter,
                          ArrayRef<int64_t> tilingFactors,
                          SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 3)
    return emitSilenceableFailure(getLoc())
           << "expected 3 tiling factors [tM,tN,tK] for gemm, got "
           << tilingFactors.size();

  Location loc = getLoc();
  OpBuilder &builder = rewriter;

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();

  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  ShapedType resultType;
  if (getResult())
    resultType = getResult().getType();
  else
    resultType = getOut().getType();

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return DiagnosedSilenceableFailure::definiteFailure();

  const int64_t M = lhsType.getDimSize(0);
  const int64_t K = lhsType.getDimSize(1);
  const int64_t N = rhsType.getDimSize(1);
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
      ShapedType::isDynamic(N))
    return emitSilenceableFailure(getLoc())
           << "Unsupported: tiling on dynamic dimensions";

  const int64_t p0 = tilingFactors[0];
  const int64_t p1 = tilingFactors[1];
  const int64_t r = tilingFactors[2];

  ValueRange initArgs{};
  if (!getOut()) {
    Value resultInit = tensor::EmptyOp::create(
        builder, loc, resultType.getShape(), resultType.getElementType());
    initArgs = resultInit;
  }

  Type eltTy = resultType.getElementType();

  SmallVector<Value> finals = createNestedAffineForLoops(
      builder, getLoc(), resultType.getShape(), {p0, p1}, initArgs,
      [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
                  ValueRange iterArgs) -> SmallVector<Value> {
        const auto parIndices = indices;
        const SmallVector<int64_t, 2> resultSizes{p0, p1};
        const ValueRange resultDynamicOffsets = parIndices;

        Value biasSlice;
        if (auto bias = getBias())
          biasSlice = extractSlice(builder, loc, bias, p0, p1, parIndices[0],
                                   parIndices[1]);

        // If the output is a memref, then slice it and possibly accumulate the bias into it before the inner loop.
        Value outBuf;
        if (auto outmemref = getOut(); outmemref && isa<MemRefType>(outmemref.getType())) {
          outBuf = extractSlice(builder, loc,
                                cast<TypedValue<ShapedType>>(outmemref), p0, p1,
                                parIndices[0], parIndices[1]);
          if (biasSlice)
            linalg::AddOp::create(builder, loc, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

        // Tensor case: seed the [i,j] tile with bias or zeros, then carry the
        // full result tensor through the reduction loop. This is on purpose as
        // having the extract/insert slice inside the inner loop improves bufferization
        // results. 
        ValueRange innerIterArgInit{};
        if (!outBuf) {
          Value initTile = biasSlice;
          if (!initTile) {
            auto tileTy = RankedTensorType::get({p0, p1}, eltTy);
            initTile =
                arith::ConstantOp::create(
                    builder, loc,
                    DenseElementsAttr::get(tileTy, builder.getZeroAttr(eltTy)))
                    .getResult();
          }
          innerIterArgInit = insertSliceND(
              builder, loc, initTile, cast<TypedValue<ShapedType>>(iterArgs[0]),
              {p0, p1}, parIndices);
        }

        SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
            builder, loc, {K}, {r}, innerIterArgInit,
            [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
                        ValueRange innerIterArgs) -> SmallVector<Value> {
              const auto indexInRedDim = indices[0];

              Value lhsSlice = extractSlice(builder, loc, lhs, p0, r,
                                            parIndices[0], indexInRedDim);
              Value rhsSlice = extractSlice(builder, loc, rhs, r, p1,
                                            indexInRedDim, parIndices[1]);
              if (innerIterArgs.empty()) {
                // memref - create in place
                cinm::GemmOp::create(builder, loc, lhsSlice, rhsSlice, Value{},
                                     outBuf);
                return {};
              }
              // Then tensor version.

              Value accSlice = extractSlice(
                  builder, loc, cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                  p0, p1, parIndices[0], parIndices[1]);
              // note we set the out buf to the acc slice for better
              // bufferization result.
              Value tileResult =
                  cinm::GemmOp::create(builder, loc, lhsSlice, rhsSlice,
                                       accSlice, accSlice)
                      .getResult();
              Value updatedTensor = tensor::InsertSliceOp::create(
                  builder, loc, tileResult, innerIterArgs[0],
                  resultDynamicOffsets, ValueRange{}, ValueRange{},
                  ArrayRef(noStaticOffsets2D), resultSizes,
                  ArrayRef(unitStrides2D));
              return {updatedTensor};
            });

        if (getOut())
          return {};
        return {reductionResult[0]};
      });

  results.append(finals.begin(), finals.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
BatchGemmOp::convertToTiledOps(RewriterBase &rewriter,
                               ArrayRef<int64_t> tilingFactors,
                               SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 4)
    return emitSilenceableFailure(getLoc())
           << "expected 4 tiling factors [batch,tM,tN,tK] for batch_gemm, got "
           << tilingFactors.size();

  Location loc = getLoc();
  OpBuilder &builder = rewriter;

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();

  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  ShapedType resultType;
  if (getResult())
    resultType = getResult().getType();
  else
    resultType = getOut().getType();

  if (lhsType.getRank() != 3 || rhsType.getRank() != 3 ||
      resultType.getRank() != 3)
    return DiagnosedSilenceableFailure::definiteFailure();

  const int64_t B = lhsType.getDimSize(0);
  const int64_t M = lhsType.getDimSize(1);
  const int64_t K = lhsType.getDimSize(2);
  const int64_t N = rhsType.getDimSize(2);
  if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
      ShapedType::isDynamic(K) || ShapedType::isDynamic(N))
    return emitSilenceableFailure(getLoc())
           << "Unsupported: tiling on dynamic dimensions";

  const int64_t pB = tilingFactors[0];
  const int64_t pM = tilingFactors[1];
  const int64_t pN = tilingFactors[2];
  const int64_t r = tilingFactors[3];

  ValueRange initArgs{};
  if (!getOut()) {
    Value resultInit = tensor::EmptyOp::create(
        builder, loc, resultType.getShape(), resultType.getElementType());
    initArgs = resultInit;
  }

  Type eltTy = resultType.getElementType();

  SmallVector<Value> finals = createNestedAffineForLoops(
      builder, getLoc(), resultType.getShape(), {pB, pM, pN}, initArgs,
      [&, pB, pM, pN](OpBuilder &builder, Location loc, ValueRange indices,
                      ValueRange iterArgs) -> SmallVector<Value> {
        const auto parIndices = indices;

        Value biasSlice;
        if (auto bias = getBias())
          biasSlice = extractSliceND(builder, loc,
                                     cast<TypedValue<ShapedType>>(bias),
                                     {pB, pM, pN}, parIndices);

        // If the output is a memref, then slice it and possibly accumulate the bias into it before the inner loop.
        Value outBuf;
        if (auto outmemref = getOut(); outmemref && isa<MemRefType>(outmemref.getType())) {
          outBuf = extractSliceND(builder, loc,
                                  cast<TypedValue<ShapedType>>(outmemref),
                                  {pB, pM, pN}, parIndices);
          if (biasSlice)
            linalg::AddOp::create(builder, loc, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

        // Tensor case: seed the [b,i,j] tile with bias or zeros, then carry the
        // full result tensor through the reduction loop. This is on purpose as
        // having the extract/insert slice inside the inner loop improves bufferization
        // results.
        ValueRange innerIterArgInit{};
        if (!outBuf) {
          Value initTile = biasSlice;
          if (!initTile) {
            auto tileTy = RankedTensorType::get({pB, pM, pN}, eltTy);
            initTile =
                arith::ConstantOp::create(
                    builder, loc,
                    DenseElementsAttr::get(tileTy, builder.getZeroAttr(eltTy)))
                    .getResult();
          }
          innerIterArgInit = insertSliceND(
              builder, loc, initTile, cast<TypedValue<ShapedType>>(iterArgs[0]),
              {pB, pM, pN}, parIndices);
        }

        SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
            builder, loc, {K}, {r}, innerIterArgInit,
            [&, pB, pM, pN](OpBuilder &builder, Location loc, ValueRange indices,
                             ValueRange innerIterArgs) -> SmallVector<Value> {
              const auto k = indices[0];

              SmallVector<Value> lhsOffsets{parIndices[0], parIndices[1], k};
              SmallVector<Value> rhsOffsets{parIndices[0], k, parIndices[2]};
              Value lhsSlice =
                  extractSliceND(builder, loc, lhs, {pB, pM, r}, lhsOffsets);
              Value rhsSlice =
                  extractSliceND(builder, loc, rhs, {pB, r, pN}, rhsOffsets);
              if (innerIterArgs.empty()) {
                // memref - create in place
                cinm::BatchGemmOp::create(builder, loc, lhsSlice, rhsSlice,
                                          Value{}, outBuf);
                return {};
              }
              // Then tensor version.
              Value accSlice = extractSliceND(
                  builder, loc, cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                  {pB, pM, pN}, parIndices);
              Value tileResult =
                  cinm::BatchGemmOp::create(builder, loc, lhsSlice, rhsSlice,
                                            accSlice, accSlice)
                      .getResult();
              Value updatedTensor = insertSliceND(
                  builder, loc, tileResult,
                  cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                  {pB, pM, pN}, parIndices);
              return {updatedTensor};
            });

        if (getOut())
          return {};
        return {reductionResult[0]};
      });

  results.append(finals.begin(), finals.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
BatchGemvOp::convertToTiledOps(RewriterBase &rewriter,
                               ArrayRef<int64_t> tilingFactors,
                               SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 3)
    return emitSilenceableFailure(getLoc())
           << "expected 3 tiling factors [batch,tM,tK] for batch_gemv, got "
           << tilingFactors.size();

  Location loc = getLoc();
  OpBuilder &builder = rewriter;

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();

  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  ShapedType resultType;
  if (getResult())
    resultType = getResult().getType();
  else
    resultType = getOut().getType();

  if (lhsType.getRank() != 3 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return DiagnosedSilenceableFailure::definiteFailure();

  const int64_t B = lhsType.getDimSize(0);
  const int64_t M = lhsType.getDimSize(1);
  const int64_t K = lhsType.getDimSize(2);
  if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
      ShapedType::isDynamic(K))
    return emitSilenceableFailure(getLoc())
           << "Unsupported: tiling on dynamic dimensions";

  const int64_t pB = tilingFactors[0];
  const int64_t pM = tilingFactors[1];
  const int64_t rK = tilingFactors[2];

  ValueRange initArgs{};
  if (!getOut()) {
    Value resultInit = tensor::EmptyOp::create(
        builder, loc, resultType.getShape(), resultType.getElementType());
    initArgs = resultInit;
  }

  Type eltTy = resultType.getElementType();

  SmallVector<Value> finals = createNestedAffineForLoops(
      builder, getLoc(), resultType.getShape(), {pB, pM}, initArgs,
      [&, pB, pM](OpBuilder &builder, Location loc, ValueRange indices,
                  ValueRange iterArgs) -> SmallVector<Value> {
        const auto parIndices = indices;

        Value biasSlice;
        if (auto bias = getBias())
          biasSlice = extractSliceND(builder, loc,
                                     cast<TypedValue<ShapedType>>(bias),
                                     {pB, pM}, parIndices);

        // If the output is a memref, then slice it and possibly accumulate the bias into it before the inner loop.
        Value outBuf;
        if (auto outmemref = getOut(); outmemref && isa<MemRefType>(outmemref.getType())) {
          outBuf = extractSliceND(builder, loc,
                                  cast<TypedValue<ShapedType>>(outmemref),
                                  {pB, pM}, parIndices);
          if (biasSlice)
            linalg::AddOp::create(builder, loc, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

        // Tensor case: seed the [b,i] tile with bias or zeros, then carry the
        // full result tensor through the reduction loop. This is on purpose as
        // having the extract/insert slice inside the inner loop improves bufferization
        // results.
        ValueRange innerIterArgInit{};
        if (!outBuf) {
          Value initTile = biasSlice;
          if (!initTile) {
            auto tileTy = RankedTensorType::get({pB, pM}, eltTy);
            initTile =
                arith::ConstantOp::create(
                    builder, loc,
                    DenseElementsAttr::get(tileTy, builder.getZeroAttr(eltTy)))
                    .getResult();
          }
          innerIterArgInit = insertSliceND(
              builder, loc, initTile, cast<TypedValue<ShapedType>>(iterArgs[0]),
              {pB, pM}, parIndices);
        }

        SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
            builder, loc, {K}, {rK}, innerIterArgInit,
            [&, pB, pM](OpBuilder &builder, Location loc, ValueRange indices,
                         ValueRange innerIterArgs) -> SmallVector<Value> {
              const auto k = indices[0];

              SmallVector<Value> lhsOffsets{parIndices[0], parIndices[1], k};
              SmallVector<Value> rhsOffsets{parIndices[0], k};
              Value lhsSlice =
                  extractSliceND(builder, loc, lhs, {pB, pM, rK}, lhsOffsets);
              Value rhsSlice =
                  extractSliceND(builder, loc, rhs, {pB, rK}, rhsOffsets);
              if (innerIterArgs.empty()) {
                // memref - create in place
                cinm::BatchGemvOp::create(builder, loc, lhsSlice, rhsSlice,
                                          Value{}, outBuf);
                return {};
              }
              // Then tensor version.
              Value accSlice = extractSliceND(
                  builder, loc, cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                  {pB, pM}, parIndices);
              Value tileResult =
                  cinm::BatchGemvOp::create(builder, loc, lhsSlice, rhsSlice,
                                            accSlice, accSlice)
                      .getResult();
              Value updatedTensor = insertSliceND(
                  builder, loc, tileResult,
                  cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                  {pB, pM}, parIndices);
              return {updatedTensor};
            });

        if (getOut())
          return {};
        return {reductionResult[0]};
      });

  results.append(finals.begin(), finals.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
GemvOp::convertToTiledOps(RewriterBase &rewriter,
                          ArrayRef<int64_t> tilingFactors,
                          SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 2)
    return emitSilenceableFailure(getLoc())
           << "expected 2 tiling factors [tM,tK] for gemv, got "
           << tilingFactors.size();

  Location loc = getLoc();
  OpBuilder &builder = rewriter;

  Value A = getLhs();
  Value x = getRhs();

  auto aTy = cast<ShapedType>(A.getType());
  auto xTy = cast<ShapedType>(x.getType());
  ShapedType yTy;
  if (getResult())
    yTy = getResult().getType();
  else
    yTy = getOut().getType();

  if (aTy.getRank() != 2 || xTy.getRank() != 1 || yTy.getRank() != 1)
    return DiagnosedSilenceableFailure::definiteFailure();
  if (aTy.getElementType() != xTy.getElementType() ||
      aTy.getElementType() != yTy.getElementType())
    return DiagnosedSilenceableFailure::definiteFailure();
  auto elTy = aTy.getElementType();

  const int64_t M = aTy.getDimSize(0);
  const int64_t K = aTy.getDimSize(1);
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K))
    return emitSilenceableFailure(getLoc())
           << "Unsupported: tiling on dynamic dimensions";

  const int64_t pM = tilingFactors[0];
  const int64_t rK = tilingFactors[1];

  ValueRange initArgs{};
  if (!getOut()) {
    Value resultInit =
        tensor::EmptyOp::create(builder, loc, yTy.getShape(), elTy);
    initArgs = resultInit;
  }

  SmallVector<Value> loopResults = createNestedAffineForLoops(
      builder, loc, ArrayRef<int64_t>{M}, ArrayRef<int64_t>{pM}, initArgs,
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iters) -> SmallVector<Value> {
        Value i = ivs[0];

        Value biasSlice;
        if (auto bias = getBias())
          biasSlice = extractSlice1D(b, loc2, bias, pM, i);

        // If the output is a memref, then slice it and possibly accumulate the bias into it before the inner loop.
        Value outBuf;
        if (auto outmemref = getOut(); outmemref && isa<MemRefType>(outmemref.getType())) {
          outBuf = extractSlice1D(
              b, loc2, cast<TypedValue<ShapedType>>(outmemref), pM, i);
          if (biasSlice)
            linalg::AddOp::create(b, loc2, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

        // Tensor case: seed the [i] tile with bias or zeros, then carry the
        // full result tensor through the reduction loop. This is on purpose as
        // having the extract/insert slice inside the inner loop improves bufferization
        // results.
        ValueRange innerIterArgInit{};
        if (!outBuf) {
          Value initTile = biasSlice;
          if (!initTile) {
            auto tileTy = RankedTensorType::get({pM}, elTy);
            initTile =
                arith::ConstantOp::create(
                    b, loc2,
                    DenseElementsAttr::get(tileTy, b.getZeroAttr(elTy)))
                    .getResult();
          }
          innerIterArgInit = insertSliceND(b, loc2, initTile,
                                           cast<TypedValue<ShapedType>>(iters[0]),
                                           {pM}, ValueRange{i});
        }

        SmallVector<Value> red = createNestedAffineForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{rK},
            innerIterArgInit,
            [&](OpBuilder &b2, Location loc3, ValueRange kIvs,
                ValueRange innerAccArgs) -> SmallVector<Value> {
              Value k = kIvs[0];

              Value aTile = extractSlice(
                  b2, loc3, cast<TypedValue<ShapedType>>(A), pM, rK, i, k);
              Value xTile = extractSlice1D(
                  b2, loc3, cast<TypedValue<ShapedType>>(x), rK, k);

              if (innerAccArgs.empty()) {
                cinm::GemvOp::create(b2, loc3, aTile, xTile, Value{}, outBuf);
                return {};
              }
              Value accSlice = extractSlice1D(
                  b2, loc3, cast<TypedValue<ShapedType>>(innerAccArgs[0]), pM,
                  i);
              // Note we set the out buf for better bufferization result.
              Value tileResult = cinm::GemvOp::create(b2, loc3, aTile, xTile,
                                                      accSlice, accSlice)
                                     .getResult();
              Value updatedTensor =
                  tensor::InsertSliceOp::create(
                      b2, loc3, tileResult, innerAccArgs[0], ValueRange{i},
                      ValueRange{}, ValueRange{}, noStaticOffsets1,
                      ArrayRef<int64_t>({pM}), unitStrides1)
                      .getResult();
              return SmallVector<Value>{updatedTensor};
            });

        if (getOut())
          return {};
        return SmallVector<Value>{red[0]};
      });

  results.append(loopResults.begin(), loopResults.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
ActivateOp::convertToTiledOps(RewriterBase &rewriter,
                              ArrayRef<int64_t> tilingFactors,
                              SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 1)
    return emitSilenceableFailure(getLoc())
           << "expected 1 tiling factor for activate, got "
           << tilingFactors.size();

  ImplicitLocOpBuilder builder(getLoc(), rewriter);
  auto inputT = getInput();
  auto inTy = inputT.getType();
  Type elt = inTy.getElementType();

  const ShapedType originalTy = inTy;
  Value originalShapeValue;
  if (inTy.getRank() > 1) {
    originalShapeValue = arith::ConstantOp::create(
        builder, RankedTensorType::get({inTy.getRank()}, builder.getI64Type()),
        builder.getI64TensorAttr(inTy.getShape()));
    inputT = reshapeStatic(builder, builder.getLoc(), inputT,
                           ArrayRef<int64_t>{inTy.getNumElements()});
    inTy = inputT.getType();
  }

  const int64_t total = inTy.getNumElements();
  const int64_t p = tilingFactors[0];

  Value init = tensor::EmptyOp::create(builder, inTy.getShape(), elt);
  Value totalC = arith::ConstantIndexOp::create(builder, total);
  Value pC = arith::ConstantIndexOp::create(builder, p);

  SmallVector<Value> finals = createNestedScfForLoops(
      builder, getLoc(), ArrayRef<int64_t>{total}, ArrayRef<int64_t>{p},
      ValueRange{init},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange iters) -> SmallVector<Value> {
        Value i = ivs[0];

        Value rem = arith::SubIOp::create(b, loc, totalC, i);
        Value useP =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, rem, pC);
        Value thisTile = arith::SelectOp::create(b, loc, useP, pC, rem);

        SmallVector<OpFoldResult> off{i};
        SmallVector<OpFoldResult> siz{thisTile};
        SmallVector<OpFoldResult> str{b.getI64IntegerAttr(1)};

        auto inSlice =
            tensor::ExtractSliceOp::create(b, loc, inputT, off, siz, str);
        auto tile =
            cinm::ActivateOp::create(b, loc, getKind(), inSlice, Value());

        Value out = tensor::InsertSliceOp::create(b, loc, tile.getResult(),
                                                  iters[0], off, siz, str);
        return {out};
      });

  if (originalTy.getRank() > 1) {
    finals[0] = tensor::ReshapeOp::create(builder, originalTy, finals[0],
                                          originalShapeValue);
  }
  results.append(finals.begin(), finals.end());
  return DiagnosedSilenceableFailure::success();
}

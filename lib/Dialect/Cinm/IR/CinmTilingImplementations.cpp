#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

static constexpr std::array<int64_t, 2> noStaticOffsets2{ShapedType::kDynamic,
                                                         ShapedType::kDynamic};
static constexpr std::array<int64_t, 2> unitStrides2{1, 1};
static constexpr std::array<int64_t, 1> noStaticOffsets1{ShapedType::kDynamic};
static constexpr std::array<int64_t, 1> unitStrides1{1};
static constexpr std::array<int64_t, 3> noStaticOffsets3{
    ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};
static constexpr std::array<int64_t, 3> unitStrides3{1, 1, 1};

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

static arith::AtomicRMWKind getArithConstant(ReduceMethod r, Type ty) {
  switch (r) {
  case mlir::cinm::ReduceMethod::ADD:
    if (ty.isFloat()) {
      return mlir::arith::AtomicRMWKind::addf;
    } else {
      return mlir::arith::AtomicRMWKind::addi;
    }
  case mlir::cinm::ReduceMethod::MUL:
    if (ty.isFloat()) {
      return mlir::arith::AtomicRMWKind::mulf;
    } else {
      return mlir::arith::AtomicRMWKind::muli;
    }
  case mlir::cinm::ReduceMethod::MAX:
    if (ty.isFloat()) {
      return mlir::arith::AtomicRMWKind::maximumf;
    } else {
      return mlir::arith::AtomicRMWKind::maxu;
    }
  case mlir::cinm::ReduceMethod::MIN:
    if (ty.isFloat()) {
      return mlir::arith::AtomicRMWKind::minimumf;
    } else {
      return mlir::arith::AtomicRMWKind::minu;
    }
  }
}

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
        Value outBuf;
        if (auto outmemref = getOut()) {
          outBuf = extractSlice(builder, loc,
                                cast<TypedValue<ShapedType>>(outmemref), p0, p1,
                                parIndices[0], parIndices[1]);
          if (biasSlice)
            linalg::AddOp::create(builder, loc, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

        // For the tensor case: seed the [i,j] slice of the output tensor with
        // biasSlice or zeros before the reduction, then carry the full tensor
        // through the inner loop. The extract/insert pair lives next to the
        // GemmOp, which lets bufferization eliminate the intermediate buffer.
        ValueRange innerIterArgInit{};
        if (!outBuf) {
          Value initSlice;
          if (biasSlice) {
            initSlice = biasSlice;
          } else {
            auto reductionAccTy = RankedTensorType::get({p0, p1}, eltTy);
            DenseElementsAttr zeros;
            if (auto floatType = dyn_cast<FloatType>(eltTy))
              zeros = DenseElementsAttr::get(
                  reductionAccTy,
                  {APFloat::getZero(floatType.getFloatSemantics())});
            else
              zeros = DenseElementsAttr::get(
                  reductionAccTy,
                  {APInt::getZero(reductionAccTy.getElementTypeBitWidth())});
            initSlice =
                arith::ConstantOp::create(builder, loc, zeros).getResult();
          }
          innerIterArgInit =
              tensor::InsertSliceOp::create(
                  builder, loc, initSlice, iterArgs[0], resultDynamicOffsets,
                  ValueRange{}, ValueRange{}, ArrayRef(noStaticOffsets2D),
                  resultSizes, ArrayRef(unitStrides2D))
                  .getResult();
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
              if (outBuf) {
                cinm::GemmOp::create(builder, loc, lhsSlice, rhsSlice, Value{},
                                     outBuf);
                return {};
              }
              Value accSlice = extractSlice(
                  builder, loc, cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                  p0, p1, parIndices[0], parIndices[1]);
              Value tileResult =
                  cinm::GemmOp::create(builder, loc, lhsSlice, rhsSlice,
                                       accSlice, Value{})
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

  Value lhs = getLhs();
  Value rhs = getRhs();
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  ShapedType resultType;
  if (getResult())
    resultType = getResult().getType();
  else
    resultType = getOut().getType();

  if (!lhsType || !rhsType || !resultType)
    return DiagnosedSilenceableFailure::definiteFailure();
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

  const int64_t bTile = tilingFactors[0];
  const int64_t mTile = tilingFactors[1];
  const int64_t nTile = tilingFactors[2];
  const int64_t rTile = tilingFactors[3];

  Type elementTy = lhsType.getElementType();

  Value resultInit =
      tensor::EmptyOp::create(builder, loc, resultType.getShape(), elementTy);
  TypedAttr zeroAttr = builder.getZeroAttr(elementTy);

  Value Bc = arith::ConstantIndexOp::create(builder, loc, B);
  Value Mc = arith::ConstantIndexOp::create(builder, loc, M);
  Value Nc = arith::ConstantIndexOp::create(builder, loc, N);
  Value Kc = arith::ConstantIndexOp::create(builder, loc, K);
  Value bTileC = arith::ConstantIndexOp::create(builder, loc, bTile);
  Value mTileC = arith::ConstantIndexOp::create(builder, loc, mTile);
  Value nTileC = arith::ConstantIndexOp::create(builder, loc, nTile);
  Value rTileC = arith::ConstantIndexOp::create(builder, loc, rTile);

  SmallVector<Value> finals = createNestedScfForLoops(
      builder, loc, ArrayRef<int64_t>{B, M, N},
      ArrayRef<int64_t>{bTile, mTile, nTile}, ValueRange{resultInit},
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value iB = ivs[0];
        Value iM = ivs[1];
        Value jN = ivs[2];

        Value remB = arith::SubIOp::create(b, loc2, Bc, iB);
        Value remM = arith::SubIOp::create(b, loc2, Mc, iM);
        Value remN = arith::SubIOp::create(b, loc2, Nc, jN);

        Value useB = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remB, bTileC);
        Value useM = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remM, mTileC);
        Value useN = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remN, nTileC);

        Value bTileDyn = arith::SelectOp::create(b, loc2, useB, bTileC, remB);
        Value mTileDyn = arith::SelectOp::create(b, loc2, useM, mTileC, remM);
        Value nTileDyn = arith::SelectOp::create(b, loc2, useN, nTileC, remN);

        Value zeroScalar = arith::ConstantOp::create(b, loc2, zeroAttr);
        Value accEmpty = tensor::EmptyOp::create(
            b, loc2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                               ShapedType::kDynamic}),
            elementTy, ValueRange{bTileDyn, mTileDyn, nTileDyn});
        Value acc0 = linalg::FillOp::create(b, loc2, ValueRange{zeroScalar},
                                            ValueRange{accEmpty})
                         .getResult(0);

        SmallVector<Value> red = createNestedScfForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{rTile},
            ValueRange{acc0},
            [&](OpBuilder &b2, Location loc3, ValueRange redIvs,
                ValueRange accArgs) -> SmallVector<Value> {
              Value k = redIvs[0];
              Value remK = arith::SubIOp::create(b2, loc3, Kc, k);
              Value useR = arith::CmpIOp::create(
                  b2, loc3, arith::CmpIPredicate::ugt, remK, rTileC);
              Value kTileDyn =
                  arith::SelectOp::create(b2, loc3, useR, rTileC, remK);

              auto lhsTileTy = RankedTensorType::get({ShapedType::kDynamic,
                                                      ShapedType::kDynamic,
                                                      ShapedType::kDynamic},
                                                     elementTy);
              Value lhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, lhsTileTy, lhs, ValueRange{iB, iM, k},
                  ValueRange{bTileDyn, mTileDyn, kTileDyn}, ValueRange{},
                  noStaticOffsets3,
                  ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                                     ShapedType::kDynamic}),
                  unitStrides3);

              Value rhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, lhsTileTy, rhs, ValueRange{iB, k, jN},
                  ValueRange{bTileDyn, kTileDyn, nTileDyn}, ValueRange{},
                  noStaticOffsets3,
                  ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                                     ShapedType::kDynamic}),
                  unitStrides3);

              auto tileGemm = cinm::BatchGemmOp::create(b2, loc3, lhsSlice,
                                                        rhsSlice, accArgs[0]);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, tileGemm.getResult(), accArgs[0]);
              return SmallVector<Value>{mat.getResult()};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iterArgs[0], ValueRange{iB, iM, jN},
            ValueRange{bTileDyn, mTileDyn, nTileDyn}, ValueRange{},
            noStaticOffsets3,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                               ShapedType::kDynamic}),
            unitStrides3);

        return SmallVector<Value>{out};
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

  Value lhs = getLhs();
  Value rhs = getRhs();
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  ShapedType resultType;
  if (getResult())
    resultType = getResult().getType();
  else
    resultType = getOut().getType();

  if (!lhsType || !rhsType || !resultType)
    return DiagnosedSilenceableFailure::definiteFailure();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return DiagnosedSilenceableFailure::definiteFailure();

  const int64_t B = lhsType.getDimSize(0);
  const int64_t M = lhsType.getDimSize(1);
  const int64_t K = lhsType.getDimSize(2);
  if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
      ShapedType::isDynamic(K))
    return DiagnosedSilenceableFailure::definiteFailure();

  const int64_t bTile = tilingFactors[0];
  const int64_t mTile = tilingFactors[1];
  const int64_t rTile = tilingFactors[2];

  Type elementTy = lhsType.getElementType();

  Value resultInit =
      tensor::EmptyOp::create(builder, loc, resultType.getShape(), elementTy);
  TypedAttr zeroAttr = builder.getZeroAttr(elementTy);

  Value Bc = arith::ConstantIndexOp::create(builder, loc, B);
  Value Mc = arith::ConstantIndexOp::create(builder, loc, M);
  Value Kc = arith::ConstantIndexOp::create(builder, loc, K);
  Value bTileC = arith::ConstantIndexOp::create(builder, loc, bTile);
  Value mTileC = arith::ConstantIndexOp::create(builder, loc, mTile);
  Value rTileC = arith::ConstantIndexOp::create(builder, loc, rTile);

  SmallVector<Value> finals = createNestedScfForLoops(
      builder, loc, ArrayRef<int64_t>{B, M}, ArrayRef<int64_t>{bTile, mTile},
      ValueRange{resultInit},
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value iB = ivs[0];
        Value iM = ivs[1];

        Value remB = arith::SubIOp::create(b, loc2, Bc, iB);
        Value remM = arith::SubIOp::create(b, loc2, Mc, iM);
        Value useB = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remB, bTileC);
        Value useM = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remM, mTileC);
        Value bTileDyn = arith::SelectOp::create(b, loc2, useB, bTileC, remB);
        Value mTileDyn = arith::SelectOp::create(b, loc2, useM, mTileC, remM);

        Value zeroScalar = arith::ConstantOp::create(b, loc2, zeroAttr);
        Value accEmpty = tensor::EmptyOp::create(
            b, loc2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),
            elementTy, ValueRange{bTileDyn, mTileDyn});
        Value acc0 = linalg::FillOp::create(b, loc2, ValueRange{zeroScalar},
                                            ValueRange{accEmpty})
                         .getResult(0);

        SmallVector<Value> red = createNestedScfForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{rTile},
            ValueRange{acc0},
            [&](OpBuilder &b2, Location loc3, ValueRange redIvs,
                ValueRange accArgs) -> SmallVector<Value> {
              Value k = redIvs[0];
              Value remK = arith::SubIOp::create(b2, loc3, Kc, k);
              Value useR = arith::CmpIOp::create(
                  b2, loc3, arith::CmpIPredicate::ugt, remK, rTileC);
              Value kTileDyn =
                  arith::SelectOp::create(b2, loc3, useR, rTileC, remK);

              auto lhsTileTy = RankedTensorType::get({ShapedType::kDynamic,
                                                      ShapedType::kDynamic,
                                                      ShapedType::kDynamic},
                                                     elementTy);
              Value lhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, lhsTileTy, lhs, ValueRange{iB, iM, k},
                  ValueRange{bTileDyn, mTileDyn, kTileDyn}, ValueRange{},
                  noStaticOffsets3,
                  ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                                     ShapedType::kDynamic}),
                  unitStrides3);

              auto rhsTileTy = RankedTensorType::get(
                  {ShapedType::kDynamic, ShapedType::kDynamic}, elementTy);
              Value rhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, rhsTileTy, rhs, ValueRange{iB, k},
                  ValueRange{bTileDyn, kTileDyn}, ValueRange{},
                  noStaticOffsets2,
                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),
                  unitStrides2);

              auto tileGemv = cinm::BatchGemvOp::create(b2, loc3, lhsSlice,
                                                        rhsSlice, accArgs[0]);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, tileGemv.getResult(), accArgs[0]);
              return SmallVector<Value>{mat.getResult()};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iterArgs[0], ValueRange{iB, iM},
            ValueRange{bTileDyn, mTileDyn}, ValueRange{}, noStaticOffsets2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),
            unitStrides2);

        return SmallVector<Value>{out};
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
        Value outBuf;
        if (auto outmemref = getOut()) {
          outBuf = extractSlice1D(
              b, loc2, cast<TypedValue<ShapedType>>(outmemref), pM, i);
          if (biasSlice)
            linalg::AddOp::create(b, loc2, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

        ValueRange innerIterArgInit{};
        if (!outBuf) {
          Value initSlice;
          if (biasSlice) {
            initSlice = biasSlice;
          } else {
            auto reductionAccTy = RankedTensorType::get({pM}, elTy);
            DenseElementsAttr zeros;
            if (auto floatType = dyn_cast<FloatType>(elTy))
              zeros = DenseElementsAttr::get(
                  reductionAccTy,
                  {APFloat::getZero(floatType.getFloatSemantics())});
            else
              zeros = DenseElementsAttr::get(
                  reductionAccTy,
                  {APInt::getZero(reductionAccTy.getElementTypeBitWidth())});
            initSlice = arith::ConstantOp::create(b, loc2, zeros).getResult();
          }
          innerIterArgInit = tensor::InsertSliceOp::create(
                                 b, loc2, initSlice, iters[0], ValueRange{i},
                                 ValueRange{}, ValueRange{}, noStaticOffsets1,
                                 ArrayRef<int64_t>({pM}), unitStrides1)
                                 .getResult();
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

              if (outBuf) {
                cinm::GemvOp::create(b2, loc3, aTile, xTile, Value{}, outBuf);
                return {};
              }
              Value accSlice = extractSlice1D(
                  b2, loc3, cast<TypedValue<ShapedType>>(innerAccArgs[0]), pM,
                  i);
              Value tileResult = cinm::GemvOp::create(b2, loc3, aTile, xTile,
                                                      accSlice, Value{})
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

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
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
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/ValueRange.h>
#include <tuple>

using namespace mlir;
using namespace mlir::cinm;

using TilingResult2 = FailureOr<SmallVector<Value>>;

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

TilingResult2 ReduceOp::convertToTiledOps(OpBuilder &builder,
                                          TilingParameters params) {
  auto ty = getInput().getType();
  auto reduceClusterSize =
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType());

  auto method = getMethod();
  if (method == ReduceMethod::ADD) {
    return TilingResult2(
        {createVectorReduceAdd(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MUL) {
    return TilingResult2(
        {createVectorReduceMul(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MAX) {
    return TilingResult2(
        {createVectorReduceMax(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MIN) {
    return TilingResult2(
        {createVectorReduceMin(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else {
    abort();
  }
}

static constexpr std::array<int64_t, 2> noStaticOffsets2{ShapedType::kDynamic,
                                                         ShapedType::kDynamic};
static constexpr std::array<int64_t, 2> unitStrides2{1, 1};
static constexpr std::array<int64_t, 1> noStaticOffsets1{ShapedType::kDynamic};
static constexpr std::array<int64_t, 1> unitStrides1{1};
static constexpr std::array<int64_t, 3> noStaticOffsets3{
    ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};
static constexpr std::array<int64_t, 3> unitStrides3{1, 1, 1};

static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getGemmTilesFromAttributes(const int64_t M, const int64_t N, const int64_t K,
                           const Type eltType,
                           const mlir::cinm::TilingParameters &params,
                           Operation *errorLoc) {

  int64_t p0 = 0, p1 = 0;
  if (auto providedPar = params.getProvidedParallelTiles()) {
    std::tie(p0, p1) = *providedPar;
    if (p0 <= 0 || p1 <= 0)
      return errorLoc->emitError("Provided M, N tile sizes (")
             << p0 << ", " << p1 << ") are invalid";
  } else {
    auto parallelTileSizes = params.parallelClusterSize(M, N);
    if (!parallelTileSizes)
      return errorLoc->emitError("Cannot determine tiling factors for M=")
             << M << ", N=" << N << " and working group shape "
             << params.workgroupShape
             << ", provide tileSizes attribute [tM,tN,tK].";
    std::tie(p0, p1) = *parallelTileSizes;
  }

  int64_t r = 0;
  if (auto providedR = params.getProvidedReductionTile()) {
    r = *providedR;
    if (r <= 0 || r > K) {
      return errorLoc->emitError("Provided K tile size (")
             << r << ") incompatible with dim size K=" << K;
    }

    const int64_t maxElems = params.maxNumElementsOfType(eltType);
    const int64_t maxSizePerBuffer = (maxElems - 1) / 2;
    if (r > maxSizePerBuffer)
      return errorLoc->emitError("Provided K tile size (")
             << r << ") incompatible with max buffer size of "
             << maxSizePerBuffer << " " << eltType;
  } else {
    // Size of the tile on the reduction dimension.
    r = params.reduceClusterSize(2, K, eltType,
                                 /*extraElements=*/1);
  }

  return std::make_tuple(p0, p1, r);
}

static FailureOr<std::tuple<int64_t, int64_t, int64_t, int64_t>>
getBatchGemmTilesFromAttributes(const ShapedType &lhsType,
                                const ShapedType &rhsType,
                                const mlir::cinm::TilingParameters &params) {
  auto tiles = params.getTileSizes();
  if (!tiles || tiles->size() < 4)
    return failure();
  int64_t bTile = (*tiles)[0];
  int64_t mTile = (*tiles)[1];
  int64_t nTile = (*tiles)[2];
  int64_t rTile = (*tiles)[3];
  if (bTile <= 0 || mTile <= 0 || nTile <= 0 || rTile <= 0)
    return failure();

  const int64_t K = lhsType.getDimSize(2);
  if (!ShapedType::isDynamic(K) && rTile > K)
    return failure();

  const int64_t maxElems =
      params.maxNumElementsOfType(lhsType.getElementType());
  const int64_t maxPerBuffer = (maxElems - 1) / 2;
  if (rTile > maxPerBuffer)
    return failure();

  return std::make_tuple(bTile, mTile, nTile, rTile);
}

static FailureOr<std::tuple<int64_t, int64_t>>
getGemvTilesFromAttributes(const int64_t M, const int64_t K, const Type eltType,
                           const mlir::cinm::TilingParameters &params,
                           Operation *errorLoc) {

  int64_t p = 0, k = 0;
  if (auto provided = params.getTileSizes()) {
    if (provided->size() == 2) {
      p = (*provided)[0];
      k = (*provided)[1];
    } else {
      return errorLoc->emitError("Need two tile sizes for GEMV, provided ")
             << *provided;
    }
    if (p <= 0 || k <= 0) {
      return errorLoc->emitError("Invalid tile sizes for GEMV <")
             << M << "x" << K << "> : " << p << ", " << k;
    }
    if ((ShapedType::isStatic(M) && M % p) ||
        (ShapedType::isStatic(K) && K % k)) {
      return errorLoc->emitError("Invalid tile sizes for GEMV <")
             << M << "x" << K << "> : " << p << ", " << k;
    }
    return std::make_tuple(p, k);
  }
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K)) {
    return errorLoc->emitError("CINM cannot determine tiling factors for dynamic dimensions, provide "
                               "tileSizes attribute [tM,tK]");
  }

  auto parallelTileSize = params.parallelClusterSize(M, 1);
  if (!parallelTileSize)
    return errorLoc->emitError("Cannot determine tiling factors for M=")
           << M << " and working group shape " << params.workgroupShape
           << ", provide tileSizes attribute [tM,tK].";
  std::tie(p, std::ignore) = *parallelTileSize;

  // Size of the tile on the reduction dimension.
  k = params.reduceClusterSize(2, K, eltType,
                               /*extraElements=*/1);

  return std::make_tuple(p, k);
}

static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getBatchGemvTilesFromAttributes(const ShapedType &lhsType,
                                const ShapedType &rhsType,
                                const mlir::cinm::TilingParameters &params) {
  auto tiles = params.getTileSizes();
  if (!tiles || tiles->size() < 3)
    return failure();
  int64_t bTile = (*tiles)[0];
  int64_t mTile = (*tiles)[1];
  int64_t rTile = (*tiles)[2];
  if (bTile <= 0 || mTile <= 0 || rTile <= 0)
    return failure();
  const int64_t K = lhsType.getDimSize(2);
  if (!ShapedType::isDynamic(K) && rTile > K)
    return failure();
  const int64_t maxElems =
      params.maxNumElementsOfType(lhsType.getElementType());
  const int64_t maxPerBuffer = (maxElems - 1) / 2;
  if (rTile > maxPerBuffer)
    return failure();
  return std::make_tuple(bTile, mTile, rTile);
}

TilingResult2 ElementwiseOp::convertToTiledOps(OpBuilder &builder0,
                                               TilingParameters params) {
  ImplicitLocOpBuilder builder(getLoc(), builder0);
  const bool isUnaryOp = !getRhs();

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();

  ShapedType tensorTy = cast<ShapedType>(lhs.getType());
  auto shape = tensorTy.getShape();
  const ShapedType originalType = tensorTy;
  Value originalShapeValue;
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
    tensorTy = lhs.getType();
  }

  int64_t tileSize = 0;
  if (auto provided = params.getTileSizes()) {
    if (!provided->empty())
      tileSize = (*provided)[0];
  }
  if (tileSize <= 0) {
    emitError() << "elementwise tiling requires a positive tile size in "
                   "compute.tileSizes";
    return failure();
  }

  const int64_t numElements = tensorTy.getNumElements();
  tileSize = std::max<int64_t>(1, std::min<int64_t>(tileSize, numElements));

  Value resultInit = tensor::EmptyOp::create(builder, tensorTy, ValueRange{});

  SmallVector<Value> result = createNestedAffineForLoops(
      builder, getLoc(), {numElements}, {tileSize}, ValueRange{resultInit},
      [&](OpBuilder &b, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value base = indices[0];
        SmallVector<OpFoldResult, 1> off{base};

        SmallVector<OpFoldResult, 1> siz{b.getIndexAttr(tileSize)};
        SmallVector<OpFoldResult, 1> str{b.getI64IntegerAttr(1)};

        Value lhsSlice =
            tensor::ExtractSliceOp::create(b, loc, lhs, off, siz, str);

        Value rhsSlice = nullptr;
        if (!isUnaryOp) {
          rhsSlice = tensor::ExtractSliceOp::create(b, loc, rhs, off, siz, str);
        }

        ElementwiseOp smaller = ElementwiseOp::create(
            b, loc, getKind(), lhsSlice, rhsSlice, Value());
        markOpAsNoTile(smaller);

        if (smaller.getResult()) {
          Value subResult = tensor::InsertSliceOp::create(
              b, loc, smaller.getResult(), iterArgs[0], off, siz, str);
          return {subResult};
        } else {
          return {};
        }
      });

  if (originalType.getRank() > 1) {
    result[0] = tensor::ReshapeOp::create(builder, originalType, result[0],
                                          originalShapeValue);
  }
  return TilingResult2(result);
}

static constexpr std::array<int64_t, 2> noStaticOffsets{ShapedType::kDynamic,
                                                        ShapedType::kDynamic};

static constexpr std::array<int64_t, 2> unitStrides{1, 1};
static Value extractSlice(OpBuilder &builder, Location loc,
                          TypedValue<ShapedType> tensorOrMemref, int64_t a,
                          int64_t b, Value ia, Value ib) {

  const ShapedType sliceTy = tensorOrMemref.getType().clone({a, b});

  if (auto lhsTy =
          llvm::dyn_cast_or_null<RankedTensorType>(tensorOrMemref.getType())) {

    return builder.create<tensor::ExtractSliceOp>(
        loc, sliceTy, tensorOrMemref, ValueRange{ia, ib}, ValueRange{},
        ValueRange{}, ArrayRef(noStaticOffsets), sliceTy.getShape(),
        ArrayRef(unitStrides));
  } else if (auto lhsTy =
                 llvm::dyn_cast_or_null<MemRefType>(tensorOrMemref.getType())) {

    return memref::SubViewOp::create(
        builder, loc, tensorOrMemref, ArrayRef<OpFoldResult>{ia, ib},
        ArrayRef<OpFoldResult>{
            builder.getI64IntegerAttr(sliceTy.getShape()[0]),
            builder.getI64IntegerAttr(sliceTy.getShape()[1])},
        ArrayRef<OpFoldResult>{builder.getI64IntegerAttr(1),
                               builder.getI64IntegerAttr(1)});
  }
  assert(false && "type not handled");
}

namespace {

  struct GemmLikeAdaptor {

  };

}

static TilingResult2 convertGemmLikeToTiledOps(OpBuilder& op, GemmOp::Adaptor gemmlike, TilingParameters parms) {



}


TilingResult2 GemmOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  Location loc = getLoc();

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();

  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  ShapedType resultType;
  if (getResult()) {
    resultType = getResult().getType();
  } else {
    resultType = getOut().getType();
  }

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return failure();

  const int64_t M = lhsType.getDimSize(0);
  const int64_t K = lhsType.getDimSize(1);
  const int64_t N = rhsType.getDimSize(1);
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
      ShapedType::isDynamic(N))
    return failure();

  ValueRange initArgs{};
  if (!getOut()) {
    Value resultInit = tensor::EmptyOp::create(
        builder, loc, resultType.getShape(), resultType.getElementType());
    initArgs = resultInit;
  }

  int64_t p0, p1, r;
  if (auto tiles = getGemmTilesFromAttributes(M, N, K, lhsType.getElementType(),
                                              params, getOperation());
      succeeded(tiles)) {
    std::tie(p0, p1, r) = *tiles;
  } else {
    return failure();
  }

  Type eltTy = resultType.getElementType();

  SmallVector<Value> finals = createNestedAffineForLoops(
      builder, getLoc(), resultType.getShape(), {p0, p1}, initArgs,
      [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
                  ValueRange iterArgs) -> SmallVector<Value> {
        const auto parIndices = indices;
        const SmallVector<int64_t, 2> resultSizes{p0, p1};
        const ValueRange resultDynamicOffsets = parIndices;

        ValueRange iterArgInit{};
        Value biasSlice;
        if (auto bias = getBias()) {
          biasSlice = extractSlice(builder, loc, bias, p0, p1, parIndices[0],
                                   parIndices[1]);
        }
        Value outBuf;
        if (auto outmemref = getOut()) {
          outBuf = extractSlice(builder, loc,
                                cast<TypedValue<ShapedType>>(outmemref), p0, p1,
                                parIndices[0], parIndices[1]);
          if (biasSlice) {
            linalg::AddOp::create(builder, loc, ValueRange{biasSlice, outBuf},
                                  outBuf);
          }
        } else {
          // only for tensor-mode
          if (biasSlice) {
            iterArgInit = biasSlice;
          } else {
            auto reductionAccTy = RankedTensorType::get({p0, p1}, eltTy);
            DenseElementsAttr zeros;
            if (auto floatType =
                    dyn_cast<FloatType>(reductionAccTy.getElementType())) {
              zeros = DenseElementsAttr::get(
                  reductionAccTy,
                  {APFloat::getZero(floatType.getFloatSemantics())});
            } else {
              zeros = DenseElementsAttr::get(
                  reductionAccTy,
                  {APInt::getZero(reductionAccTy.getElementTypeBitWidth())});
            }

            iterArgInit =
                builder.create<arith::ConstantOp>(loc, zeros)->getResults();
          }
        }

        // this is the reduction loop
        SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
            builder, loc, {K}, {r}, iterArgInit,
            [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
                        ValueRange iterArgs) -> SmallVector<Value> {
              const auto indexInRedDim = indices[0];

              Value lhsSlice = extractSlice(builder, loc, lhs, p0, r,
                                            parIndices[0], indexInRedDim);

              Value rhsSlice = extractSlice(builder, loc, rhs, r, p1,
                                            indexInRedDim, parIndices[1]);
              Value bias;
              if (!getOut()) {
                // tensor mode
                bias = iterArgs[0];
              }

              auto tmpReduce = builder.create<cinm::GemmOp>(
                  loc, lhsSlice, rhsSlice, bias, outBuf);
              cinm::markOpAsNoTile(tmpReduce);
              if (outBuf) {
                return {};
              } else {
                return {tmpReduce.getResult()};
              }
            });

        if (getOut()) {
          return {};
        } else {
          const Value result = builder.create<tensor::InsertSliceOp>(
              loc, reductionResult[0], iterArgs[0], resultDynamicOffsets,
              ValueRange{}, ValueRange{}, ArrayRef(noStaticOffsets),
              resultSizes, ArrayRef(unitStrides));
          return {result};
        }
      });

  return TilingResult2(finals);
}

TilingResult2 BatchGemmOp::convertToTiledOps(OpBuilder &builder,
                                             TilingParameters params) {
  Location loc = getLoc();

  Value lhs = getLhs();
  Value rhs = getRhs();
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  ShapedType resultType;
  if (getResult()) {
    resultType = getResult().getType();
  } else {
    resultType = getOut().getType();
  }
  if (!lhsType || !rhsType || !resultType)
    return failure();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 3 ||
      resultType.getRank() != 3)
    return failure();

  const int64_t B = lhsType.getDimSize(0);
  const int64_t M = lhsType.getDimSize(1);
  const int64_t K = lhsType.getDimSize(2);
  const int64_t N = rhsType.getDimSize(2);
  if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
      ShapedType::isDynamic(K) || ShapedType::isDynamic(N))
    return failure();

  Type elementTy = lhsType.getElementType();

  int64_t bTile = 0, mTile = 0, nTile = 0, rTile = 0;
  if (auto provided = getBatchGemmTilesFromAttributes(lhsType, rhsType, params);
      succeeded(provided)) {
    std::tie(bTile, mTile, nTile, rTile) = *provided;
  } else {
    getOperation()->emitError()
        << "requires tileSizes attribute with [batch, M, N, K] entries";
    return failure();
  }

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
              cinm::markOpAsNoTile(tileGemm);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, tileGemm.getResult(), accArgs[0]);
              Value updatedAcc = mat.getResult();
              return SmallVector<Value>{updatedAcc};
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

  return TilingResult2(finals);
}

TilingResult2 BatchGemvOp::convertToTiledOps(OpBuilder &builder,
                                             TilingParameters params) {
  Location loc = getLoc();

  Value lhs = getLhs();
  Value rhs = getRhs();
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  ShapedType resultType;
  if (getResult()) {
    resultType = getResult().getType();
  } else {
    resultType = getOut().getType();
  }
  if (!lhsType || !rhsType || !resultType)
    return failure();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return failure();

  const int64_t B = lhsType.getDimSize(0);
  const int64_t M = lhsType.getDimSize(1);
  const int64_t K = lhsType.getDimSize(2);
  if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
      ShapedType::isDynamic(K))
    return failure();

  Type elementTy = lhsType.getElementType();

  int64_t bTile = 0, mTile = 0, rTile = 0;
  if (auto provided = getBatchGemvTilesFromAttributes(lhsType, rhsType, params);
      succeeded(provided)) {
    std::tie(bTile, mTile, rTile) = *provided;
  } else {
    getOperation()->emitError()
        << "requires tileSizes attribute with [batch, M, K] entries";
    return failure();
  }

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
              cinm::markOpAsNoTile(tileGemv);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, tileGemv.getResult(), accArgs[0]);
              Value updatedAcc = mat.getResult();
              return SmallVector<Value>{updatedAcc};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iterArgs[0], ValueRange{iB, iM},
            ValueRange{bTileDyn, mTileDyn}, ValueRange{}, noStaticOffsets2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),
            unitStrides2);

        return SmallVector<Value>{out};
      });

  return TilingResult2(finals);
}

TilingResult2 GemvOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  Location loc = getLoc();

  Value A = getLhs();
  Value x = getRhs();

  auto aTy = cast<ShapedType>(A.getType());
  auto xTy = cast<ShapedType>(x.getType());
  ShapedType yTy;
  if (getResult()) {
    yTy = getResult().getType();
  } else {
    yTy = getOut().getType();
  }
  if (aTy.getRank() != 2 || xTy.getRank() != 1 || yTy.getRank() != 1)
    return failure();
  if (aTy.getElementType() != xTy.getElementType() ||
      aTy.getElementType() != yTy.getElementType())
    return failure();

  const int64_t M = aTy.getDimSize(0);
  const int64_t K = aTy.getDimSize(1);

  auto tileSizes = getGemvTilesFromAttributes(M, K, aTy.getElementType(),
                                              params, getOperation());
  if (llvm::failed(tileSizes)) {
    return failure();
  }
  // Those are block sizes
  auto [pM, rK] = *tileSizes;

  Type elt = aTy.getElementType();


  // todo finish this
  // SmallVector<Value> finals = createNestedAffineForLoops(
  //     builder, getLoc(), resultType.getShape(), {p0, p1}, initArgs,
  //     [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
  //                 ValueRange iterArgs) -> SmallVector<Value> {

  //                 });


  Value init = tensor::EmptyOp::create(builder, loc, yTy.getShape(), elt);

  TypedAttr zeroAttr = builder.getZeroAttr(elt);

  Value Mc = arith::ConstantIndexOp::create(builder, loc, M);
  Value Kc = arith::ConstantIndexOp::create(builder, loc, K);
  Value pMc = arith::ConstantIndexOp::create(builder, loc, pM);
  Value rKc = arith::ConstantIndexOp::create(builder, loc, rK);

  SmallVector<Value> results = createNestedScfForLoops(
      builder, loc, ArrayRef<int64_t>{M}, ArrayRef<int64_t>{pM},
      ValueRange{init},
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iters) -> SmallVector<Value> {
        Value iM = ivs[0];

        Value remM = arith::SubIOp::create(b, loc2, Mc, iM);
        Value usePM = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                            remM, pMc);
        Value pMTile = arith::SelectOp::create(b, loc2, usePM, pMc, remM);

        Value zeroScalar = arith::ConstantOp::create(b, loc2, zeroAttr);
        Value accEmpty = tensor::EmptyOp::create(
            b, loc2, ArrayRef<int64_t>({ShapedType::kDynamic}), elt,
            ValueRange{pMTile});
        Value acc0 = linalg::FillOp::create(b, loc2, ValueRange{zeroScalar},
                                            ValueRange{accEmpty})
                         .getResult(0);

        SmallVector<Value> red = createNestedScfForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{rK},
            ValueRange{acc0},
            [&](OpBuilder &b2, Location loc3, ValueRange kIvs,
                ValueRange accArgs) -> SmallVector<Value> {
              Value k = kIvs[0];

              Value remK = arith::SubIOp::create(b2, loc3, Kc, k);
              Value useR = arith::CmpIOp::create(
                  b2, loc3, arith::CmpIPredicate::ugt, remK, rKc);
              Value kTile = arith::SelectOp::create(b2, loc3, useR, rKc, remK);

              auto aTileTy = RankedTensorType::get(
                  {ShapedType::kDynamic, ShapedType::kDynamic}, elt);
              Value aTile = tensor::ExtractSliceOp::create(
                  b2, loc3, aTileTy, A, ValueRange{iM, k},
                  ValueRange{pMTile, kTile}, ValueRange{},

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),
                  ArrayRef<int64_t>({1, 1}));

              auto xTileTy = RankedTensorType::get({ShapedType::kDynamic}, elt);
              Value xTile = tensor::ExtractSliceOp::create(
                  b2, loc3, xTileTy, x, ValueRange{k}, ValueRange{kTile},
                  ValueRange{}, ArrayRef<int64_t>({ShapedType::kDynamic}),
                  ArrayRef<int64_t>({ShapedType::kDynamic}),
                  ArrayRef<int64_t>({1}));

              auto yTileResTy =
                  RankedTensorType::get({ShapedType::kDynamic}, elt);
              auto gemv =
                  cinm::GemvOp::create(b2, loc3, aTile, xTile, accArgs[0]);
              cinm::markOpAsNoTile(gemv);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, gemv.getResult(), accArgs[0]);
              Value updatedAcc = mat.getResult();
              return SmallVector<Value>{updatedAcc};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iters[0], ValueRange{iM}, ValueRange{pMTile},
            ValueRange{}, ArrayRef<int64_t>({ShapedType::kDynamic}),
            ArrayRef<int64_t>({ShapedType::kDynamic}), ArrayRef<int64_t>({1}));

        return SmallVector<Value>{out};
      });

  return TilingResult2(results);
}

TilingResult2 ActivateOp::convertToTiledOps(OpBuilder &builder0,
                                            TilingParameters params) {
  ImplicitLocOpBuilder builder(getLoc(), builder0);
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
  int64_t p = 0;
  if (auto par = params.parallelClusterSize(total, 1))
    p = std::max<int64_t>(1, par->first);
  if (p <= 0)
    p = std::max<int64_t>(1, params.workingGroupSize());
  p = std::min<int64_t>(p, total);

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
        cinm::markOpAsNoTile(tile);

        Value out = tensor::InsertSliceOp::create(b, loc, tile.getResult(),
                                                  iters[0], off, siz, str);
        return {out};
      });

  if (originalTy.getRank() > 1) {
    finals[0] = tensor::ReshapeOp::create(builder, originalTy, finals[0],
                                          originalShapeValue);
  }
  return TilingResult2(finals);
}

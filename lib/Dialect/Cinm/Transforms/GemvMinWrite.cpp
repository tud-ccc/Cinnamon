//===- GemvMinWrite.cpp - cinm-gemv-min-write -------------------*- C++ -*-===//
//
// Transform nested GEMV loops so each row tile of A is consumed exactly once
// and the partial result is kept in SSA form until the tile is complete.
// The resulting loop structure is:
//   row-loop (carries whole result tensor)
//     reduction-loop (carries row accumulator)
//       column-loop (updates accumulator purely in SSA)
// After each row tile we issue a single tensor.insert_slice into the result.
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_CINMGEMVMINWRITEPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace mlir::cinm {
namespace {

using namespace mlir;

static Value makeZeroLike(OpBuilder &b, Location loc, Type elementType) {
  if (auto ft = dyn_cast<FloatType>(elementType))
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(ft, 0.0));
  if (auto it = dyn_cast<IntegerType>(elementType))
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(it, 0));
  return {};
}

//===----------------------------------------------------------------------===//
// Deep IV dependence and chain rebuilding helpers
//===----------------------------------------------------------------------===//

static bool ofrUsesValue(OpFoldResult ofr, Value v) {
  if (auto x = ofr.dyn_cast<Value>())
    return x == v;
  return false;
}

static bool anyMixedUsesValue(ArrayRef<OpFoldResult> values, Value v) {
  for (OpFoldResult ofr : values)
    if (ofrUsesValue(ofr, v))
      return true;
  return false;
}

static bool valueDependsOnIVDeep(Value val, Value iv) {
  SmallPtrSet<Operation *, 32> visited;
  SmallVector<Value, 16> worklist{val};
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;
    if (!visited.insert(def).second)
      continue;

    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def)) {
      if (anyMixedUsesValue(slice.getMixedOffsets(), iv) ||
          anyMixedUsesValue(slice.getMixedSizes(), iv) ||
          anyMixedUsesValue(slice.getMixedStrides(), iv))
        return true;
      worklist.push_back(slice.getSource());
      continue;
    }

    if (auto materialize =
            dyn_cast<bufferization::MaterializeInDestinationOp>(def)) {
      worklist.push_back(materialize.getSource());
      continue;
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
      worklist.push_back(collapse.getSrc());
      continue;
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      worklist.push_back(expand.getSrc());
      continue;
    }
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(def)) {
      worklist.push_back(reshape.getSource());
      worklist.push_back(reshape.getShape());
      continue;
    }
    if (auto cast = dyn_cast<tensor::CastOp>(def)) {
      worklist.push_back(cast.getSource());
      continue;
    }
    // Unknown producer -> stop chasing along this path.
  }
  return false;
}

static LogicalResult ensureHelperValue(Value value, IRMapping &mapper,
                                       OpBuilder &rewriter, Operation *scope,
                                       SmallPtrSetImpl<Operation *> &visited) {
  if (Value mapped = mapper.lookupOrNull(value))
    return success();

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    if (!owner || owner != scope)
      return success();
    if (Value mapped = mapper.lookupOrNull(blockArg))
      return success();
    return failure();
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return success();

  if (!scope->isAncestor(def))
    return success();

  if (!visited.insert(def).second)
    return success();

  if (def->getNumRegions() != 0 || def->hasTrait<OpTrait::IsTerminator>())
    return failure();

  for (Value operand : def->getOperands())
    if (failed(ensureHelperValue(operand, mapper, rewriter, scope, visited)))
      return failure();

  Operation *clone = rewriter.clone(*def, mapper);
  for (auto [oldResult, newResult] :
       llvm::zip(def->getResults(), clone->getResults()))
    mapper.map(oldResult, newResult);

  return success();
}

static FailureOr<Value>
rebuildSliceChain(Value root, Value oldRow, Value newRow, Value oldRed,
                  Value newRed, Value oldCol, Value newCol, Value newRowSize,
                  Value newRedSize, Value newColSize, OpBuilder &rewriter,
                  Location loc, Operation *scope, IRMapping *mapper = nullptr) {
  SmallVector<Operation *, 16> chain;
  Value cur = root;
  if (mapper)
    if (Value mapped = mapper->lookupOrNull(cur))
      cur = mapped;
  SmallPtrSet<Operation *, 32> seen;
  while (Operation *def = cur.getDefiningOp()) {
    if (!seen.insert(def).second)
      break;
    if (!isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp,
             tensor::ExpandShapeOp, tensor::ReshapeOp, tensor::CastOp,
             bufferization::MaterializeInDestinationOp>(def))
      break;
    chain.push_back(def);

    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def)) {
      cur = slice.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
      cur = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      cur = expand.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(def)) {
      cur = reshape.getSource();
      continue;
    }
    if (auto cast = dyn_cast<tensor::CastOp>(def)) {
      cur = cast.getSource();
      continue;
    }
    if (auto materialize =
            dyn_cast<bufferization::MaterializeInDestinationOp>(def)) {
      cur = materialize.getSource();
      continue;
    }
  }

  Value rebuilt = cur;
  SmallPtrSet<Operation *, 16> clonedHelpers;
  for (Operation *op : llvm::reverse(chain)) {
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
      auto remap =
          [&](ArrayRef<OpFoldResult> mixed,
              bool forSize) -> FailureOr<SmallVector<OpFoldResult, 8>> {
        SmallVector<OpFoldResult, 8> result;
        result.reserve(mixed.size());
        for (OpFoldResult ofr : mixed) {
          if (auto value = ofr.dyn_cast<Value>()) {
            if (oldRow && newRow && value == oldRow) {
              result.push_back(newRow);
              continue;
            }
            if (oldRed && newRed && value == oldRed) {
              result.push_back(newRed);
              continue;
            }
            if (oldCol && newCol && value == oldCol) {
              result.push_back(newCol);
              continue;
            }
            if (forSize && oldRow && newRowSize &&
                valueDependsOnIVDeep(value, oldRow)) {
              result.push_back(newRowSize);
              continue;
            }
            if (forSize && oldRed && newRedSize &&
                valueDependsOnIVDeep(value, oldRed)) {
              result.push_back(newRedSize);
              continue;
            }
            if (forSize && oldCol && newColSize &&
                valueDependsOnIVDeep(value, oldCol)) {
              result.push_back(newColSize);
              continue;
            }
            if (mapper)
              if (Value mapped = mapper->lookupOrNull(value)) {
                result.push_back(mapped);
                continue;
              }

            if (scope) {
              bool insideScope = false;
              if (auto blockArg = dyn_cast<BlockArgument>(value)) {
                insideScope = blockArg.getOwner()->getParentOp() == scope;
              } else if (Operation *def = value.getDefiningOp()) {
                insideScope = scope->isAncestor(def);
              }

              if (!insideScope) {
                result.push_back(value);
                continue;
              }
            }

            if (!mapper)
              return failure();

            if (failed(ensureHelperValue(value, *mapper, rewriter, scope,
                                         clonedHelpers)))
              return failure();
            if (Value mapped = mapper->lookupOrNull(value)) {
              result.push_back(mapped);
              continue;
            }
            return failure();
          }
          result.push_back(ofr);
        }
        return result;
      };

      FailureOr<SmallVector<OpFoldResult, 8>> offsets =
          remap(slice.getMixedOffsets(), /*forSize=*/false);
      if (failed(offsets))
        return failure();
      FailureOr<SmallVector<OpFoldResult, 8>> sizes =
          remap(slice.getMixedSizes(), /*forSize=*/true);
      if (failed(sizes))
        return failure();
      FailureOr<SmallVector<OpFoldResult, 8>> strides =
          remap(slice.getMixedStrides(), /*forSize=*/false);
      if (failed(strides))
        return failure();

      rebuilt = rewriter.create<tensor::ExtractSliceOp>(loc, rebuilt, *offsets,
                                                        *sizes, *strides);
      continue;
    }

    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      rebuilt = rewriter.create<tensor::CollapseShapeOp>(
          loc, collapse.getResultType(), rebuilt, collapse.getReassociation());
      continue;
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      if (!expand.getOutputShape().empty() ||
          expand.getStaticOutputShapeAttr()) {
        SmallVector<Value> mappedShape;
        if (auto outputs = expand.getOutputShape(); !outputs.empty()) {
          mappedShape.reserve(outputs.size());
          for (Value v : outputs) {
            if (mapper)
              if (Value mapped = mapper->lookupOrNull(v)) {
                mappedShape.push_back(mapped);
                continue;
              }
            mappedShape.push_back(v);
          }
        }
        rebuilt = rewriter.create<tensor::ExpandShapeOp>(
            loc, expand.getResultType(), rebuilt, expand.getReassociationAttr(),
            mappedShape, expand.getStaticOutputShapeAttr());
      } else {
        rebuilt = rewriter.create<tensor::ExpandShapeOp>(
            loc, expand.getResultType(), rebuilt,
            expand.getReassociationIndices());
      }
      continue;
    }
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(op)) {
      Value mappedShape = reshape.getShape();
      if (mapper) {
        if (Value remapped = mapper->lookupOrNull(mappedShape)) {
          mappedShape = remapped;
        } else {
          if (failed(ensureHelperValue(mappedShape, *mapper, rewriter, scope,
                                       clonedHelpers)))
            return failure();
          if (Value remapped = mapper->lookupOrNull(mappedShape))
            mappedShape = remapped;
        }
      }
      rebuilt = rewriter.create<tensor::ReshapeOp>(loc, reshape.getResultType(),
                                                   rebuilt, mappedShape);
      continue;
    }
    if (auto cast = dyn_cast<tensor::CastOp>(op)) {
      rebuilt = rewriter.create<tensor::CastOp>(loc, cast.getType(), rebuilt);
      continue;
    }
    // Drop materialize_in_destination, we only need its source.
  }
  return rebuilt;
}

//===----------------------------------------------------------------------===//
// Pattern detection
//===----------------------------------------------------------------------===//

struct GemvNest {
  scf::ForOp outer;
  scf::ForOp row;
  scf::ForOp col;
  scf::ForOp red;
  cinm::GemvOp gemv;
};

static scf::ForOp getSoleNestedFor(Block *body) {
  scf::ForOp found;
  for (Operation &op : body->without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (found)
        return scf::ForOp();
      found = forOp;
    }
  }
  return found;
}

static FailureOr<GemvNest> matchTripleNestUnderCompute(ComputeOp compute) {
  Block &entry = compute.getBody().front();

  for (Operation &candidate : entry) {
    auto outer = dyn_cast<scf::ForOp>(candidate);
    if (!outer)
      continue;
    if (outer.getInitArgs().size() != 1 ||
        !isa<RankedTensorType>(outer.getInitArgs()[0].getType()))
      continue;

    scf::ForOp middle = nullptr;
    scf::ForOp inner = nullptr;
    if ((middle = getSoleNestedFor(outer.getBody())) &&
        (inner = getSoleNestedFor(middle.getBody()))) {
      // Recognize gemv inside inner loop.
      cinm::GemvOp gemv;
      bool multiGemv = false;
      inner.walk([&](cinm::GemvOp op) {
        if (!gemv)
          gemv = op;
        else
          multiGemv = true;
      });
      if (!gemv || multiGemv)
        continue;

      auto behavesAs = [&](scf::ForOp rowLoop, scf::ForOp colLoop) {
        Value aVal = gemv.getLhs();
        Value bVal = gemv.getRhs();
        Value rowIV = rowLoop.getInductionVar();
        Value colIV = colLoop.getInductionVar();
        bool aRow = valueDependsOnIVDeep(aVal, rowIV);
        bool aCol = valueDependsOnIVDeep(aVal, colIV);
        bool bCol = valueDependsOnIVDeep(bVal, colIV);
        return aRow && !aCol && bCol;
      };

      if (behavesAs(outer, middle))
        return GemvNest{outer, outer, middle, inner, gemv};
      if (behavesAs(middle, outer))
        return GemvNest{outer, middle, outer, inner, gemv};
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Rewrite implementation
//===----------------------------------------------------------------------===//

static FailureOr<Value> buildRowCentric(GemvNest &nest, IRMapping &mapper,
                                        OpBuilder &rewriter, Block *destBlock,
                                        ComputeOp compute) {
  Location loc = compute.getLoc();
  auto resTy = dyn_cast<RankedTensorType>(nest.outer.getResult(0).getType());
  if (!resTy || resTy.getRank() != 2)
    return compute.emitOpError("gemv-min-write expects rank-2 tensor results"),
           failure();

  Type elemTy = resTy.getElementType();

  auto mapVal = [&](Value v) -> FailureOr<Value> {
    if (Value mapped = mapper.lookupOrNull(v))
      return mapped;

    bool insideScope = false;
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      insideScope = blockArg.getOwner()->getParentOp() == compute;
    } else if (Operation *def = v.getDefiningOp()) {
      insideScope = compute->isAncestor(def);
    }

    if (!insideScope)
      return v;

    SmallPtrSet<Operation *, 16> localVisited;
    if (failed(ensureHelperValue(v, mapper, rewriter, compute, localVisited)))
      return failure();

    if (Value mapped = mapper.lookupOrNull(v))
      return mapped;

    return failure();
  };

  auto clampToStep = [&](Value upper, Value iv, Value step) -> Value {
    Value remaining = rewriter.create<arith::SubIOp>(loc, upper, iv);
    Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                               remaining, step);
    return rewriter.create<arith::SelectOp>(loc, cmp, step, remaining);
  };

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(destBlock);

  FailureOr<Value> initResOr = mapVal(nest.outer.getInitArgs()[0]);
  if (failed(initResOr))
    return failure();
  Value initRes = *initResOr;

  FailureOr<Value> rowLBOr = mapVal(nest.row.getLowerBound());
  if (failed(rowLBOr))
    return failure();
  Value rowLB = *rowLBOr;

  FailureOr<Value> rowUBOr = mapVal(nest.row.getUpperBound());
  if (failed(rowUBOr))
    return failure();
  Value rowUB = *rowUBOr;

  FailureOr<Value> rowStOr = mapVal(nest.row.getStep());
  if (failed(rowStOr))
    return failure();
  Value rowSt = *rowStOr;

  auto ensureYield = [&](scf::ForOp loop) {
    Block *body = loop.getBody();
    if (!body->empty())
      return;
    OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToEnd(body);
    if (loop.getNumResults() == 0)
      rewriter.create<scf::YieldOp>(loc);
    else
      rewriter.create<scf::YieldOp>(loc, body->getArgument(1));
  };

  scf::ForOp newRow = rewriter.create<scf::ForOp>(loc, rowLB, rowUB, rowSt,
                                                  ValueRange{initRes});
  ensureYield(newRow);

  Block *rowBody = newRow.getBody();
  rewriter.setInsertionPointToStart(rowBody);
  Value ivRow = newRow.getInductionVar();
  Value resIn = rowBody->getArgument(1);

  mapper.map(nest.row.getInductionVar(), ivRow);
  if (nest.row.getBody()->getNumArguments() > 1)
    mapper.map(nest.row.getBody()->getArgument(1), resIn);

  auto getDimValue = [&](int64_t dim) -> Value {
    if (resTy.isDynamicDim(dim))
      return rewriter.create<tensor::DimOp>(loc, resIn, dim);
    return rewriter.create<arith::ConstantIndexOp>(loc, resTy.getDimSize(dim));
  };

  Value fullColsVal;
  OpFoldResult fullCols;
  if (resTy.isDynamicDim(1)) {
    fullColsVal = getDimValue(1);
    fullCols = OpFoldResult(fullColsVal);
  } else {
    fullCols = OpFoldResult(rewriter.getIndexAttr(resTy.getDimSize(1)));
  }

  Value rowSize = clampToStep(rowUB, ivRow, rowSt);

  OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
  OpFoldResult oneAttr = rewriter.getIndexAttr(1);

  Value zeroVal = makeZeroLike(rewriter, loc, elemTy);
  if (!zeroVal)
    return compute.emitOpError("gemv-min-write: unsupported element type"),
           failure();

  SmallVector<OpFoldResult, 2> tileShape{OpFoldResult(rowSize), fullCols};
  Value tileEmpty = rewriter.create<tensor::EmptyOp>(loc, tileShape, elemTy);
  Value tileZero =
      rewriter.create<linalg::FillOp>(loc, zeroVal, tileEmpty).getResult(0);

  FailureOr<Value> redLBOr = mapVal(nest.red.getLowerBound());
  if (failed(redLBOr))
    return failure();
  Value redLB = *redLBOr;

  FailureOr<Value> redUBOr = mapVal(nest.red.getUpperBound());
  if (failed(redUBOr))
    return failure();
  Value redUB = *redUBOr;

  FailureOr<Value> redStOr = mapVal(nest.red.getStep());
  if (failed(redStOr))
    return failure();
  Value redSt = *redStOr;

  scf::ForOp newRed = rewriter.create<scf::ForOp>(loc, redLB, redUB, redSt,
                                                  ValueRange{tileZero});
  ensureYield(newRed);
  Block *redBody = newRed.getBody();
  rewriter.setInsertionPointToStart(redBody);
  Value ivRed = newRed.getInductionVar();
  Value accIn = redBody->getArgument(1);

  mapper.map(nest.red.getInductionVar(), ivRed);
  if (nest.red.getBody()->getNumArguments() > 1)
    mapper.map(nest.red.getBody()->getArgument(1), accIn);

  Value redSize = clampToStep(redUB, ivRed, redSt);

  FailureOr<Value> aHoisted = rebuildSliceChain(
      nest.gemv.getLhs(),
      /*oldRow*/ nest.row.getInductionVar(), /*newRow*/ ivRow,
      /*oldRed*/ nest.red.getInductionVar(), /*newRed*/ ivRed,
      /*oldCol*/ nest.col.getInductionVar(), /*newCol*/ Value(),
      /*newRowSize*/ rowSize,
      /*newRedSize*/ redSize,
      /*newColSize*/ Value(), rewriter, loc, compute, &mapper);
  if (failed(aHoisted))
    return failure();

  FailureOr<Value> colLBOr = mapVal(nest.col.getLowerBound());
  if (failed(colLBOr))
    return failure();
  Value colLB = *colLBOr;

  FailureOr<Value> colUBOr = mapVal(nest.col.getUpperBound());
  if (failed(colUBOr))
    return failure();
  Value colUB = *colUBOr;

  FailureOr<Value> colStOr = mapVal(nest.col.getStep());
  if (failed(colStOr))
    return failure();
  Value colSt = *colStOr;

  scf::ForOp newCol =
      rewriter.create<scf::ForOp>(loc, colLB, colUB, colSt, ValueRange{accIn});
  ensureYield(newCol);
  Block *colBody = newCol.getBody();
  rewriter.setInsertionPointToStart(colBody);
  Value ivCol = newCol.getInductionVar();
  Value accCol = colBody->getArgument(1);

  mapper.map(nest.col.getInductionVar(), ivCol);
  if (nest.col.getBody()->getNumArguments() > 1)
    mapper.map(nest.col.getBody()->getArgument(1), accCol);

  Value colSize = clampToStep(colUB, ivCol, colSt);

  FailureOr<Value> bRebuilt = rebuildSliceChain(
      nest.gemv.getRhs(),
      /*oldRow*/ nest.row.getInductionVar(), /*newRow*/ ivRow,
      /*oldRed*/ nest.red.getInductionVar(), /*newRed*/ ivRed,
      /*oldCol*/ nest.col.getInductionVar(), /*newCol*/ ivCol,
      /*newRowSize*/ Value(),
      /*newRedSize*/ redSize,
      /*newColSize*/ colSize, rewriter, loc, compute, &mapper);
  if (failed(bRebuilt))
    return failure();

  Value newBias;
  if (Value bias = nest.gemv.getBias()) {
    Value oldRedAcc;
    if (nest.red.getBody()->getNumArguments() > 1)
      oldRedAcc = nest.red.getBody()->getArgument(1);
    if (!oldRedAcc || bias != oldRedAcc) {
      FailureOr<Value> rebuiltBias = rebuildSliceChain(
          bias,
          /*oldRow*/ nest.row.getInductionVar(), /*newRow*/ ivRow,
          /*oldRed*/ nest.red.getInductionVar(), /*newRed*/ ivRed,
          /*oldCol*/ nest.col.getInductionVar(), /*newCol*/ ivCol,
          /*newRowSize*/ Value(),
          /*newRedSize*/ redSize,
          /*newColSize*/ colSize, rewriter, loc, compute, &mapper);
      if (failed(rebuiltBias))
        return failure();
      newBias = *rebuiltBias;
    }
  }

  SmallVector<OpFoldResult, 2> accOff{zeroAttr, OpFoldResult(ivCol)};
  SmallVector<OpFoldResult, 2> accSz{OpFoldResult(rowSize),
                                     OpFoldResult(colSize)};
  SmallVector<OpFoldResult, 2> accStr{oneAttr, oneAttr};
  Value accChunk2D = rewriter.create<tensor::ExtractSliceOp>(
      loc, accCol, accOff, accSz, accStr);
  SmallVector<ReassociationIndices, 1> collapse{{0, 1}};
  Value accChunk = rewriter.create<tensor::CollapseShapeOp>(
      loc, nest.gemv.getResult().getType(), accChunk2D, collapse);

  Value biasForGemv = accChunk;
  if (newBias)
    biasForGemv = rewriter
                      .create<cinm::ElementwiseOp>(
                          loc, cinm::ElementwiseKind::Add, biasForGemv, newBias)
                      .getResult();

  auto newGemv =
      rewriter.create<cinm::GemvOp>(loc, *aHoisted, *bRebuilt, biasForGemv);
  newGemv->setAttrs(nest.gemv->getAttrDictionary());
  Value gemvResult = newGemv.getResult();

  RankedTensorType expandedType = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, elemTy);
  Value reshapeShape = rewriter.create<tensor::FromElementsOp>(
      loc, ValueRange{rowSize, colSize});
  Value expanded = rewriter.create<tensor::ReshapeOp>(loc, expandedType,
                                                      gemvResult, reshapeShape);

  Value accUpd = rewriter.create<tensor::InsertSliceOp>(loc, expanded, accCol,
                                                        accOff, accSz, accStr);

  auto colYield = cast<scf::YieldOp>(colBody->getTerminator());
  rewriter.setInsertionPoint(colYield);
  rewriter.create<scf::YieldOp>(loc, accUpd);
  colYield.erase();

  auto redYield = cast<scf::YieldOp>(redBody->getTerminator());
  rewriter.setInsertionPoint(redYield);
  rewriter.create<scf::YieldOp>(loc, newCol.getResult(0));
  redYield.erase();

  rewriter.setInsertionPointAfter(newRed);
  SmallVector<OpFoldResult, 2> finalOff{OpFoldResult(ivRow), zeroAttr};
  SmallVector<OpFoldResult, 2> finalSz{OpFoldResult(rowSize), fullCols};
  SmallVector<OpFoldResult, 2> finalStr{oneAttr, oneAttr};
  Value resOut = rewriter.create<tensor::InsertSliceOp>(
      loc, newRed.getResult(0), resIn, finalOff, finalSz, finalStr);

  auto rowYield = cast<scf::YieldOp>(rowBody->getTerminator());
  rewriter.setInsertionPoint(rowYield);
  rewriter.create<scf::YieldOp>(loc, resOut);
  rowYield.erase();

  return newRow.getResult(0);
}

//===----------------------------------------------------------------------===//
// Pass plumbing
//===----------------------------------------------------------------------===//

static LogicalResult rewriteCompute(cinm::ComputeOp compute,
                                    OpBuilder &rewriter) {
  if (compute.getNumResults() == 0)
    return failure();

  FailureOr<GemvNest> matched = matchTripleNestUnderCompute(compute);
  if (failed(matched))
    return failure();

  GemvNest &nest = *matched;
  if (compute.getNumResults() != 1)
    return failure();

  Location loc = compute.getLoc();
  rewriter.setInsertionPoint(compute);
  auto newCompute =
      rewriter.create<cinm::ComputeOp>(loc, compute.getResultTypes());
  newCompute->setAttrs(compute->getAttrDictionary());

  Block &oldBody = compute.getBody().front();
  Block &newBody = newCompute.getBody().front();

  IRMapping mapper;
  for (auto [oldArg, newArg] :
       llvm::zip(oldBody.getArguments(), newBody.getArguments()))
    mapper.map(oldArg, newArg);
  rewriter.setInsertionPointToEnd(&newBody);
  for (Operation &op : oldBody) {
    if (&op == nest.outer.getOperation())
      break;
    if (isa<cinm::YieldOp>(op))
      continue;
    rewriter.clone(op, mapper);
  }

  FailureOr<Value> newResult =
      buildRowCentric(nest, mapper, rewriter, &newBody, compute);
  if (failed(newResult)) {
    newCompute.erase();
    return failure();
  }

  rewriter.setInsertionPointToEnd(&newBody);
  rewriter.create<cinm::YieldOp>(loc, ValueRange{*newResult});

  compute.replaceAllUsesWith(newCompute.getResults());
  compute.erase();
  return success();
}

struct CinmGemvMinWritePass
    : public ::impl::CinmGemvMinWritePassBase<CinmGemvMinWritePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, arith::ArithDialect,
                    cinm::CinmDialect, bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    OpBuilder rewriter(&getContext());
    SmallVector<cinm::ComputeOp, 4> computes;
    getOperation()->walk(
        [&](cinm::ComputeOp compute) { computes.push_back(compute); });

    for (cinm::ComputeOp compute : computes)
      (void)rewriteCompute(compute, rewriter);
  }
};

} // namespace
} // namespace mlir::cinm

namespace mlir::cinm::impl {
std::unique_ptr<::mlir::Pass> createCinmGemvMinWritePass() {
  return std::make_unique<CinmGemvMinWritePass>();
}
} // namespace mlir::cinm::impl

namespace mlir::cinm {
std::unique_ptr<::mlir::Pass> createCinmGemvMinWritePass() {
  return impl::createCinmGemvMinWritePass();
}
} // namespace mlir::cinm

/// Implements the Cnm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h"
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmScatterMap.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <cinm-mlir/Dialect/Cnm/IR/CnmTypes.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/OpImplementation.h>

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Utils/MemRefUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "cnm-ops"

using namespace mlir;
using namespace mlir::cnm;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CnmDialect
//===----------------------------------------------------------------------===//

void CnmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.cpp.inc"
      >();
}

void DeclareBufferOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "cnm_buf");
}

::mlir::LogicalResult GatherOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    GatherOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  auto out = adaptor.getOutputBuf();
  if (isa<MemRefType>(out.getType())) {
    return success();
  } else if (isa<RankedTensorType>(out.getType())) {
    ShapedType ty = cast<RankedTensorType>(out.getType());

    inferredReturnShapes.push_back(ShapedTypeComponents(ty));
    return success();
  }

  return failure();
}

static void printShorthandBufferType(OpAsmPrinter &p, cnm::BufferType bufTy) {
  p << "<";
  for (auto dim : bufTy.getShape())
    p << dim << "x";
  p << bufTy.getElementType();
  if (auto level = bufTy.getLevel())
    p << ", " << level;
  p << ">";
}

void LaunchOp::print(OpAsmPrinter &p) {
  p << " " << getWg();

  auto bodyArgs = getBody().getArguments();
  auto printArgsList = [&](StringRef kw, ValueRange operands,
                           unsigned argOffset) {
    if (operands.empty())
      return;
    p << " " << kw << "(";
    llvm::interleaveComma(llvm::enumerate(operands), p, [&](auto indexed) {
      auto [i, operand] = indexed;
      p << bodyArgs[argOffset + i] << " = " << operand << " : ";
      printShorthandBufferType(p, cast<cnm::BufferType>(operand.getType()));
    });
    p << ")";
  };

  printArgsList("ins", getInputs(), 0);
  printArgsList("outs", getOutBuffers(), getInputs().size());

  p << " ";
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {"operandSegmentSizes"});

  p << "on ";
  p.printType(getWg().getType());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false, false);
}

namespace {
struct PartialBufferType {
  SmallVector<int64_t> dims;
  Type elementType;
  Attribute level;
};

} // namespace

static ParseResult parseShorthandBufferType(OpAsmParser &parser,
                                            PartialBufferType &result) {
  if (parser.parseLess() ||
      parser.parseDimensionList(result.dims, false, true) ||
      parser.parseType(result.elementType))
    return failure();
  if (parser.parseOptionalComma().succeeded())
    if (parser.parseAttribute(result.level))
      return failure();

  if (parser.parseGreater())
    return failure();
  return success();
}

static void
inflatePartialBufferTypes(WorkgroupType wgTy,

                          SmallVectorImpl<PartialBufferType> &partiaTypes,
                          SmallVectorImpl<Type> &result) {
  for (auto partial : partiaTypes) {
    result.push_back(cnm::BufferType::get(partial.dims, partial.elementType,
                                          wgTy.getAccelerator(),
                                          partial.level));
  }
}

static ParseResult
parseLaunchArgsList(OpAsmParser &parser, llvm::StringLiteral kw,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                    SmallVectorImpl<OpAsmParser::Argument> &regionArgs,
                    SmallVectorImpl<PartialBufferType> &types) {

  if (parser.parseOptionalKeyword(kw).succeeded()) {
    if (parser.parseLParen() || parser.parseCommaSeparatedList([&]() {
          auto &arg = regionArgs.emplace_back();
          auto &partialTy = types.emplace_back();
          if (parser.parseArgument(arg) || parser.parseEqual() ||
              parser.parseOperand(operands.emplace_back()) ||
              parser.parseColon() ||
              parseShorthandBufferType(parser, partialTy))
            return failure();

          arg.type = MemRefType::get(partialTy.dims, partialTy.elementType,
                                     nullptr, partialTy.level);
          return success();
        }) ||
        parser.parseRParen())
      return failure();
  }
  return success();
}

ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse workgroup operand
  OpAsmParser::UnresolvedOperand wg;
  if (parser.parseOperand(wg))
    return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;

  SmallVector<OpAsmParser::UnresolvedOperand> inputs;
  SmallVector<PartialBufferType> inputTypesPartial;
  if (parseLaunchArgsList(parser, "ins", inputs, regionArgs, inputTypesPartial))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> outBuffers;
  SmallVector<PartialBufferType> outputTypesPartial;
  if (parseLaunchArgsList(parser, "outs", outBuffers, regionArgs,
                          outputTypesPartial))
    return failure();

  // Parse optional attr-dict
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse `on WorkgroupType`
  cnm::WorkgroupType wgType;
  if (parser.parseKeyword("on") || parser.parseType(wgType))
    return failure();

  SmallVector<Type> inputTypes;
  SmallVector<Type> outputTypes;
  inflatePartialBufferTypes(wgType, inputTypesPartial, inputTypes);
  inflatePartialBufferTypes(wgType, outputTypesPartial, outputTypes);

  // Resolve operands
  if (parser.resolveOperand(wg, wgType, result.operands) ||
      parser.resolveOperands(inputs, inputTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(outBuffers, outputTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Required by AttrSizedOperandSegments
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(inputs.size()),
                           static_cast<int32_t>(outBuffers.size())}));

  // Parse body region
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs, true))
    return failure();
  LaunchOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  return success();
}

LogicalResult LaunchOp::verify() {
  auto bodyArgs = getBody().getArguments();
  auto operands = getParams();
  if (bodyArgs.size() != operands.size())
    return emitOpError("expected ")
           << operands.size() << " arguments, got " << bodyArgs.size();

  for (auto [arg, operand] : llvm::zip(bodyArgs, operands)) {
    if (auto bufTy = dyn_cast<cnm::BufferType>(operand.getType())) {
      auto memrefTy = MemRefType::get(bufTy.getShape(), bufTy.getElementType(),
                                      nullptr, bufTy.getLevel());
      if (arg.getType() != memrefTy)
        return emitError("Mismatched type for launch argument, expected ")
               << memrefTy << ", got " << arg.getType();
    } else if (operand.getType().isIntOrIndexOrFloat()) {
      if (arg.getType() != operand.getType())
        return emitError("Mismatched type for launch argument, expected ")
               << arg.getType();
    } else {
      return emitError("Invalid type for argument ")
             << operand << ", expecting !cnm.buffer or scalar type";
    }
  }
  return success();
}

LogicalResult LocalTransferOp::verify() {
  auto srcTy = getSource().getType();
  auto dstTy = getTarget().getType();
  // Compatible, not equal: a transfer between a statically shaped buffer and a
  // dynamically shaped view of one is legal as long as they agree at runtime.
  // memref.copy has the same rule (SameOperandsShape).
  if (failed(verifyCompatibleShape(srcTy, dstTy)))
    return emitOpError("source shape ")
           << srcTy.getShape() << " is not compatible with target shape "
           << dstTy.getShape();
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("source element type ")
           << srcTy.getElementType() << " does not match target element type "
           << dstTy.getElementType();
  return success();
}

LogicalResult CompactBufferOp::verify() {
  auto srcTy = getSource().getType();
  auto dstTy = getTarget().getType();
  AffineMap map = getMap();

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("source element type ")
           << srcTy.getElementType() << " does not match target element type "
           << dstTy.getElementType();

  if (map.getNumDims() != static_cast<unsigned>(dstTy.getRank()))
    return emitOpError("map takes ")
           << map.getNumDims() << " index(es) but the target has rank "
           << dstTy.getRank()
           << "; the map's domain is the target's index space";
  if (map.getNumResults() != static_cast<unsigned>(srcTy.getRank()))
    return emitOpError("map produces ")
           << map.getNumResults() << " index(es) but the source has rank "
           << srcTy.getRank() << "; the map's results are source indices";

  // Producing something a single flat transfer can move is the reason the op
  // exists, so a target that is not contiguous is a contradiction rather than
  // a case to handle.
  if (!memrefIsContiguous(dstTy))
    return emitOpError("target is not contiguous: ")
           << dstTy << ". A compacted buffer is by definition packed";

  return success();
}

namespace {

/// A view's result indices written in terms of its source's, for the
/// reshaping views that only relabel a buffer without moving anything.
///
/// memref.reshape is included but is the restricted case: its shape is a
/// runtime operand, so there is a map to derive only when both types are
/// statically shaped. Its contract already requires identity layouts on both
/// sides, which is what makes the relabel a plain row-major
/// linearize/delinearize.
FailureOr<AffineMap> viewIndexMap(Operation *op, MLIRContext *ctx) {
  if (auto expand = dyn_cast<memref::ExpandShapeOp>(op)) {
    MemRefType resultTy = expand.getType();
    SmallVector<AffineExpr> results;
    for (ReassociationIndices group : expand.getReassociationIndices()) {
      SmallVector<AffineExpr> groupIndices;
      SmallVector<int64_t> groupShape;
      for (int64_t d : group) {
        groupIndices.push_back(getAffineDimExpr(d, ctx));
        groupShape.push_back(resultTy.getDimSize(d));
      }
      results.push_back(mlir::linearize(ctx, groupIndices, groupShape));
    }
    return AffineMap::get(resultTy.getRank(), 0, results, ctx);
  }

  if (auto collapse = dyn_cast<memref::CollapseShapeOp>(op)) {
    MemRefType srcTy = collapse.getSrcType();
    SmallVector<AffineExpr> results(srcTy.getRank());
    for (auto [pos, group] :
         llvm::enumerate(collapse.getReassociationIndices())) {
      SmallVector<int64_t> groupShape;
      for (int64_t d : group)
        groupShape.push_back(srcTy.getDimSize(d));
      SmallVector<AffineExpr> groupIndices =
          mlir::delinearize(getAffineDimExpr(pos, ctx), groupShape);
      for (auto [d, index] : llvm::zip_equal(group, groupIndices))
        results[d] = index;
    }
    return AffineMap::get(collapse.getType().getRank(), 0, results, ctx);
  }

  if (auto reshape = dyn_cast<memref::ReshapeOp>(op)) {
    auto srcTy = dyn_cast<MemRefType>(reshape.getSource().getType());
    auto resultTy = dyn_cast<MemRefType>(reshape.getType());
    if (!srcTy || !resultTy || !srcTy.hasStaticShape() ||
        !resultTy.hasStaticShape())
      return failure();
    SmallVector<AffineExpr> resultIndices;
    for (int64_t d = 0; d < resultTy.getRank(); ++d)
      resultIndices.push_back(getAffineDimExpr(d, ctx));
    AffineExpr flat = mlir::linearize(ctx, resultIndices, resultTy.getShape());
    return AffineMap::get(resultTy.getRank(), 0,
                          mlir::delinearize(flat, srcTy.getShape()), ctx);
  }

  return failure();
}

/// Absorbs a reshaping view of a repack's source into the repack's own map.
///
/// The view moves nothing, so reading through it is the same as reading the
/// buffer underneath at relabelled indices -- which is exactly what the map is
/// for. Doing it here rather than inside the chain fold below keeps that fold
/// to the one thing it is about.
struct AbsorbViewIntoCompact : public OpRewritePattern<CompactBufferOp> {
  using OpRewritePattern<CompactBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompactBufferOp op,
                                PatternRewriter &rewriter) const override {
    Operation *view = op.getSource().getDefiningOp();
    if (!view ||
        !isa<memref::ExpandShapeOp, memref::CollapseShapeOp, memref::ReshapeOp>(
            view))
      return failure();
    FailureOr<AffineMap> viewMap = viewIndexMap(view, getContext());
    if (failed(viewMap))
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      op.getSourceMutable().assign(view->getOperand(0));
      op.setMap(viewMap->compose(op.getMap()));
    });
    return success();
  }
};

/// Collapses `a -> mid -> b` into `a -> b`.
///
/// The intermediate exists only to be read straight back, so the two repacks
/// are one: composing the maps names, for each element of the final buffer,
/// the element of the original it ultimately comes from.
///
/// Only fires when nothing else touches the intermediate, since a repack
/// writes it and dropping the write would be visible to any other reader. The
/// merged op inherits the *first* repack's `cinm.static`: it now reads that
/// op's source, and the second could not have been tagged anyway -- its source
/// was a fresh allocation, which is never a declared static value.
struct FoldChainedCompacts : public OpRewritePattern<CompactBufferOp> {
  using OpRewritePattern<CompactBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompactBufferOp op,
                                PatternRewriter &rewriter) const override {
    Value mid = op.getSource();
    CompactBufferOp producer;
    for (Operation *user : mid.getUsers())
      if (auto candidate = dyn_cast<CompactBufferOp>(user))
        if (candidate != op && candidate.getTarget() == mid) {
          producer = candidate;
          break;
        }
    if (!producer)
      return failure();

    if (!llvm::all_of(mid.getUsers(), [&](Operation *user) {
          return user == op || user == producer;
        }))
      return failure();
    if (producer->getBlock() != op->getBlock() ||
        !producer->isBeforeInBlock(op))
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      op.getSourceMutable().assign(producer.getSource());
      op.setMap(producer.getMap().compose(op.getMap()));
      if (producer.isStatic())
        op->setAttr(cinm::CinmDialect::STATIC_ATTR_NAME,
                    rewriter.getUnitAttr());
      else
        op->removeAttr(cinm::CinmDialect::STATIC_ATTR_NAME);
    });
    rewriter.eraseOp(producer);
    return success();
  }
};

} // namespace

LogicalResult CompactBufferOp::fold(FoldAdaptor,
                                    SmallVectorImpl<OpFoldResult> &) {
  // Folding only ever swaps an operand for a *more* static type, so a target
  // that verified as contiguous stays contiguous.
  return memref::foldMemRefCast(*this);
}

void CompactBufferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<AbsorbViewIntoCompact, FoldChainedCompacts>(context);
}

LogicalResult LocalTransferOp::fold(FoldAdaptor,
                                    SmallVectorImpl<OpFoldResult> &) {
  // Promotion hands us dynamically shaped views of statically shaped buffers;
  // looking through the cast is what recovers the static transfer size.
  bool folded = false;
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    auto cast = operand.get().getDefiningOp<memref::CastOp>();
    if (cast && memref::CastOp::canFoldIntoConsumerOp(cast)) {
      operand.set(cast.getSource());
      folded = true;
    }
  }
  return success(folded);
}
namespace {

/// Simplifies the map over its own domain, which the workgroup and buffer
/// shapes give exactly.
template <class Op> class SimplifyScatterMap : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto bufferTy = op.getBuffer().getType();
    auto map = op.getScatterMap();
    auto simplified =
        simplifyAffineMapWithBounds(map, getScatterIndexSpace(bufferTy));
    if (simplified == map)
      return failure();

    rewriter.modifyOpInPlace(op, [&] { op.setScatterMap(simplified); });
    return success();
  }
};
} // namespace

void GatherOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                           ::mlir::MLIRContext *context) {
  results.insert<SimplifyScatterMap<GatherOp>>(context);
}

void ScatterOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                            ::mlir::MLIRContext *context) {
  results.insert<SimplifyScatterMap<ScatterOp>>(context);
}

/// The map is pointwise: its domain is the workgroup shape followed by all of
/// the buffer's own dimensions, and it has one result per host dimension. What
/// travels as one block is not recorded here; a consumer that moves blocks
/// derives it with cnm::deflateScatterMap, from the map and the host layout.
///
/// `requireInjective` is set for gathers only: two leaves reading the same
/// host element is a broadcast, two leaves *writing* it is a race.
static LogicalResult verifyScatterGatherMap(Operation *op, ShapedType hostTy,
                                            cnm::BufferType bufferTy,
                                            AffineMap map,
                                            bool requireInjective) {
  ArrayRef<int64_t> wgShape = bufferTy.getWorkgroupShape();
  ArrayRef<int64_t> bufShape = bufferTy.getShape();
  ArrayRef<int64_t> hostShape = hostTy.getShape();

  if (map.getNumInputs() != wgShape.size() + bufShape.size())
    return op->emitOpError("map has ")
           << map.getNumInputs()
           << " dimension(s); a pointwise map has the workgroup's "
           << wgShape.size() << " followed by the buffer's " << bufShape.size();

  if (map.getNumResults() != hostShape.size())
    return op->emitOpError("map has ")
           << map.getNumResults()
           << " result(s); a pointwise map has one per host dimension, of "
              "which there are "
           << hostShape.size();

  // What is left is where the transfer lands, which needs the host value's
  // element order. A memref with a non-identity layout does not have one until
  // its strides are taken into account, and that is the business of
  // --cnm-ensure-scatter-gather-contiguous, not of a verifier.
  if (!hostTy.hasStaticShape())
    return success();
  if (auto memrefTy = dyn_cast<MemRefType>(hostTy))
    if (!memrefTy.getLayout().isIdentity())
      return success();

  FailureOr<AffineExpr> offset = cnm::linearizeScatterMap(map, hostShape);
  if (failed(offset))
    return success();
  SmallVector<int64_t> extents = cnm::getScatterIndexSpace(bufferTy);

  int64_t hostElements = computeProduct(hostShape);
  if (std::optional<int64_t> highest =
          mlir::getAffineUpperBound(*offset, extents))
    if (*highest >= hostElements)
      return op->emitOpError("transfer reaches element ")
             << *highest << " of a host value that has only " << hostElements;

  if (requireInjective)
    if (std::optional<bool> injective =
            mlir::isAffineExprInjective(*offset, extents))
      if (!*injective)
        return op->emitOpError(
            "map is not injective: two leaves would write the same host "
            "element. A gather must partition the host value");

  return success();
}

LogicalResult ScatterOp::verify() {
  return verifyScatterGatherMap(*this, getInput().getType(),
                                getBuffer().getType(), getScatterMap(),
                                /*requireInjective=*/false);
}

LogicalResult GatherOp::verify() {
  return verifyScatterGatherMap(*this, getOutputBuf().getType(),
                                getBuffer().getType(), getGatherMap(),
                                /*requireInjective=*/true);
}

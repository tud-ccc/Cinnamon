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

void AllocOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
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

template <class Op> class SimplifyScatterMap : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto map = op.getScatterMap();
    auto simplified = simplifyAffineMapWithBounds(
        map, getScatterMapDomain(map, op.getBuffer().getType()));
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

/// The map's domain is the workgroup shape followed by the first `p` of the
/// buffer's own dimensions, for any `p`. The `bufferRank - p` dimensions left
/// out are transferred as a block, and the same number of host dimensions are
/// left out of the results: they are the block's shape, so they have to match
/// extent for extent. `p = bufferRank` names a host element per buffer
/// element; `p = 0` is one whole-buffer block per leaf.
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

  if (map.getNumInputs() < wgShape.size() ||
      map.getNumInputs() > wgShape.size() + bufShape.size())
    return op->emitOpError("map has ")
           << map.getNumInputs() << " dimension(s); expected the workgroup's "
           << wgShape.size() << ", optionally followed by up to "
           << bufShape.size() << " leading buffer dimension(s)";

  int64_t blockRank = cnm::getNumImplicitHostDims(map, bufferTy);
  if (static_cast<int64_t>(map.getNumResults()) + blockRank !=
      static_cast<int64_t>(hostShape.size()))
    return op->emitOpError("map has ")
           << map.getNumResults() << " result(s) and leaves " << blockRank
           << " buffer dimension(s) implicit, which does not add up to the "
           << hostShape.size() << " dimension(s) of the host value";

  ArrayRef<int64_t> blockShape = cnm::getScatterBlockShape(map, bufferTy);
  if (hostShape.take_back(blockRank) != blockShape)
    return op->emitOpError("the implicit block has shape ")
           << blockShape << " but the host dimensions it covers have shape "
           << hostShape.take_back(blockRank);

  // What is left is where the transfer lands, which needs the host value's
  // element order. A memref with a non-identity layout does not have one until
  // its strides are taken into account, and that is the business of
  // --cnm-ensure-scatter-gather-contiguous, not of a verifier.
  if (!hostTy.hasStaticShape())
    return success();
  if (auto memrefTy = dyn_cast<MemRefType>(hostTy))
    if (!memrefTy.getLayout().isIdentity())
      return success();

  FailureOr<AffineExpr> offset = cnm::linearizeScatterMap(
      cnm::inflateScatterMapToPointwise(map, bufferTy), hostShape);
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

/// Implements the Cnm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h"
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>

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
#include <mlir/Dialect/MemRef/Utils/MemRefUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/TypeRange.h>
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
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("source shape ")
           << srcTy.getShape() << " does not match target shape "
           << dstTy.getShape();
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("source element type ")
           << srcTy.getElementType() << " does not match target element type "
           << dstTy.getElementType();
  return success();
}

LogicalResult ScatterOp::verify() {
  auto tensorTy = getInput().getType();
  auto bufferTy = getBuffer().getType();
  auto map = getScatterMap();
  // The affine map maps every WG element to a prefix of the input tensor which
  // has buffer shape

  if (map.getNumInputs() != bufferTy.getWorkgroupShape().size()) {
    return emitError() << "Affine map inputs (" << map.getNumInputs()
                       << " dims) do not correspond to workgroup dimensions ("
                       << bufferTy.getWorkgroupShape().size() << " dims)";
  }

  auto truncatedDims = tensorTy.getShape().size() - bufferTy.getShape().size();
  if (map.getNumResults() != truncatedDims) {
    return emitError()
           << "Affine map results (" << map.getNumResults()
           << ") do not correspond to truncated scattered tensor dimensions ("
           << tensorTy.getShape().size() << " - " << bufferTy.getShape().size()
           << ")";
  }

  if (tensorTy.getShape().slice(truncatedDims) != bufferTy.getShape()) {
    return emitError()
           << "Scattered tensor shape should end with buffer shape, ("
           << tensorTy.getShape().slice(truncatedDims)
           << " != " << bufferTy.getShape() << ")";
  }

  // Note: we used to reject non-contiguous scattered memrefs here, but
  // scatteredMemrefIsContiguous only checks contiguity of the bufShape
  // suffix, which isn't sufficient to guarantee a valid single-DMA transfer
  // once lowered (e.g. it misses non-contiguity introduced by the workgroup's
  // thread dimension). Instead of rejecting here, the
  // cnm-ensure-scatter-gather-contiguous pass detects genuinely
  // non-contiguous transfers and inserts a packing buffer before lowering.

  if (auto accelerator =
          cinm::getEnclosingAcceleratorAs<CnmAcceleratorAttrInterface>(
              *this)) {
    // todo if there is an accelerator, we could give it an opportunity to
    //  verify the scattering. For instance for upmem it is illegal to use
    //  the thread ID to scattering from host to mram. 


  }

  return success();
}

LogicalResult GatherOp::verify() {
  auto tensorTy = getOutputBuf().getType();
  auto bufferTy = getBuffer().getType();
  auto map = getGatherMap();
  // The affine map maps every WG-element index and buffer element index
  // to a result tensor index

  if (map.getNumInputs() != bufferTy.getWorkgroupShape().size()) {
    return emitError() << "Affine map inputs (" << map.getNumInputs()
                       << " dims) do not correspond to workgroup dimensions ("
                       << bufferTy.getWorkgroupShape().size() << " dims)";
  }

  auto truncatedDims = tensorTy.getShape().size() - bufferTy.getShape().size();
  if (map.getNumResults() != truncatedDims) {
    return emitError()
           << "Affine map results (" << map.getNumResults()
           << ") do not correspond to truncated scattered tensor dimensions ("
           << tensorTy.getShape().size() << " - " << bufferTy.getShape().size()
           << ")";
  }

  if (tensorTy.getShape().slice(truncatedDims) != bufferTy.getShape()) {
    return emitError()
           << "Scattered tensor shape should end with buffer shape, ("
           << tensorTy.getShape().slice(truncatedDims)
           << " != " << bufferTy.getShape() << ")";
  }

  // See the note in ScatterOp::verify(): contiguity is ensured later by the
  // cnm-ensure-scatter-gather-contiguous pass rather than rejected here.

  return success();
}

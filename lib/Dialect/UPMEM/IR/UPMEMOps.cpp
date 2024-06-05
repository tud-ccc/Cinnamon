/// Implements the UPMEM dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "upmem-ops"

using namespace mlir;
using namespace mlir::upmem;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// UPMEMDialect
//===----------------------------------------------------------------------===//

void UPMEMDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.cpp.inc"
      >();
}

// parsers/printers

LogicalResult UPMEMDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  if (!llvm::isa<UnitAttr>(attr.getValue()) ||
      attr.getName() != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';
  return success();
}

static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

static ParseResult
parseAttributions(OpAsmParser &parser, StringRef keyword,
                  SmallVectorImpl<OpAsmParser::Argument> &args) {
  // If we could not parse the keyword, just assume empty list and succeed.
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  return parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                                  /*allowType=*/true);
}

/// Prints a UPMEM function memory attribution.
static void printAttributions(OpAsmPrinter &p, StringRef keyword,
                              ArrayRef<BlockArgument> values) {
  if (values.empty())
    return;

  p << ' ' << keyword << '(';
  llvm::interleaveComma(
      values, p, [&p](BlockArgument v) { p << v << " : " << v.getType(); });
  p << ')';
}

/// Verifies a UPMEM function memory attribution.
// static LogicalResult verifyAttributions(Operation *op,
//                                         ArrayRef<BlockArgument> attributions,
//                                         upmem::AddressSpace memorySpace) {
//   for (Value v : attributions) {
//     auto type = llvm::dyn_cast<MemRefType>(v.getType());
//     if (!type)
//       return op->emitOpError() << "expected memref type in attribution";

//     // We can only verify the address space if it hasn't already been lowered
//     // from the AddressSpaceAttr to a target-specific numeric value.
//     auto addressSpace =
//         llvm::dyn_cast_or_null<upmem::AddressSpaceAttr>(type.getMemorySpace());
//     if (!addressSpace)
//       continue;
//     if (addressSpace.getValue() != memorySpace)
//       return op->emitOpError()
//              << "expected memory space " <<
//              stringifyAddressSpace(memorySpace)
//              << " in attribution";
//   }
//   return success();
// }

//===----------------------------------------------------------------------===//
// AsyncOpInterface
//===----------------------------------------------------------------------===//

void upmem::addAsyncDependency(Operation *op, Value token) {
  op->insertOperands(0, {token});
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseI32ArrayAttr>(attrName);

  // Async dependencies is the only variadic operand.
  if (!sizeAttr)
    return;

  SmallVector<int32_t, 8> sizes(sizeAttr.asArrayRef());
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getDenseI32ArrayAttr(sizes));
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     Value device_hierarchy, Value rankSize, Value dpuSize,
                     Value taskletSize, Value dynamicSharedMemorySize,
                     Type asyncTokenType, ValueRange asyncDependencies,
                     TypeRange workgroupAttributions,
                     TypeRange privateAttributions) {

  // Add a WorkGroup attribution attribute. This attribute is required to
  // identify private attributions in the list of block argguments.
  result.addAttribute(getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(workgroupAttributions.size()));

  // Add Op operands.
  result.addOperands(asyncDependencies);
  if (asyncTokenType)
    result.types.push_back(builder.getType<AsyncTokenType>());

  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands(device_hierarchy);
  result.addOperands({rankSize, dpuSize, taskletSize});
  if (dynamicSharedMemorySize)
    result.addOperands(dynamicSharedMemorySize);

  // Create a kernel body region with kNumConfigRegionAttributes + N memory
  // attributions, where the first kNumConfigRegionAttributes arguments have
  // `index` type and the rest have the same types as the data operands.
  Region *kernelRegion = result.addRegion();
  Block *body = new Block();
  // TODO: Allow passing in proper locations here.
  for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
    body->addArgument(builder.getIndexType(), result.location);
  // Add WorkGroup & Private attributions to the region arguments.
  for (Type argTy : workgroupAttributions)
    body->addArgument(argTy, result.location);
  for (Type argTy : privateAttributions)
    body->addArgument(argTy, result.location);
  kernelRegion->push_back(body);
  // Fill OperandSegmentSize Attribute.
  SmallVector<int32_t, 5> segmentSizes(5, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = dynamicSharedMemorySize ? 1 : 0;
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));
}

KernelDim LaunchOp::getRankIdClass() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim{args[0]};
}

KernelDim LaunchOp::getDPUIdClass() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim{args[1]};
}

KernelDim LaunchOp::getTaskletIdClass() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim{args[2]};
}

KernelDim LaunchOp::getRankSizeClass() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim{args[3]};
}

KernelDim LaunchOp::getDPUSizeClass() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim{args[4]};
}

KernelDim LaunchOp::getTaskletSizeClass() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim{args[5]};
}

KernelDim LaunchOp::getRankSizeOperandValue() {
  auto operands = getOperands().drop_front(1+getAsyncDependencies().size());
  return KernelDim{operands[0]};
}

KernelDim LaunchOp::getDPUSizeOperandValue() {
  auto operands = getOperands().drop_front(1+getAsyncDependencies().size());
  return KernelDim{operands[1]};
}

KernelDim LaunchOp::getTaskletSizeOperandValue() {
  auto operands = getOperands().drop_front(1+getAsyncDependencies().size());
  return KernelDim{operands[2]};
}

LogicalResult LaunchOp::verifyRegions() {
  if (!getBody().empty()) {
    if (getBody().getNumArguments() <
        kNumConfigRegionAttributes + getNumWorkgroupAttributions())
      return emitOpError("unexpected number of region arguments");
  }

  for (Block &block : getBody()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
    if (!isa<upmem::TerminatorOp>(&block.back())) {
      return block.back()
          .emitError()
          .append("expected '", upmem::TerminatorOp::getOperationName(),
                  "' or a terminator with successors")
          .attachNote(getLoc())
          .append("in '", LaunchOp::getOperationName(), "' body region");
    }
  }

  if (getNumResults() == 0 && getAsyncToken())
    return emitOpError("needs to be named when async keyword is specified");

  return success();
}

static void printSizeAssignment(OpAsmPrinter &p, KernelDim size,
                                KernelDim operand, KernelDim id) {
  p << '(' << id.x << ") in (";
  p << size.x << " = " << operand.x << ')';
}
static void printSizeAssignment(OpAsmPrinter &p, Value size,
                                BlockArgument arg) {
  p << '(';
  p.printRegionArgument(arg, {}, true);
  p << " upto " << size << ")";
}

void LaunchOp::print(OpAsmPrinter &p) {
  if (getAsyncToken()) {
    p << " async";
    if (!getAsyncDependencies().empty())
      p << " [" << getAsyncDependencies() << ']';
  }
  // Print the launch configuration.
  p << ' ' << getDeviceHierarchy();
  p << ' ' << getRanksKeyword();
  auto blockArgs = getRegion().getArguments();
  printSizeAssignment(p, getRankSizeOperandValue().x, blockArgs[0]);

  p << ' ' << getDPUsKeyword();
  printSizeAssignment(p, getDPUSizeOperandValue().x, blockArgs[1]);

  p << ' ' << getTaskletsKeyword();
  printSizeAssignment(p, getTaskletSizeOperandValue().x, blockArgs[2]);

  if (getDynamicSharedMemorySize())
    p << ' ' << getDynamicSharedMemorySizeKeyword() << ' '
      << getDynamicSharedMemorySize();

  p << " on " << getDeviceHierarchy().getType() << " ";

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              LaunchOp::getOperandSegmentSizeAttr(),
                              getNumWorkgroupAttributionsAttrName()});
}

static ParseResult
parseSizeAssignment(OpAsmParser &parser,
                    MutableArrayRef<OpAsmParser::UnresolvedOperand> sizes,
                    MutableArrayRef<OpAsmParser::UnresolvedOperand> regionSizes,
                    MutableArrayRef<OpAsmParser::UnresolvedOperand> indices) {
  // assert(indices.size() == 3 && "space for three indices expected");
  SmallVector<OpAsmParser::UnresolvedOperand, 1> args;
  if (parser.parseOperandList(args, OpAsmParser::Delimiter::Paren,
                              /*allowResultNumber=*/false) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();
  std::move(args.begin(), args.end(), indices.begin());

  if (parser.parseOperand(regionSizes[0], /*allowResultNumber=*/false) ||
      parser.parseEqual() || parser.parseOperand(sizes[0]))
    return failure();
  return parser.parseRParen();
}

static ParseResult
parseSizeAssignment(OpAsmParser &parser, OpAsmParser::Argument &arg,
                    OpAsmParser::UnresolvedOperand &upperBound) {
  if (parser.parseLParen() || parser.parseArgument(arg, false) ||
      parser.parseKeyword("upto") || parser.parseOperand(upperBound) ||
      parser.parseRParen())
    return failure();
  arg.type = parser.getBuilder().getIndexType();
  return success();
}

ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Sizes of the grid and block.
  SmallVector<OpAsmParser::UnresolvedOperand, LaunchOp::kNumConfigOperands>
      sizes(LaunchOp::kNumConfigOperands);
  MutableArrayRef<OpAsmParser::UnresolvedOperand> sizesRef(sizes);

  // Actual (data) operands passed to the kernel.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dataOperands;

  SmallVector<Value, 1> deviceHierarchyOperand;
  result.operands.emplace_back();

  // Region arguments to be created.

  // Parse optional async dependencies.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  Type asyncTokenType;
  if (failed(
          parseAsyncDependencies(parser, asyncTokenType, asyncDependencies)) ||
      parser.resolveOperands(asyncDependencies, asyncTokenType,
                             result.operands))
    return failure();
  if (parser.getNumResults() > 0)
    result.types.push_back(asyncTokenType);

  OpAsmParser::UnresolvedOperand deviceHierarchy;
  if (parser.parseOperand(deviceHierarchy)) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> regionArgs2;
  SmallVector<OpAsmParser::UnresolvedOperand> upperBounds;
  if (parser.parseKeyword(LaunchOp::getRanksKeyword().data()) ||
      parseSizeAssignment(parser, regionArgs2.emplace_back(),
                          upperBounds.emplace_back()) ||
      parser.parseKeyword(LaunchOp::getDPUsKeyword().data()) ||
      parseSizeAssignment(parser, regionArgs2.emplace_back(),
                          upperBounds.emplace_back()) ||
      parser.parseKeyword(LaunchOp::getTaskletsKeyword().data()) ||
      parseSizeAssignment(parser, regionArgs2.emplace_back(),
                          upperBounds.emplace_back()) ||
      parser.resolveOperands(upperBounds, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();

  OpAsmParser::UnresolvedOperand dynamicSharedMemorySize;
  bool hasDynamicSharedMemorySize = false;
  if (!parser.parseOptionalKeyword(
          LaunchOp::getDynamicSharedMemorySizeKeyword())) {
    hasDynamicSharedMemorySize = true;
    if (parser.parseOperand(dynamicSharedMemorySize) ||
        parser.resolveOperand(dynamicSharedMemorySize,
                              parser.getBuilder().getI32Type(),
                              result.operands))
      return failure();
  }

  Type index = parser.getBuilder().getIndexType();
  SmallVector<Type, LaunchOp::kNumConfigRegionAttributes> dataTypes(
      LaunchOp::kNumConfigRegionAttributes, index);

  for (auto arg : regionArgs2) {
  }

  Type deviceHierarchyType;
  if (parser.parseKeyword("on") || parser.parseType(deviceHierarchyType) ||
      parser.resolveOperand(deviceHierarchy, deviceHierarchyType,
                            deviceHierarchyOperand)) {
    return failure();
  }
  result.operands[0] = deviceHierarchyOperand[0];

  Builder &builder = parser.getBuilder();
  // Parse workgroup memory attributions.
  if (failed(parseAttributions(parser, LaunchOp::getWorkgroupKeyword(),
                               regionArgs2)))
    return failure();

  // Store the number of operands we just parsed as the number of workgroup
  // memory attributions.
  unsigned numWorkgroupAttrs =
      regionArgs2.size() - LaunchOp::kNumConfigRegionAttributes;
  result.addAttribute(LaunchOp::getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(numWorkgroupAttrs));

  // Parse private memory attributions.
  if (failed(parseAttributions(parser, LaunchOp::getPrivateKeyword(),
                               regionArgs2)))
    return failure();

  // Introduce the body region and parse it. The region has
  // kNumConfigRegionAttributes arguments that correspond to
  // block/thread identifiers and grid/block sizes, all having `index` type.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs2) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<int32_t, 6> segmentSizes(6, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = hasDynamicSharedMemorySize ? 1 : 0;
  result.addAttribute(LaunchOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

/// Simplify the upmem.launch when the range of a thread or block ID is
/// trivially known to be one.
struct FoldLaunchArguments : public OpRewritePattern<LaunchOp> {
  using OpRewritePattern<LaunchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LaunchOp op,
                                PatternRewriter &rewriter) const override {
    // If the range implies a single value for `id`, replace `id`'s uses by
    // zero.
    Value zero;
    bool simplified = false;
    auto constPropIdUses = [&](Value id, Value size) {
      // Check if size is trivially one.
      if (!matchPattern(size, m_One()))
        return;
      if (id.getUses().empty())
        return;
      if (!simplified) {
        // Create a zero value the first time.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&op.getBody().front());
        zero =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), /*value=*/0);
      }
      rewriter.replaceAllUsesWith(id, zero);
      simplified = true;
    };
    constPropIdUses(op.getRankIdClass().x, op.getRankSizeClass().x);
    constPropIdUses(op.getDPUIdClass().x, op.getDPUSizeClass().x);
    constPropIdUses(op.getTaskletIdClass().x, op.getTaskletSizeClass().x);

    return success(simplified);
  }
};

void LaunchOp::getCanonicalizationPatterns(RewritePatternSet &rewrites,
                                           MLIRContext *context) {
  rewrites.add<FoldLaunchArguments>(context);
}

/// Adds a new block argument that corresponds to buffers located in
/// workgroup memory.
BlockArgument LaunchOp::addWorkgroupAttribution(Type type, Location loc) {
  auto attrName = getNumWorkgroupAttributionsAttrName();
  auto attr = (*this)->getAttrOfType<IntegerAttr>(attrName);
  (*this)->setAttr(attrName,
                   IntegerAttr::get(attr.getType(), attr.getValue() + 1));
  return getBody().insertArgument(
      LaunchOp::kNumConfigRegionAttributes + attr.getInt(), type, loc);
}

/// Adds a new block argument that corresponds to buffers located in
/// private memory.
BlockArgument LaunchOp::addPrivateAttribution(Type type, Location loc) {
  // Buffers on the private memory always come after buffers on the workgroup
  // memory.
  return getBody().addArgument(type, loc);
}



//===----------------------------------------------------------------------===//
// UPMEMModuleOp
//===----------------------------------------------------------------------===//

void UPMEMModuleOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.attributes.push_back(builder.getNamedAttr(
      ::mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));

}

ParseResult UPMEMModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  ArrayAttr targetsAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the module body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return failure();

  // Ensure that this module has a valid terminator.
  UPMEMModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void UPMEMModuleOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());

  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}
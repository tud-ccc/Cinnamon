/// Implements the Cinm dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmWorkgroupTypeInterface.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include <cmath>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/InliningUtils.h>

#define DEBUG_TYPE "cinm-base"

using namespace mlir;
using namespace mlir::cinm;

namespace mlir::cinm {
/// One #cinm.host_platform parameter: an integer literal is accepted as well
/// as a float, so that `vector_bytes = 64` need not be spelled `64.`.
static FailureOr<llvm::APFloat> parseHostParameter(AsmParser &parser) {
  int64_t integral;
  OptionalParseResult asInteger = parser.parseOptionalInteger(integral);
  if (asInteger.has_value()) {
    if (failed(*asInteger))
      return failure();
    return llvm::APFloat(static_cast<double>(integral));
  }
  double value;
  if (parser.parseFloat(value))
    return failure();
  return llvm::APFloat(value);
}
} // namespace mlir::cinm

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.cpp.inc"
#include "cinm-mlir/Dialect/Cinm/IR/CinmComputeOpInterface.cpp.inc"
#include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.cpp.inc"
#include "cinm-mlir/Dialect/Cinm/IR/CinmWorkgroupTypeInterface.cpp.inc"
// #define GET_ATTRDEF_CLASSES

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//
struct CinmInlinerInterface : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const override {
    // register that it is legal to inline an operation (eg func.func)
    // containing a cinm.compute_block op. This may duplicate the compute block
    // though.
    return true;
  }
};

void CinmDialect::initialize() {
  registerOps();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.cpp.inc"
      >();
  this->addInterfaces<CinmInlinerInterface>();
  this->declarePromisedInterfaces<CinmTilingInterface, GemmOp, GemvOp,
                                  BatchGemmOp, BatchGemvOp, ReduceOp,
                                  ElementwiseOp>();
}

::mlir::LogicalResult
CinmDialect::verifyOperationAttribute(::mlir::Operation *op,
                                      ::mlir::NamedAttribute attribute) {

  if (attribute.getName() == CinmDialect::TILING_FACTORS_NAME) {
    if (!op->hasPromiseOrImplementsInterface<CinmTilingInterface>()) {
      return op->emitOpError() << attribute.getName()
                               << " attribute can only be used ops "
                                  "implementing the CinmTilingInterface";
    }
    if (!llvm::isa<DenseI64ArrayAttr>(attribute.getValue()))
      return op->emitOpError() << attribute.getName()
                               << " attribute should be a dense i64 array attr";

    auto tileSizes =
        llvm::cast<DenseI64ArrayAttr>(attribute.getValue()).asArrayRef();
    auto tilingIface = llvm::cast<CinmTilingInterface>(op);
    SmallVector<int64_t> dimSizes;
    tilingIface.getTilableDimSizes(dimSizes);

    if (tileSizes.size() != dimSizes.size())
      return op->emitError()
             << "Attribute " << attribute.getName().strref() << " has "
             << tileSizes.size() << " tiling factor(s) but op has "
             << dimSizes.size() << " tileable dimension(s)";

    for (auto [i, dim, tile] : llvm::enumerate(dimSizes, tileSizes)) {
      if (ShapedType::isDynamic(dim))
        continue;
      if (tile <= 0)
        return op->emitError()
               << "Attribute " << attribute.getName().strref()
               << " tiling factor #" << i << " must be positive";
      if (dim % tile != 0)
        return op->emitError() << "Attribute " << attribute.getName().strref()
                               << " tiling factor #" << i << " (" << tile
                               << ") does not divide dimension size " << dim;
    }
    return success();
  }
  if (attribute.getName() == CinmDialect::OPERAND_REUSE_NAME) {
    if (!isa<cinm::ComputeBlockOp>(op))
      return op->emitOpError() << attribute.getName()
                               << " can only be set on a cinm.compute_block";
    auto entries = llvm::dyn_cast<DenseI64ArrayAttr>(attribute.getValue());
    if (!entries)
      return op->emitOpError()
             << attribute.getName() << " should be a dense i64 array attr";
    // A pass that changes the operand list without updating the entries
    // would otherwise leave them describing other operands.
    if (entries.size() != static_cast<int64_t>(op->getNumOperands()))
      return op->emitOpError()
             << attribute.getName() << " has " << entries.size()
             << " entries for " << op->getNumOperands() << " operands";
    if (llvm::any_of(entries.asArrayRef(), [](int64_t e) { return e < -1; }))
      return op->emitOpError()
             << attribute.getName() << " entries are byte counts, or -1";
    return success();
  }
  if (attribute.getName() == CinmDialect::AVAILABLE_PLATFORMS_NAME) {
    bool validHost = op->hasTrait<FunctionOpInterface::Trait>() ||
                     isa<cinm::ComputeOp, cinm::ComputeBlockOp>(op);
    if (!validHost)
      return op->emitOpError("Attribute ")
             << CinmDialect::AVAILABLE_PLATFORMS_NAME
             << " must be specified on a function op or cinm.compute op";
    auto arr = llvm::dyn_cast<ArrayAttr>(attribute.getValue());
    if (!arr)
      return op->emitOpError("Attribute ")
             << CinmDialect::AVAILABLE_PLATFORMS_NAME
             << " must be an array attribute";
    for (auto elem : arr) {
      if (!llvm::isa<CinmPlatformAttrInterface>(elem))
        return op->emitOpError("Attribute ")
               << CinmDialect::AVAILABLE_PLATFORMS_NAME
               << " elements must implement CinmPlatformAttrInterface";
    }
    return success();
  }
  if (attribute.getName() == CinmDialect::DEBUG_TAG_NAME) {
    if (!llvm::isa<StringAttr>(attribute.getValue()))
      return op->emitOpError("Attribute ")
             << CinmDialect::DEBUG_TAG_NAME << " must be a string attribute";
    return success();
  }
  if (attribute.getName() == CinmDialect::GRAPH_ALLOC_NAME) {
    if (!isa<cinm::ComputeBlockOp>(op))
      return op->emitOpError("Attribute ")
             << CinmDialect::GRAPH_ALLOC_NAME
             << " describes a compute block's place in a solved graph and "
                "belongs on a cinm.compute_block op";
    if (!llvm::isa<DictionaryAttr>(attribute.getValue()))
      return op->emitOpError("Attribute ") << CinmDialect::GRAPH_ALLOC_NAME
                                           << " must be a dictionary attribute";
    return success();
  }
  if (attribute.getName() == CinmDialect::STATIC_ATTR_NAME) {
    // On a function argument this declares the serving contract that the
    // argument holds the same data on every inference (see
    // cinm::isStaticValue). On an operation it records that a conclusion was
    // already drawn about its operands -- a repack of static data is
    // amortizable over the serving lifetime, which decides how it is timed --
    // at the point where the defining ops were still reachable.
    if (!llvm::isa<UnitAttr>(attribute.getValue()))
      return op->emitOpError("Attribute ")
             << CinmDialect::STATIC_ATTR_NAME << " must be a unit attribute";
    return success();
  }
  if (attribute.getName() == CinmDialect::OUTLINED_NAME) {
    // The prototype of an outlined host block; its arguments carry
    // cinm.reads / cinm.writes (verifyRegionArgAttribute).
    if (!op->hasTrait<FunctionOpInterface::Trait>() ||
        !llvm::isa<UnitAttr>(attribute.getValue()))
      return op->emitOpError("Attribute ")
             << CinmDialect::OUTLINED_NAME
             << " is a unit attribute on a function";
    return success();
  }
  return op->emitOpError("unknown attribute ") << attribute.getName();
}

LogicalResult CinmDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIndex,
                                                    unsigned argIndex,
                                                    NamedAttribute attribute) {
  if (attribute.getName() == CinmDialect::READS_NAME ||
      attribute.getName() == CinmDialect::WRITES_NAME) {
    if (!op->hasAttr(CinmDialect::OUTLINED_NAME) ||
        !llvm::isa<UnitAttr>(attribute.getValue()))
      return op->emitOpError("Attribute ")
             << attribute.getName()
             << " is a unit attribute on an argument of a cinm.outlined "
                "function";
    return success();
  }
  return success();
}

Attribute CinmDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  if (parser.parseOptionalKeyword(CinmPlatformArrayAttr::getMnemonic())
          .succeeded())
    return CinmPlatformArrayAttr::parse(parser, type);
  if (parser.parseOptionalKeyword(CostModelDataAttr::getMnemonic()).succeeded())
    return CostModelDataAttr::parse(parser, type);

  StringRef mnemonic;
  Attribute result;
  // The struct parser reports a missing host parameter only as a missing
  // ',', so a host that fails to parse also says what it takes.
  if (parser.parseOptionalKeyword(HostPlatformAttr::getMnemonic())
          .succeeded()) {
    Attribute host = HostPlatformAttr::parse(parser, type);
    if (!host)
      parser.emitError(parser.getNameLoc())
          << "#cinm.host_platform takes exactly these parameters, each once, "
             "in any order: ops_per_second, dram_bytes_per_second, "
             "scalar_op_ns, vector_op_ns, vector_bytes, "
             "stream_bytes_per_second, copy_bytes_per_second";
    return host;
  }

  auto res = generatedAttributeParser(parser, &mnemonic, type, result);
  if (res.has_value() && succeeded(*res))
    return result;
  if (!res.has_value())
    parser.emitError(parser.getNameLoc(), "Unknown attribute ") << mnemonic;
  return {};
}

void CinmDialect::printAttribute(Attribute attr, DialectAsmPrinter &out) const {
  if (auto myAttr = llvm::dyn_cast<CostModelDataAttr>(attr)) {
    myAttr.print(out);
    return;
  }
  (void)generatedAttributePrinter(attr, out);
}

LogicalResult HostPlatformAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, llvm::APFloat opsPerSecond,
    llvm::APFloat dramBytesPerSecond, llvm::APFloat scalarOpNs,
    llvm::APFloat vectorOpNs, llvm::APFloat vectorBytes,
    llvm::APFloat streamBytesPerSecond, llvm::APFloat copyBytesPerSecond) {
  const std::pair<llvm::StringRef, const llvm::APFloat *> parameters[] = {
      {"ops_per_second", &opsPerSecond},
      {"dram_bytes_per_second", &dramBytesPerSecond},
      {"scalar_op_ns", &scalarOpNs},
      {"vector_op_ns", &vectorOpNs},
      {"vector_bytes", &vectorBytes},
      {"stream_bytes_per_second", &streamBytesPerSecond},
      {"copy_bytes_per_second", &copyBytesPerSecond},
  };
  for (auto [key, value] : parameters) {
    if (&value->getSemantics() != &llvm::APFloat::IEEEdouble())
      return emitError() << "host platform parameter '" << key
                         << "' must be a double";
    double v = value->convertToDouble();
    if (!(v > 0.0) || !std::isfinite(v))
      return emitError() << "host platform parameter '" << key
                         << "' must be positive and finite, got " << v;
  }
  return success();
}

HostModel HostPlatformAttr::getModel() const {
  HostModel model;
  model.opsPerSecond = getOpsPerSecond().convertToDouble();
  model.dramBytesPerSecond = getDramBytesPerSecond().convertToDouble();
  model.scalarOpNs = getScalarOpNs().convertToDouble();
  model.vectorOpNs = getVectorOpNs().convertToDouble();
  model.vectorBytes = getVectorBytes().convertToDouble();
  model.streamBytesPerSecond = getStreamBytesPerSecond().convertToDouble();
  model.copyBytesPerSecond = getCopyBytesPerSecond().convertToDouble();
  return model;
}

HostPlatformAttr HostPlatformAttr::getInScope(Operation *op) {
  for (Operation *scope = op; scope; scope = scope->getParentOp()) {
    auto available =
        scope->getAttrOfType<ArrayAttr>(CinmDialect::AVAILABLE_PLATFORMS_NAME);
    if (!available)
      continue;
    for (Attribute attr : available)
      if (auto host = llvm::dyn_cast<HostPlatformAttr>(attr))
        return host;
  }
  return {};
}

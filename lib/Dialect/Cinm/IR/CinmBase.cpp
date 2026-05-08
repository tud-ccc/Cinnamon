/// Implements the Cinm dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
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

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.cpp.inc"
#include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.cpp.inc"
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
}

::mlir::LogicalResult
CinmDialect::verifyOperationAttribute(::mlir::Operation *op,
                                      ::mlir::NamedAttribute attribute) {

  if (attribute.getName() == CinmDialect::TILING_FACTORS_NAME) {
    if (!op->hasTrait<CinmTilingInterface::Trait>()) {
      return op->emitOpError() << attribute.getName()
                               << " attribute can only be used ops "
                                  "implementing the CinmTilingInterface";
    }
    if (!llvm::isa<DenseI64ArrayAttr>(attribute.getValue()))
      return op->emitOpError() << attribute.getName()
                               << " attribute should be a dense i64 array attr";
    return success();
  }
  if (attribute.getName() == CinmDialect::AVAILABLE_PLATFORMS_NAME) {
    if (op->hasTrait<FunctionOpInterface::Trait>()) {
      return success();
    }
    return op->emitOpError("Attribute ")
           << CinmDialect::AVAILABLE_PLATFORMS_NAME
           << " must be specified on a function op";
  }
  return op->emitOpError("unknown attribute ") << attribute.getName();
}

Attribute CinmDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  if (parser.parseOptionalKeyword(CinmPlatformArrayAttr::getMnemonic())
          .succeeded())
    return CinmPlatformArrayAttr::parse(parser, type);
  if (parser.parseOptionalKeyword(HostPlatformAttr::getMnemonic()).succeeded())
    return HostPlatformAttr::parse(parser, type);
  if (parser.parseOptionalKeyword(CostModelDataAttr::getMnemonic()).succeeded())
    return CostModelDataAttr::parse(parser, type);

  StringRef mnemonic;
  Attribute result;
  auto res = generatedAttributeParser(parser, &mnemonic, type, result);
  if (res.has_value() && res.value().succeeded())
    return result;
  parser.emitError(parser.getNameLoc(), "Unknown attribute ") << mnemonic;
  return {};
}

void CinmDialect::printAttribute(Attribute attr, DialectAsmPrinter &out) const {
  if (auto a = llvm::dyn_cast_or_null<HostPlatformAttr>(attr)) {
    out << HostPlatformAttr::getMnemonic();
    a.print(out);
    return;
  } else if (auto myAttr = llvm::dyn_cast<CostModelDataAttr>(attr)) {
    myAttr.print(out);
    return;
  }
  (void)generatedAttributePrinter(attr, out);
}

void CinmVarDefAttr::print(AsmPrinter &out) const {
  out.printKeywordOrString(getIndexVarName());
  out << " : ";
  out.printKeywordOrString(getBoundVarName());
  if (getLowerBoundInclusive() == getUpperBoundInclusive()) {
    out << " = " << getLowerBoundInclusive();
  } else {
    out << " in " << getLowerBoundInclusive() << " to "
        << getUpperBoundInclusive();
  }
}

Attribute CinmVarDefAttr::parse(::mlir::AsmParser &parser, ::mlir::Type) {
  std::string name;
  if (parser.parseKeywordOrString(&name)) {
    return {};
  }
  auto indexName = parser.getBuilder().getStringAttr(name);
  if (parser.parseColon() || parser.parseKeywordOrString(&name)) {
    return {};
  }
  auto boundName = parser.getBuilder().getStringAttr(name);

  if (parser.parseOptionalEqual().succeeded()) {
    uint64_t bound;
    if (parser.parseInteger(bound))
      return {};
    return parser.getBuilder().getAttr<CinmVarDefAttr>(indexName, boundName,
                                                       bound, bound);
  }

  uint64_t lbound;
  uint64_t ubound;
  if (parser.parseKeyword("in") || parser.parseInteger(lbound) ||
      parser.parseKeyword("to") || parser.parseInteger(ubound))
    return {};

  return parser.getBuilder().getAttr<CinmVarDefAttr>(indexName, boundName,
                                                     lbound, ubound);
}

Attribute HostPlatformAttr::parse(::mlir::AsmParser &parser, ::mlir::Type) {
  return get(parser.getContext());
}
void HostPlatformAttr::print(::mlir::AsmPrinter &) const {}

CinmVarDefArrayAttr cinm::detail::instantiateDesignParams(
    CinmVarDefArrayAttr array,
    const llvm::MapVector<StringRef, long> &instantiations) {
  if (instantiations.empty())
    return array;

  llvm::SmallVector<CinmVarDefAttr> parms(array.getValue());
  for (auto &parm : parms) {
    if (auto value = instantiations.lookup(parm.getBoundVarName())) {
      parm = parm.withValue(value);
    }
  }

  return CinmVarDefArrayAttr::get(array.getContext(), std::move(parms));
}

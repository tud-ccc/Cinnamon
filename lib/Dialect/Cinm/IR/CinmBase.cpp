/// Implements the Cinm dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LogicalResult.h>

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

void CinmDialect::initialize() {
  registerOps();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.cpp.inc"
      >();
}

::mlir::LogicalResult
CinmDialect::verifyOperationAttribute(::mlir::Operation *op,
                                      ::mlir::NamedAttribute attribute) {

  if (attribute.getName() == CinmDialect::NOTILE_NAME) {
    if (op->getDialect() == this) {
      return success();
    }
    return op->emitOpError()
           << CinmDialect::NOTILE_NAME
           << " attribute can only be used on cinm dialect operations";
  }
  return op->emitOpError("unknown attribute ") << attribute.getName();
}

Attribute CinmDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  if (parser.parseOptionalKeyword(HostPlatformAttr::getMnemonic()).succeeded())
    return HostPlatformAttr::parse(parser, type);
  if (parser.parseOptionalKeyword(CostModelDataAttr::getMnemonic()).succeeded())
    return CostModelDataAttr::parse(parser, type);
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

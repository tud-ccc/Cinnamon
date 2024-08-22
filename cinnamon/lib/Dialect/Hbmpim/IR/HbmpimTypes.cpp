/// Implements the Hbmpim dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimAttributes.h"
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Debug.h"

#include "llvm/ADT/TypeSwitch.h"


#define DEBUG_TYPE "hbmpim-types"

using namespace mlir;
using namespace mlir::hbmpim;

//===- Generated implementation -------------------------------------------===//
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HbmpimDialect
//===----------------------------------------------------------------------===//

void HbmpimDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimTypes.cpp.inc"
      >();
}

// parsers/printers

// Type mlir::hbmpim::PIMCMDsType::parse(mlir::AsmParser &parser) {
//   SmallVector<PIMCMD> shape;

//   PIMCMDAttr pim_cmd_attr;
//   NamedAttrList attrStorage;
//   auto loc = parser.getCurrentLocation();
//   auto parseCmdList = [&]() -> ParseResult {
//     StringRef attrStr;
//     StringAttr attrVal;
//     if(parser.parseKeyword(&attrStr)){
//       OptionalParseResult parseResult =
//         parser.parseOptionalAttribute(attrVal,
//                                       parser.getBuilder().getNoneType(),
//                                       "pim_cmd", attrStorage);
//       if (parseResult.has_value()) {
//         if (failed(*parseResult))
//           return failure();
//         attrStr = attrVal.getValue();
//       } else {
//         return parser.emitError(loc, "expected string or keyword containing one of enum values for attribute pim_cmd ");
//       }
//     }
//     if (!attrStr.empty()) {
//       auto attr = ::mlir::hbmpim::symbolizePIMCMD(attrStr);
//       if (!attr)
//         return parser.emitError(loc, "invalid ")
//                << "pim_cmd attribute specification: \"" << attrStr << '"';;
//       shape.push_back(*attr);
//     }
//     return success();
//   };
//   if (parser.parseLess() || parser.parseCommaSeparatedList(parseCmdList) ||
//       parser.parseGreater()) 
//     return Type();

//   return hbmpim::PIMCMDsType::get(parser.getContext(), shape);
// }

// void mlir::hbmpim::PIMCMDsType::print(mlir::AsmPrinter &printer) const {
//   printer << "<";
//   printer.printStrippedAttrOrType(getShape());
//   printer << ">";
// }


Type mlir::hbmpim::PIMCMDVecType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess()) return Type();
  PimKernelType kerType;
  PimKernelTypeAttr kerType_attr;
  NamedAttrList attrStorage;
  auto loc = parser.getCurrentLocation();
  StringRef attrStr;
  StringAttr attrVal;
  if(parser.parseKeyword(&attrStr)){
    OptionalParseResult parseResult =
      parser.parseOptionalAttribute(attrVal,
                                    parser.getBuilder().getNoneType(),
                                    "pim_kern_type", attrStorage);
    if (parseResult.has_value()) {
      if (failed(*parseResult)){
        parser.emitError(loc, "expected string or keyword containing one of enum values for attribute pim_cmd "); 
        return Type();
      }

      attrStr = attrVal.getValue();
    } else {
      parser.emitError(loc, "expected string or keyword containing one of enum values for attribute pim_cmd ");
      return Type();
    }
  }
  if (!attrStr.empty()) {
    auto attr = ::mlir::hbmpim::symbolizePimKernelType(attrStr);
    if (!attr){
      parser.emitError(loc, "invalid ")
               << "pim_cmd attribute specification: \"" << attrStr << '"';;
      return Type();
    }
      if(parser.parseGreater()) return Type();
      return hbmpim::PIMCMDVecType::get(parser.getContext(), *attr); 
  }
  return Type();
}

void mlir::hbmpim::PIMCMDVecType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getKerType());
  printer << ">";
}


//===----------------------------------------------------------------------===//
// DeviceConfiguration
//===----------------------------------------------------------------------===//

Type mlir::hbmpim::DeviceConfigurationType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t, 3> shape;
  if (parser.parseLess() || parser.parseDimensionList(shape, false, false) ||
      parser.parseGreater()) {
    return Type();
  }

  return hbmpim::DeviceConfigurationType::get(parser.getContext(), shape);
}

void mlir::hbmpim::DeviceConfigurationType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << ">";
}

LogicalResult mlir::hbmpim::DeviceConfigurationType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape) {
  if (shape.size() != 4)
    return emitError() << "hbmpim device configuration should have 4 dimensions: "
                       << shape;
  return success();
}

//===----------------------------------------------------------------------===//
// BurstType 
//===----------------------------------------------------------------------===//

Type mlir::hbmpim::BurstType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess()) return Type();
  DeviceBurstType kerType;
  DeviceBurstTypeAttr kerType_attr;
  NamedAttrList attrStorage;
  auto loc = parser.getCurrentLocation();
  StringRef attrStr;
  StringAttr attrVal;
  if(parser.parseKeyword(&attrStr)){
    OptionalParseResult parseResult =
      parser.parseOptionalAttribute(attrVal,
                                    parser.getBuilder().getNoneType(),
                                    "pim_kern_type", attrStorage);
    if (parseResult.has_value()) {
      if (failed(*parseResult)){
        parser.emitError(loc, "expected string or keyword containing one of enum values for attribute burst_type"); 
        return Type();
      }

      attrStr = attrVal.getValue();
    } else {
      parser.emitError(loc, "expected string or keyword containing one of enum values for attribute burst_type");
      return Type();
    }
  }
  if (!attrStr.empty()) {
    auto attr = ::mlir::hbmpim::symbolizeDeviceBurstType(attrStr);
    if (!attr){
      parser.emitError(loc, "invalid ")
               << "burst_type attribute specification: \"" << attrStr << '"';;
      return Type();
    }
      if(parser.parseGreater()) return Type();
      return hbmpim::BurstType::get(parser.getContext(), *attr); 
  }
  return Type();
}

void mlir::hbmpim::BurstType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  // printer.printDimensionList(getShape());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// OperandDataDim 
//===----------------------------------------------------------------------===//

Type mlir::hbmpim::OperandDataDimType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess()) return Type();
  DataDimType kerType;
  DataDimTypeAttr kerType_attr;
  NamedAttrList attrStorage;
  auto loc = parser.getCurrentLocation();
  StringRef attrStr;
  StringAttr attrVal;
  if(parser.parseKeyword(&attrStr)){
    OptionalParseResult parseResult =
      parser.parseOptionalAttribute(attrVal,
                                    parser.getBuilder().getNoneType(),
                                    "pim_kern_type", attrStorage);
    if (parseResult.has_value()) {
      if (failed(*parseResult)){
        parser.emitError(loc, "expected string or keyword containing one of enum values for attribute burst_type"); 
        return Type();
      }

      attrStr = attrVal.getValue();
    } else {
      parser.emitError(loc, "expected string or keyword containing one of enum values for attribute burst_type");
      return Type();
    }
  }
  if (!attrStr.empty()) {
    auto attr = ::mlir::hbmpim::symbolizeDataDimType(attrStr);
    if (!attr){
      parser.emitError(loc, "invalid ")
               << "burst_type attribute specification: \"" << attrStr << '"';;
      return Type();
    }
      if(parser.parseGreater()) return Type();
      return hbmpim::OperandDataDimType::get(parser.getContext(), *attr); 
  }
  return Type();
}

void mlir::hbmpim::OperandDataDimType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  // printer.printDimensionList(getShape());
  printer << ">";
}

// LogicalResult mlir::hbmpim::BurstType::verify(
//     function_ref<InFlightDiagnostic()> emitError, DeviceBurstType type) {
//   // if (shape.size() != 4)
//   //   return emitError() << "hbmpim device configuration should have 4 dimensions: "
//   //                      << shape;
//   return success();
// }
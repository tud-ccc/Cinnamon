/// Declaration of the Cinm dialect attributes.
///
/// @file

#pragma once

#include <cinm-mlir/Dialect/Cinm/IR/CinmBase.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Attributes.h>

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmBaseAttributes.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttr.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributesForOps.h.inc"
namespace mlir::cinm::detail {

CinmVarDefArrayAttr
instantiateDesignParams(CinmVarDefArrayAttr array,
                        const llvm::MapVector<StringRef, long> &instantiations);
}
//===----------------------------------------------------------------------===//

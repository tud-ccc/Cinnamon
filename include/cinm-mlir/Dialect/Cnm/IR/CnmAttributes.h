/// Declaration of the Cnm dialect attributes.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <mlir/IR/Attributes.h>

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cnm/IR/CnmAttributes.h.inc"

// CnmTypes.h includes CnmInterfaces.h (interfaces) then CnmTypes.h.inc (types),
// and defines getWorkgroupType() once both are complete.
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

//===----------------------------------------------------------------------===//

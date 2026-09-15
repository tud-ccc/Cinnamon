/// Declaration of the Cnm dialect attributes.
///
/// @file

#pragma once

#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmBase.h>
#include <mlir/IR/Attributes.h>

//===- Generated includes -------------------------------------------------===//

// Forward-declare to break cycle: CnmInterfaces.h.inc declares
// getWorkgroupType() returning WorkgroupType, which is defined in
// CnmTypes.h.inc and depends on this interface.
namespace mlir::cnm {
class WorkgroupType;
}

#include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h.inc"

//===----------------------------------------------------------------------===//

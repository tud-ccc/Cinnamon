/// Declaration of the Cnm dialect types.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include <functional>
#include <numeric>

//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h.inc"

// Both CnmAcceleratorAttrInterface and WorkgroupType are now complete.
inline mlir::cnm::WorkgroupType
mlir::cnm::CnmAcceleratorAttrInterface::getWorkgroupType() const {
  return ::mlir::cnm::WorkgroupType::get(getContext(), *this);
}

//===----------------------------------------------------------------------===//

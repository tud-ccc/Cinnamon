/// Declares the EKL TypeCheckOpInterface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

namespace mlir::cinm::impl {

LogicalResult verifyTilingInterface(Operation *op);

} // namespace mlir::cinm::impl

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h.inc"

//===----------------------------------------------------------------------===//

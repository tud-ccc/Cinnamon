/// Convenience include for the Cinm dialect.
///

#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

namespace mlir::cinm {
void registerCinmBufferizableOpInterfaces(mlir::DialectRegistry &);
} // namespace mlir::cinm
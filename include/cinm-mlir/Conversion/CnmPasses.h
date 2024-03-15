/// Declaration of the conversion pass within Cnm dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/CnmToGPU/CnmToGPU.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir

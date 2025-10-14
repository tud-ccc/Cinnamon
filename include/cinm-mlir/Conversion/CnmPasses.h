/// Declaration of the conversion pass within Cnm dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/CnmToGPU/CnmToGPU.h"
#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir

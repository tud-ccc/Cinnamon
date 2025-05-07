/// Declaration of the conversion pass within Cinm dialect.
///
/// @file

#pragma once

#include <cinm-mlir/Conversion/TorchToCinm/TorchToCinm.h>

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/CinmFrontendPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
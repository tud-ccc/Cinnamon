/// Declaration of the conversion pass within Cinm dialect.
///
/// @file

#pragma once

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
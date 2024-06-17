/// Declaration of the conversion pass within Cinm dialect.
///
/// @file

#pragma once

#include <cinm-mlir/Conversion/CinmToCnm/CinmToCnm.h>

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
/// Declaration of the conversion pass within ${dialectNameUpper} dialect.
///
/// @file

#pragma once

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "${projectPrefix}/Conversion/${dialectNameUpper}Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
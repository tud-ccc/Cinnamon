/// Declaration of the conversion pass within Memristor dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/MemristorToFunc/MemristorToFunc.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/MemristorPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir

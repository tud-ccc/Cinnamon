/// Declaration of the conversion pass within Cim dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/CimToMemristor/CimToMemristor.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/CimPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir

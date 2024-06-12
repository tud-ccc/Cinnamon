/// Declaration of the conversion pass within UPMEM dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/UPMEMToLLVM/UPMEMToLLVM.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/UPMEMPasses.h.inc"

//===----------------------------------------------------------------------===//

} 
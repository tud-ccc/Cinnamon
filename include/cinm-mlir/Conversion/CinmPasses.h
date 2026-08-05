/// @file

#pragma once

#include <cinm-mlir/Conversion/CinmToCim/CinmToCim.h>
#include <cinm-mlir/Conversion/CinmToCnm/CinmToCnm.h>
#include <cinm-mlir/Conversion/CinmToLinalg/CinmToLinalg.h>
#include <cinm-mlir/Conversion/LinalgToCinm/Im2ColToMatmul.h>
#include <cinm-mlir/Conversion/LinalgToCinm/LinalgToCinm.h>

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir

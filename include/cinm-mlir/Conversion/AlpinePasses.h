#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

// Declarations from Passes.td
#define GEN_PASS_DECL_CONVERTALPINETOFUNC
#include "cinm-mlir/Conversion/AlpinePasses.h.inc"

namespace mlir {
namespace alpine {

std::unique_ptr<mlir::Pass> createConvertAlpineToFuncPass();

void registerAlpineConversionPasses();

} // namespace alpine
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/AlpinePasses.h.inc"

namespace mlir {
namespace alpine {

inline void registerAlpineConversionPasses() {
  registerConvertAlpineToFuncPass();
}

} // namespace alpine
} // namespace mlir

#pragma once
#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace alpine {
std::unique_ptr<mlir::Pass> createConvertAlpineToFuncPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "cinm-mlir/Conversion/AlpinePasses.h.inc"

} // namespace alpine
} // namespace mlir

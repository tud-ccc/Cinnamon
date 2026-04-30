#pragma once

#include <cinm-mlir/Dialect/Cinm/IR/CinmDialect.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cinm {

#define GEN_PASS_DECL_CONVERTLINALGTOCINM
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

void registerLinalgToCinmPipeline();
std::unique_ptr<Pass> createConvertLinalgToCinmPass();

} // namespace mlir::cinm
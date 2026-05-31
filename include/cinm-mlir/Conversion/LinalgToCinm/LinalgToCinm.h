#pragma once

#include <cinm-mlir/Dialect/Cinm/IR/CinmDialect.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cinm {

void registerLinalgToCinmPipeline();
std::unique_ptr<Pass> createConvertLinalgToCinmPass();

} // namespace mlir::cinm
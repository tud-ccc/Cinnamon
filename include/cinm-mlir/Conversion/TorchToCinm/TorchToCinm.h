#pragma once
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"

#include <mlir/Pass/Pass.h>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>

namespace mlir::cinm_frontend {
std::unique_ptr<Pass> createConvertTorchToCinmPass();
} // namespace mlir::cinm_frontend
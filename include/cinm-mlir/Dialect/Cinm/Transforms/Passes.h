/// Declaration of the transform pass within Cinm dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>

namespace mlir::cinm {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::cinm

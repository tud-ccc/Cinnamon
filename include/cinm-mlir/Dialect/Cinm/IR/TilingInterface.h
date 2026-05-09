#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir::cinm {
struct ComputeBlockOp;

enum class ReductionDimMode : uint8_t {
  Unspecified = 0,
  Fixed = 1,
  Heuristic = 2
};


} // namespace mlir::cinm

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h.inc"

//===----------------------------------------------------------------------===//
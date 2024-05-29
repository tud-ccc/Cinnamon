/// Declares the EKL TypeCheckOpInterface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/STLExtras.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <utility>

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::cinm {

using BodyBuilderCallback = function_ref<SmallVector<Value>(
    OpBuilder &, Location, ValueRange, ValueRange)>;

ResultRange createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> loopSizes,
                                       ValueRange iterArgInit,
                                       BodyBuilderCallback bodyBuilder);

Value createVectorReduceSum(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createMatmul(OpBuilder builder, Location loc, Value lhs, Value rhs,
                   int64_t reduceClusterSize = 1);

} // namespace mlir::cinm

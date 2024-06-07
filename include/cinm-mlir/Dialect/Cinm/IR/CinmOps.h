/// Declaration of the Cinm dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmTypes.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::cinm {

cinm::ComputeOp getEnclosingComputeBlock(Operation *op);

} // namespace mlir::cinm
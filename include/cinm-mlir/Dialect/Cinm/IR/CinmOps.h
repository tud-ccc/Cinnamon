/// Declaration of the Cinm dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmTypes.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <llvm/Support/Casting.h>

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::cinm {

Type inferGemmReturnType(Type lhsType, Type rhsType);

cinm::ComputeBlockOp getEnclosingComputeBlock(Operation *op);
cinm::CinmAcceleratorAttrInterface getEnclosingAccelerator(Operation *op);
template <class T> T getEnclosingAcceleratorAs(Operation *op) {
  auto ax = getEnclosingAccelerator(op);
  if (ax) {
    T res = llvm::dyn_cast_or_null<T>(ax);
    return res;
  }
  return {};
}

} // namespace mlir::cinm

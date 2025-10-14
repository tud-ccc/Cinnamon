/// Declaration of the UPMEM dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/UPMEM/IR/AsyncOpInterface.h"  
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace upmem {

struct KernelDim {
  Value x;
};

// Adds a `upmem.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);

} // namespace upmem
} // namespace mlir

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h.inc"

// namespace mlir::upmem
//===----------------------------------------------------------------------===//
/// Declaration of the UPMEM dialect types.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>


namespace mlir {
namespace upmem {


} // namespace upmem
} // namespace mlir


//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h.inc"

//===----------------------------------------------------------------------===//


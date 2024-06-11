/// Declaration of the UPMEM CPP emitter.
///
/// @file

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace mlir {
namespace upmem_emitc {

/// Translates the given operation to C++ code. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect. The parameter
/// 'declareVariablesAtTop' enforces that all variables for op results and block
/// arguments are declared at the beginning of the function.
LogicalResult UPMEMtranslateToCpp(Operation *op, raw_ostream &os,
                             bool declareVariablesAtTop = false);

void registerUPMEMCppTranslation();
} // namespace emitc
} // namespace mlir

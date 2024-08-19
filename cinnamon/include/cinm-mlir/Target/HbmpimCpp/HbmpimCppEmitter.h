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
namespace hbmpim_emitc {

LogicalResult HbmpimTranslateToCpp(Operation *op, raw_ostream &os);

void registerHbmpimCppTranslation();
} // namespace emitc
} // namespace mlir

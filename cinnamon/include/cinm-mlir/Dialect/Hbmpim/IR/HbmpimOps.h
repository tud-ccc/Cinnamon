/// Declaration of the Hbmpim dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/STLExtras.h"


//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.h.inc"

//===----------------------------------------------------------------------===//

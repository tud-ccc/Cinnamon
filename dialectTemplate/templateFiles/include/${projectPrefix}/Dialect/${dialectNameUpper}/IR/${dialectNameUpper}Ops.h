/// Declaration of the ${dialectNameUpper} dialect ops.
///
/// @file

#pragma once

#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"



//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Ops.h.inc"

//===----------------------------------------------------------------------===//

/// Declaration of the Cim dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Cim/IR/CimAttributes.h"
#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"

#include "cinm-mlir/Dialect/Cim/IR/CimTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h.inc"

//===----------------------------------------------------------------------===//

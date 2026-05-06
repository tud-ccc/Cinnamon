/// Convenience include for the Cnm dialect.
///

#pragma once
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

#include <cinm-mlir/Dialect/Cnm/IR/CnmAttributes.h>

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h.inc"
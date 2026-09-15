/// Declaration of the Cim dialect attributes.
#pragma once
#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "mlir/IR/Attributes.h"

// Enums first so scoped enumerators (e.g. ::mlir::cim::RoundingMode::Nearest)
// exist
#include "cinm-mlir/Dialect/Cim/IR/CimEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cim/IR/CimAttributes.h.inc"

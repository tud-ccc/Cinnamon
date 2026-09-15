#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimDialect.h"

using namespace mlir;
using namespace mlir::cim;

#include "cinm-mlir/Dialect/Cim/IR/CimBase.cpp.inc"

// Bring in enum helpers (stringify/symbolize) once.
#include "cinm-mlir/Dialect/Cim/IR/CimEnums.cpp.inc"

// Bring in attribute class definitions once.
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cim/IR/CimAttributes.cpp.inc"

void CimDialect::initialize() {
  registerOps();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/Cim/IR/CimAttributes.cpp.inc"
      >();
}

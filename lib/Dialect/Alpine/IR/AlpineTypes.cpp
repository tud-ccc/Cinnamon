#include "cinm-mlir/Dialect/Alpine/IR/AlpineTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#define DEBUG_TYPE "alpine-types"

using namespace mlir;
using namespace mlir::alpine;


#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Alpine/IR/AlpineTypes.cpp.inc"



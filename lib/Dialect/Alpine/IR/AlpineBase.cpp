#include "cinm-mlir/Dialect/Alpine/IR/AlpineBase.h"

#include "cinm-mlir/Dialect/Alpine/IR/AlpineDialect.h"

#define DEBUG_TYPE "alpine-base"

using namespace mlir;
using namespace mlir::alpine;

#include "cinm-mlir/Dialect/Alpine/IR/AlpineBase.cpp.inc"

void AlpineDialect::initialize() { registerOps(); }

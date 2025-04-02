
#pragma once

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OpImplementation.h>

#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>

#include <mlir/Dialect/Transform/IR/TransformAttrs.h>
#include <mlir/Dialect/Transform/IR/TransformDialect.h>
#include <mlir/Dialect/Transform/IR/TransformTypes.h>
#include <mlir/Dialect/Transform/Interfaces/MatchInterfaces.h>
#include <mlir/Dialect/Transform/Interfaces/TransformInterfaces.h>

#define GET_OP_CLASSES
#include <cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformOps.h.inc>

namespace mlir::cnm {

void registerTransformDialectExtension(::mlir::DialectRegistry &registry);

} // namespace mlir::cnm
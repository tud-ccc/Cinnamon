/// Declaration of conversion passes for the Cinm dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace cinm {
class CinmDialect;
} // namespace cinm
namespace func { class FuncOp; }

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir/// Declaration of conversion passes for the Cnm dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace cnm {
class CnmDialect;
} // namespace cnm

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir/// Declaration of conversion passes for the UPMEM dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace upmem {
class UPMEMDialect;
} // namespace upmem

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/UPMEMPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
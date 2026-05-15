/// Declaration of conversion passes for the Cinm dialect.
///
/// @file

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace cinm {
class CinmDialect;
} // namespace cinm
namespace func {
class FuncOp;
}

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

//===----------------------------------------------------------------------===//

namespace cnm {
class CnmDialect;
} // namespace cnm

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

//===----------------------------------------------------------------------===//

namespace upmem {
class UPMEMDialect;
} // namespace upmem

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "cinm-mlir/Conversion/UPMEMPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
/// Declaration of the UPMEM dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

//===- Generated includes -------------------------------------------------===//

namespace mlir::upmem::detail {
MemRefType flatMemRefType(Type structured);
}
namespace mlir::upmem {

struct DpuSetResource : public SideEffects::Resource::Base<DpuSetResource> {
  DpuSetResource() = default;
  StringRef getName() const override { return "<DPU set>"; }
};

/// Unit attribute on a loop in a DPU program that the DPU compiler must not
/// unroll: --upmem-register-tile-loops has already shaped it, and the C
/// translator emits `#pragma clang loop unroll(disable)` before it.
constexpr llvm::StringLiteral kNoUnrollAttr = "upmem.nounroll";
} // namespace mlir::upmem

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h.inc"

// namespace mlir::upmem
//===----------------------------------------------------------------------===//

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

/// The runtime's node timers (upmem.node_begin / upmem.node_end): an effect
/// on it keeps the markers from being erased or merged, and touches no host
/// memory and no DPU set, so nothing is ordered against it but the markers.
struct NodeTimerResource
    : public SideEffects::Resource::Base<NodeTimerResource> {
  NodeTimerResource() = default;
  StringRef getName() const override { return "<node timer>"; }
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

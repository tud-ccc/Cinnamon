//===- UPMEMOccupancy.h - Per-DPU memory footprint of a kernel -----------===//
//
// How much MRAM and WRAM one `upmem.dpu_program` occupies on a DPU once it is
// compiled. Shared by the C translator (which has to declare the stack size
// the SDK compiles the kernel with) and by --upmem-check-occupancy (which
// rejects a program that does not fit).
//
// Keeping one definition matters: the two numbers have to agree, or the check
// passes a configuration the translator then emits code too large for.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

namespace mlir::upmem {

/// Bytes one tasklet's stack needs: a fixed reserve for the runtime, plus
/// every `upmem.pwram_alloc` in the program, which the translator emits as a
/// stack array. Allocations are counted even when they are in disjoint scopes
/// -- C would let those overlap, but the SDK is given a single stack size.
int64_t taskletStackBytes(DpuProgramOp program);

/// What `program` occupies on one DPU, in bytes.
struct DpuOccupancy {
  /// MRAM is per-DPU and its static allocations are shared by the tasklets, so
  /// each is counted once.
  int64_t mramBytes = 0;
  /// `numTasklets * taskletStackBytes`, since private WRAM is per-tasklet,
  /// plus the static WRAM allocations, which are not.
  int64_t wramBytes = 0;
  /// The stack size the kernel is compiled with; a component of `wramBytes`,
  /// reported separately because that is what the SDK is told.
  int64_t taskletStackBytes = 0;
};

DpuOccupancy measureOccupancy(DpuProgramOp program);

} // namespace mlir::upmem

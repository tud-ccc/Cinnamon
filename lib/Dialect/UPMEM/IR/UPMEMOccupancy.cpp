//===- UPMEMOccupancy.cpp - Per-DPU memory footprint of a kernel ---------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOccupancy.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <variant>

namespace mlir::upmem {

namespace {

/// Bytes the translator declares for a buffer of this type. It emits arrays
/// padded to an 8-byte boundary, because MRAM DMA transfers require 8-byte
/// alignment, so the padding is part of the footprint.
int64_t declaredBytes(MemRefType type) {
  int64_t eltBytes = type.getElementTypeBitWidth() / 8;
  return llvm::alignTo(type.getNumElements() * eltBytes, int64_t{8});
}

/// Bytes reserved on every tasklet's stack for the runtime itself (locals,
/// saved registers, the barrier structures) on top of what the kernel's own
/// buffers need.
constexpr int64_t kStackReserveBytes = 1024;

} // namespace

int64_t taskletStackBytes(DpuProgramOp program) {
  int64_t bytes = kStackReserveBytes;
  program->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<memref::AllocaOp, upmem::PrivateWRAMAllocOp>(
            [&](auto alloc) { bytes += declaredBytes(alloc.getType()); })
        .Default([](auto) {});
  });
  return llvm::alignTo(bytes, int64_t{8});
}

DpuOccupancy measureOccupancy(DpuProgramOp program) {
  DpuOccupancy result;
  result.taskletStackBytes = taskletStackBytes(program);
  result.wramBytes = result.taskletStackBytes * program.getNumTasklets();

  program->walk([&](StaticAllocOp alloc) {
    int64_t bytes =
        declaredBytes(cast<MemRefType>(alloc.getBuffer().getType()));
    // Static allocations are file-scope arrays in the generated C: one per
    // DPU, not one per tasklet, in either memory space.
    if (alloc.isMram())
      result.mramBytes += bytes;
    else
      result.wramBytes += bytes;
  });

  return result;
}

} // namespace mlir::upmem

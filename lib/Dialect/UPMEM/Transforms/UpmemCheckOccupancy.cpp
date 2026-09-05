//===- UpmemCheckOccupancy.cpp - Reject kernels that do not fit ----------===//
//
// Measures what each DPU program allocates and fails if it does not fit the
// device. Runs last, after every memory optimization: how much a program needs
// is a property of the lowered code, not of the configuration it was lowered
// from. See docs/CnmMemoryLevelsDesign.md §H3.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOccupancy.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMCHECKOCCUPANCYPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {

/// Capacity of one DPU's memories, in bytes.
struct Capacities {
  int64_t mram = 0;
  int64_t wram = 0;
};

/// Read the capacities off the accelerator `load` runs on. Returns nullopt if
/// the accelerator, its platform, or either level is missing -- the caller
/// reports that, rather than quietly checking nothing.
std::optional<Capacities> capacitiesOf(LoadProgramOp load) {
  auto block = load->getParentOfType<cinm::ComputeBlockOp>();
  if (!block)
    return std::nullopt;
  auto accelerator = block.getAcceleratorAttr();
  if (!accelerator)
    return std::nullopt;
  auto platform = accelerator.getPlatform();
  if (!platform)
    return std::nullopt;

  MLIRContext *ctx = load->getContext();
  Capacities result;
  for (auto [space, out] : {std::pair{DpuMemSpace::MRAM, &result.mram},
                            std::pair{DpuMemSpace::WRAM, &result.wram}}) {
    cinm::CinmLevelDefAttr level =
        platform.getLevelOfMemspace(DpuMemSpaceAttr::get(ctx, space));
    if (!level)
      return std::nullopt;
    *out = level.getSizeInBytes();
  }
  return result;
}

struct UpmemCheckOccupancyPass
    : public impl::UpmemCheckOccupancyPassBase<UpmemCheckOccupancyPass> {
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Only programs something actually loads are checked: --upmem-dedup-kernels
    // can leave a program behind with no users, and a dead kernel occupies
    // nothing.
    llvm::DenseMap<Operation *, LoadProgramOp> loaders;
    module->walk([&](LoadProgramOp load) {
      if (DpuProgramOp program = load.getDpuProgram())
        loaders.try_emplace(program, load);
    });

    for (auto [op, load] : loaders) {
      auto program = cast<DpuProgramOp>(op);
      Capacities capacities{mramSize, wramSize};
      if (!mramSize || !wramSize) {
        std::optional<Capacities> declared = capacitiesOf(load);
        if (!declared) {
          load->emitError("cannot check whether @")
              << program.getSymName()
              << " fits: no accelerator with an mram and a wram level is in "
                 "scope, and the mram-size/wram-size options do not supply "
                 "both capacities";
          return signalPassFailure();
        }
        if (!mramSize)
          capacities.mram = declared->mram;
        if (!wramSize)
          capacities.wram = declared->wram;
      }

      DpuOccupancy occupancy = measureOccupancy(program);
      bool fits = true;
      if (occupancy.mramBytes > capacities.mram) {
        program->emitError("MRAM occupancy of ")
            << occupancy.mramBytes << " bytes exceeds the " << capacities.mram
            << " bytes a DPU has";
        fits = false;
      }
      if (occupancy.wramBytes > capacities.wram) {
        program->emitError("WRAM occupancy of ")
            << occupancy.wramBytes << " bytes exceeds the " << capacities.wram
            << " bytes a DPU has (" << program.getNumTasklets()
            << " tasklets x " << occupancy.taskletStackBytes
            << " bytes of stack, plus "
            << occupancy.wramBytes -
                   occupancy.taskletStackBytes * program.getNumTasklets()
            << " bytes of static buffers)";
        fits = false;
      }
      if (!fits)
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::upmem

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>
#include <variant>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMANNOTATECOSTSPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

namespace {

struct UpmemAnnotateCostsPass
    : impl::UpmemAnnotateCostsPassBase<UpmemAnnotateCostsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *container = getOperation();

    std::unique_ptr<UpmemSimulator> sim = createSimulator(simulator, true);

    container->walk([&](cinm::ComputeBlockOp computeBlock) {
      auto res = sim->simulate(computeBlock.getBody());
      if (std::holds_alternative<DiagnosedSilenceableFailure>(res)) {
        (void)std::get<DiagnosedSilenceableFailure>(res).checkAndReport();
        signalPassFailure();
      }
    });
  }
};

} // namespace

} // namespace mlir::upmem

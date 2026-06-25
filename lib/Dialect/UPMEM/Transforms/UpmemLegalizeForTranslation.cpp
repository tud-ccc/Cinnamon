//===- UpmemLegalizeForTranslation.cpp - Strip host ops for DPU translation
//===//

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/WalkResult.h>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMLEGALIZEFORTRANSLATIONPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

struct UpmemLegalizeForTranslationPass
    : impl::UpmemLegalizeForTranslationPassBase<
          UpmemLegalizeForTranslationPass> {

  using impl::UpmemLegalizeForTranslationPassBase<
      UpmemLegalizeForTranslationPass>::UpmemLegalizeForTranslationPassBase;

  void runOnOperation() override {
    llvm::SmallVector<Operation *> toErase;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<ModuleOp>(op)) {
        return WalkResult::advance();
      } else if (!isa<DpuProgramOp>(op)) {
        toErase.push_back(op);
      }
      return WalkResult::skip();
    });

    for (Operation *op : toErase)
      op->erase();
  }
};

} // namespace mlir::upmem

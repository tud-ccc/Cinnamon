
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMHOISTWORKGROUPSPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;
namespace {}

struct CnmHoistWorkgroupsPass
    : public cnm::impl::CnmHoistWorkgroupsPassBase<CnmHoistWorkgroupsPass> {
  void runOnOperation() override {
    // todo
    auto fun = getOperation();
    if (fun.isDeclaration())
      return;

    llvm::SmallVector<cnm::WorkgroupOp> allocs;
    fun->walk([&](cnm::WorkgroupOp op) { allocs.push_back(op); });

    OpBuilder rewriter(&getContext());
    rewriter.setInsertionPointToStart(&fun.getBody().front());
    IRMapping mapper;
    for (auto alloc : allocs) {
      Operation *parent = alloc;
      while (parent->getParentOp() != fun) {
        parent = parent->getParentOp();
      }
      if (parent == alloc) {
        // nothing to hoist
        continue;
      }
      rewriter.setInsertionPoint(parent);
      auto newAlloc = rewriter.clone(*alloc, mapper);
      alloc.replaceAllUsesWith(newAlloc);
      alloc->erase();

      for (auto user : newAlloc->getUsers()) {
        if (llvm::isa<cnm::FreeWorkgroupOp>(user)) {
          rewriter.setInsertionPointAfter(parent);
          rewriter.clone(*user, mapper);
          user->erase();
          break;
        }
      }
    }
  }
};
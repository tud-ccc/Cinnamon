
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
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
    auto* fun = getOperation();
    if (fun->getNumRegions() == 0) return;

    llvm::SmallVector<cnm::WorkgroupOp> allocs;
    fun->walk([&](cnm::WorkgroupOp op) { allocs.push_back(op); });

    OpBuilder rewriter(&getContext());
    rewriter.setInsertionPointToStart(&fun->getRegion(0).front());
    IRMapping mapper;
    for (auto alloc : allocs) {
      Operation *parent = alloc;
      while (parent->getParentOp() && parent->getParentOp() != fun && !parent->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
        parent = parent->getParentOp();
      }
      if (parent == alloc) {
        // nothing to hoist
        continue;
      }
      alloc->remove();
      rewriter.setInsertionPoint(parent);
      rewriter.insert(alloc);

      for (auto user : alloc->getUsers()) {
        if (llvm::isa<cnm::FreeWorkgroupOp>(user)) {
          user->remove();
          rewriter.setInsertionPointAfter(parent);
          rewriter.insert(user);
          break;
        }
      }
    }

    // Hoist buffer alloc ops after their respective workgroup op.
    llvm::SmallVector<cnm::AllocOp> bufAllocs;
    fun->walk([&](cnm::AllocOp op) { bufAllocs.push_back(op); });

    for (auto bufAlloc : bufAllocs) {
      Operation *parent = bufAlloc;
      while (parent->getParentOp() && parent->getParentOp() != fun &&
             !parent->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
        parent = parent->getParentOp();
      }
      if (parent == bufAlloc)
        continue;
      Operation *wgDef = bufAlloc.getWg().getDefiningOp();
      if (!wgDef)
        continue;
      bufAlloc->remove();
      rewriter.setInsertionPointAfter(wgDef);
      rewriter.insert(bufAlloc);
    }
  }
};

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
    auto *root = getOperation();
    if (root->getNumRegions() == 0)
      return;

    // Find the nearest IsolatedFromAbove ancestor of `op` that is still
    // a descendant of (or equal to) `root`. Returns nullptr if none found.
    auto findScope = [&](Operation *op) -> Operation * {
      Operation *scope = op->getParentOp();
      while (scope && scope != root) {
        if (scope->hasTrait<OpTrait::IsIsolatedFromAbove>())
          return scope;
        scope = scope->getParentOp();
      }
      // No isolated ancestor found below root — hoist to root itself.
      return root;
    };

    // Find the direct child of `scope` that is an ancestor of `op`.
    auto childOfScope = [](Operation *op, Operation *scope) -> Operation * {
      Operation *child = op;
      while (child->getParentOp() && child->getParentOp() != scope)
        child = child->getParentOp();
      return child;
    };

    OpBuilder rewriter(&getContext());

    llvm::SmallVector<cnm::WorkgroupOp> wgOps;
    root->walk([&](cnm::WorkgroupOp op) { wgOps.push_back(op); });

    for (auto wgOp : wgOps) {
      Operation *scope = findScope(wgOp);
      Operation *parent = childOfScope(wgOp, scope);
      if (parent == wgOp)
        continue; // already at scope level

      wgOp->remove();
      rewriter.setInsertionPoint(parent);
      rewriter.insert(wgOp);

      for (auto *user : wgOp->getUsers()) {
        if (llvm::isa<cnm::FreeWorkgroupOp>(user)) {
          user->remove();
          rewriter.setInsertionPointAfter(parent);
          rewriter.insert(user);
          break;
        }
      }
    }

    llvm::SmallVector<cnm::DeclareBufferOp> bufAllocs;
    root->walk([&](cnm::DeclareBufferOp op) { bufAllocs.push_back(op); });

    for (auto bufAlloc : bufAllocs) {
      Operation *scope = findScope(bufAlloc);
      Operation *parent = childOfScope(bufAlloc, scope);
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


#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>

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
      auto newAlloc = rewriter.clone(*alloc, mapper);
      alloc.replaceAllUsesWith(newAlloc);
      alloc->erase();
    }
  }
};
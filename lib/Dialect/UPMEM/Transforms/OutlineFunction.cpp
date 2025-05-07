#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include <cinm-mlir/Utils/CinmUtils.h>

#include "mlir/IR/SymbolTable.h"
// #include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/Regex.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_UPMEMOUTLINEKERNELPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

namespace {
/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static LogicalResult outlineKernelFuncImpl(func::FuncOp parent,
                                           upmem::InlineDpuProgramOp launchOp,
                                           SymbolTable &kernelContainer) {
  OpBuilder builder(parent->getContext());
  Location loc = launchOp.getLoc();
  auto hierarchy = launchOp.getWg().getType();

  auto outlinedFunc = builder.create<upmem::DpuProgramOp>(
      loc, parent.getName(), hierarchy.getNumTaskletsPerDpu());
  outlinedFunc.getBody().takeBody(launchOp.getBody());
  // this also does renaming if name is not unique
  kernelContainer.insert(outlinedFunc);

  SymbolTable table(
      SymbolTable::getNearestSymbolTable(launchOp->getParentOp()));

  auto ref = upmem::getSymbolPath(table, outlinedFunc);
  if (failed(ref))
    return failure();

  builder.setInsertionPoint(launchOp);

  builder.create<upmem::WaitForOp>(loc, launchOp.getWg(), *ref);

  launchOp->erase();
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
struct UPMEMOutlineKernelPass
    : public impl::UPMEMOutlineKernelPassBase<UPMEMOutlineKernelPass> {
  using Base::Base;

  void runOnOperation() final;

  ModuleOp createKernelModule(StringRef moduleName,
                              const SymbolTable &parentSymbolTable);
};

void UPMEMOutlineKernelPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);

  auto *context = getOperation().getContext();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());

  auto kernelModule = builder.create<ModuleOp>(module.getLoc(), "dpu_kernels");
  auto *newBlock = &kernelModule.getBodyRegion().front();

  SymbolTable kernelModuleSymTable(kernelModule);
  builder.setInsertionPointToStart(newBlock);

  for (auto func : getOperation().getOps<func::FuncOp>()) {
    Block::iterator insertPt(func->getNextNode());
    auto res = func.walk([&](upmem::InlineDpuProgramOp op) {
      OpBuilder::InsertionGuard guard(builder);
      if (outlineKernelFuncImpl(func, op, kernelModuleSymTable).failed())
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      signalPassFailure();
      return;
    }
  }
}
} // namespace mlir

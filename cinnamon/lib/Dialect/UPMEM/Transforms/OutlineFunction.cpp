#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include <cinm-mlir/Utils/CinmUtils.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Regex.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_UPMEMOUTLINEKERNELPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

namespace {
/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static upmem::UPMEMFuncOp outlineKernelFuncImpl(func::FuncOp parent,
                                                upmem::LaunchOp launchOp,
                                                SymbolTable& kernelContainer) {
  OpBuilder builder(parent->getContext());
  Location loc = launchOp.getLoc();

  auto outlinedFunc = builder.create<upmem::UPMEMFuncOp>(
      loc, parent.getName(),
      launchOp.getDeviceHierarchy().getType().getNumTaskletsPerDpu());
  // this also does renaming if name is not unique
  kernelContainer.insert(outlinedFunc);

  IRMapping map;
  Block &outlinedEntryBlock = outlinedFunc.getBody().emplaceBlock();

  Region &launchOpBody = launchOp.getBody();
  Block &launchOpEntry = launchOpBody.front();

  ///  CLone the region into the func, we remap the block arguments
  {
    auto taskletArg = launchOpEntry.getArgument(2);
    auto taskletId = builder.create<upmem::TaskletIDOp>(taskletArg.getLoc());
    map.map(taskletArg, taskletId);

    builder.setInsertionPointToEnd(&outlinedEntryBlock);

    for (auto &op : launchOpEntry.without_terminator()) {
      builder.clone(op, map);
    }
    builder.create<upmem::ReturnOp>(launchOpEntry.getTerminator()->getLoc());
  }

  return outlinedFunc;
}

static void convertToLaunchFuncOp(upmem::LaunchOp launchOp,
                                  upmem::UPMEMFuncOp kernelFunc) {
  OpBuilder builder(launchOp);
  // The launch op has an optional dynamic shared memory size. If it doesn't
  // exist, we use zero.
  Value asyncToken = launchOp.getAsyncToken();
  auto launchFunc = builder.create<upmem::LaunchFuncOp>(
      launchOp.getLoc(), kernelFunc, launchOp.getDeviceHierarchy(),
      launchOp.getDynamicSharedMemorySize(), ValueRange{},
      asyncToken ? asyncToken.getType() : nullptr,
      launchOp.getAsyncDependencies());
  launchOp.replaceAllUsesWith(launchFunc);
  launchOp.erase();
}
} // namespace

//===----------------------------------------------------------------------===//
struct UPMEMOutlineKernelPass
    : public impl::UPMEMOutlineKernelPassBase<UPMEMOutlineKernelPass> {
  using Base::Base;

  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &) const override {}
  upmem::UPMEMModuleOp createKernelModule(StringRef moduleName,
                                          const SymbolTable &parentSymbolTable);
};

void UPMEMOutlineKernelPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);

  auto *context = getOperation().getContext();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto kernelModule =
      builder.create<upmem::UPMEMModuleOp>(module.getLoc(), "dpu_kernels");
  kernelModule.getBodyRegion().emplaceBlock();
  SymbolTable kernelModuleSymTable(kernelModule);
  builder.setInsertionPointToStart(&kernelModule.getBodyRegion().front());

  for (auto func : getOperation().getOps<func::FuncOp>()) {
    Block::iterator insertPt(func->getNextNode());
    func.walk([&](upmem::LaunchOp op) {
      OpBuilder::InsertionGuard guard(builder);
      upmem::UPMEMFuncOp outlinedFunc =
          outlineKernelFuncImpl(func, op, kernelModuleSymTable);

      //     // Potentially changes signature, pulling in constants.
      convertToLaunchFuncOp(op, outlinedFunc);
      return WalkResult::advance();
    });
  }
}
} // namespace mlir

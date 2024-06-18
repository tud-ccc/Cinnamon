#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Regex.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_UPMEMOUTLINEKERNELPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static upmem::UPMEMFuncOp outlineKernelFuncImpl(upmem::LaunchOp launchOp,
                                                StringRef kernelFnName,
                                                SetVector<Value> &operands) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());

  auto outlinedFunc = builder.create<upmem::UPMEMFuncOp>(loc, kernelFnName);

  outlinedFunc.setNumTasklets(
      launchOp.getDeviceHierarchy().getType().getNumTaskletsPerDpu());

  IRMapping map;
  Region &outlinedFuncBody = outlinedFunc.getBody();
  Block &outlinedEntryBlock = outlinedFuncBody.front();

  Region &launchOpBody = launchOp.getBody();
  Block &launchOpEntry = launchOpBody.front();

  ///  CLone the region into the func, we remap the block arguments
  {
    auto taskletArg = launchOpEntry.getArgument(2);
    auto taskletId = builder.create<upmem::TaskletIDOp>(taskletArg.getLoc());
    map.map(taskletArg, taskletId);
    outlinedEntryBlock.push_back(taskletId);

    builder.setInsertionPointToEnd(&outlinedEntryBlock);

    for (auto &op : launchOpEntry.without_terminator()) {
      builder.clone(op, map);
    }
    builder.create<upmem::ReturnOp>(launchOpEntry.getTerminator()->getLoc());
  }

  return outlinedFunc;
}

static void convertToLaunchFuncOp(upmem::LaunchOp launchOp,
                                  upmem::UPMEMFuncOp kernelFunc,
                                  ValueRange operands) {
  OpBuilder builder(launchOp);
  // The launch op has an optional dynamic shared memory size. If it doesn't
  // exist, we use zero.
  Value asyncToken = launchOp.getAsyncToken();
  auto launchFunc = builder.create<upmem::LaunchFuncOp>(
      launchOp.getLoc(), kernelFunc, launchOp.getDeviceHierarchy(),
      launchOp.getDynamicSharedMemorySize(), operands,
      asyncToken ? asyncToken.getType() : nullptr,
      launchOp.getAsyncDependencies());
  launchOp.replaceAllUsesWith(launchFunc);
  launchOp.erase();
}

static void createFreeDPUsOp(upmem::GatherOp gatherOp) {
  OpBuilder builder(gatherOp);
  builder.setInsertionPointAfter(gatherOp);
  builder.create<upmem::FreeDPUsOp>(gatherOp.getLoc(), gatherOp.getHierarchy());
}

//===----------------------------------------------------------------------===//
struct UPMEMOutlineKernelPass
    : public impl::UPMEMOutlineKernelPassBase<UPMEMOutlineKernelPass> {
  using Base::Base;

  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &registry) const override {}
  upmem::UPMEMModuleOp createKernelModule(upmem::UPMEMFuncOp kernelFunc,
                                          StringRef moduleName,
                                          const SymbolTable &parentSymbolTable);
};

void UPMEMOutlineKernelPass::runOnOperation() {
  SymbolTable symbolTable(getOperation());
  bool modified = false;
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    Block::iterator insertPt(func->getNextNode());
    auto funcWalkResult = func.walk([&](upmem::LaunchOp op) {
      SetVector<Value> operands;

      upmem::UPMEMFuncOp outlinedFunc =
          outlineKernelFuncImpl(op, "main", operands);

      auto moduleName = (func.getName() + "_dpu").str();
      auto kernelModule =
          createKernelModule(outlinedFunc, moduleName, symbolTable);
      symbolTable.insert(kernelModule, insertPt);

      //     // Potentially changes signature, pulling in constants.
      convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
      modified = true;
      return WalkResult::advance();
    });
    if (funcWalkResult.wasInterrupted())
      return signalPassFailure();

    // Inserting free dpus after the last gather operation on a dpu allocation
    auto funcWalkResult2 = func.walk([&](upmem::AllocDPUsOp op) {
      SetVector<Value> operands;
      upmem::GatherOp lastOp;
      for (auto user : op.getHierarchyShape().getUsers()) {
        if (dyn_cast_or_null<upmem::GatherOp>(user)) {
          lastOp = dyn_cast_or_null<upmem::GatherOp>(user);
        }
      }
      createFreeDPUsOp(lastOp);
      modified = true;
      return WalkResult::advance();
    });
    if (funcWalkResult2.wasInterrupted())
      return signalPassFailure();
  }

  // // If any new module was inserted in this module, annotate this module as
  // // a container module.
  // if (modified)
  //   getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
  //                           UnitAttr::get(&getContext()));
}

upmem::UPMEMModuleOp UPMEMOutlineKernelPass::createKernelModule(
    upmem::UPMEMFuncOp kernelFunc, StringRef moduleName,
    const SymbolTable &parentSymbolTable) {
  auto *context = getOperation().getContext();
  OpBuilder builder(context);
  auto kernelModule =
      builder.create<upmem::UPMEMModuleOp>(kernelFunc.getLoc(), moduleName);

  SymbolTable symbolTable(kernelModule);
  symbolTable.insert(kernelFunc);

  SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
  while (!symbolDefWorklist.empty()) {
    if (std::optional<SymbolTable::UseRange> symbolUses =
            SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
      for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
        StringRef symbolName =
            cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
        if (symbolTable.lookup(symbolName))
          continue;

        Operation *symbolDefClone =
            parentSymbolTable.lookup(symbolName)->clone();
        symbolDefWorklist.push_back(symbolDefClone);
        symbolTable.insert(symbolDefClone);
      }
    }
  }

  return kernelModule;
}

} // namespace mlir

#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.h"
#include "cinm-mlir/Dialect/Hbmpim/Transforms/Passes.h"

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

#define GEN_PASS_DEF_HBMPIMOUTLINEKERNELPASS
#include "cinm-mlir/Dialect/Hbmpim/Transforms/Passes.h.inc"

namespace {
static hbmpim::HbmpimFuncOp outlineKernelFuncImpl(func::FuncOp parent,
                                                hbmpim::SetDeviceConfigOp setOp,
                                                SymbolTable& kernelContainer) {
  OpBuilder builder(parent->getContext());
  Location loc = setOp.getLoc();

//   auto outlinedFunc = builder.create<hbmpim::HbmpimFuncOp>(
//       loc, parent.getName());
  // this also does renaming if name is not unique
//   kernelContainer.insert(outlinedFunc);

//   IRMapping map;
//   Block &outlinedEntryBlock = outlinedFunc.getBody().emplaceBlock();

//   Region &launchOpBody = launchOp.getBody();
//   Block &launchOpEntry = launchOpBody.front();

  ///  Clone the region into the func, we remap the block arguments
  {
    // auto taskletArg = launchOpEntry.getArgument(2);
    // auto taskletId = builder.create<upmem::TaskletIDOp>(taskletArg.getLoc());
    // map.map(taskletArg, taskletId);

    // builder.setInsertionPointToEnd(&outlinedEntryBlock);
    // Operation *toClone = setOp;
    // builder.clone(*toClone, map);
    // for (auto &use : setOp.getResult().getUses()){
    //     Operation *toClone = use.getOwner();
    //     builder.clone(*toClone, map);
    // }
    // for (auto &op : launchOpEntry.without_terminator()) {
    //   builder.clone(op, map);
    // }
    // builder.create<hbmpim::ReturnOp>(loc);
  }

//   return outlinedFunc;
  return nullptr;
}

static void convertToLaunchFuncOp(hbmpim::SetDeviceConfigOp setOp,
                                  hbmpim::HbmpimFuncOp kernelFunc) {
  OpBuilder builder(setOp);
//   auto launchFunc = builder.create<hbmpim::LaunchFuncOp>(
//       setOp.getLoc(), kernelFunc, ValueRange{});
//   launchOp.replaceAllUsesWith(launchFunc);
//   launchOp.erase();
}
} // namespace

//===----------------------------------------------------------------------===//
struct HbmpimOutlineKernelPass
    : public impl::HbmpimOutlineKernelPassBase<HbmpimOutlineKernelPass> {
  using Base::Base;

  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &) const override {}
  hbmpim::HbmpimModuleOp createKernelModule(StringRef moduleName,
                                          const SymbolTable &parentSymbolTable);
};

void HbmpimOutlineKernelPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);

  auto *context = getOperation().getContext();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto kernelModule =
      builder.create<hbmpim::HbmpimModuleOp>(module.getLoc(), "hbmpim_kernels");
  kernelModule.getBodyRegion().emplaceBlock();
  SymbolTable kernelModuleSymTable(kernelModule);
  builder.setInsertionPointToStart(&kernelModule.getBodyRegion().front());

  for (auto func : getOperation().getOps<func::FuncOp>()) {
    Block::iterator insertPt(func->getNextNode());
    func.walk([&](hbmpim::SetDeviceConfigOp op) {
      OpBuilder::InsertionGuard guard(builder);
      hbmpim::HbmpimFuncOp outlinedFunc =
          outlineKernelFuncImpl(func, op, kernelModuleSymTable);

      //     // Potentially changes signature, pulling in constants.
      convertToLaunchFuncOp(op, outlinedFunc);
      return WalkResult::advance();
    });
  }
}
} // namespace mlir

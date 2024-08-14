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

    auto outlinedFunc = builder.create<hbmpim::HbmpimFuncOp>(
      loc, parent.getName());

    kernelContainer.insert(outlinedFunc);

    IRMapping map;
    Block &outlinedEntryBlock = outlinedFunc.getBody().emplaceBlock();

    builder.setInsertionPointToStart(&outlinedEntryBlock);
    Operation *toClone = setOp;
    builder.clone(*toClone, map);

    for (auto &use : setOp.getResult().getUses()){

        if (dyn_cast<hbmpim::PreloadNoReplacementOp>(use.getOwner())){
            auto oldOp = dyn_cast<hbmpim::PreloadNoReplacementOp>(use.getOwner());
            builder.clone(*oldOp.getStartRow().getDefiningOp(), map);
            builder.clone(*oldOp.getStartCol().getDefiningOp(), map);
            builder.create<hbmpim::SimulatorPreloadNoReplacementOp>(loc, map.lookup(oldOp.getConfig()),
                 map.lookup(oldOp.getStartRow()), map.lookup(oldOp.getStartCol()));
            // oldOp.erase();
        }
    }
    for (auto &use : setOp.getResult().getUses()){
        if (dyn_cast<hbmpim::ExecuteElementwiseOp>(use.getOwner())){
            auto oldOp = dyn_cast<hbmpim::ExecuteElementwiseOp>(use.getOwner());
            builder.clone(*oldOp.getDim().getDefiningOp(), map);
            builder.clone(*oldOp.getInput0row().getDefiningOp(), map);
            builder.clone(*oldOp.getResultRow().getDefiningOp(), map);
            builder.clone(*oldOp.getInput1row().getDefiningOp(), map);
            builder.create<hbmpim::ExecuteElementwiseOp>(loc, map.lookup(oldOp.getConfig()),
                 map.lookup(oldOp.getDim()), oldOp.getBankType(), oldOp.getKernelType(), 
                 map.lookup(oldOp.getInput0row()), map.lookup(oldOp.getResultRow()),
                 map.lookup(oldOp.getInput1row()));
            // oldOp.erase();
        }
    }
    builder.create<hbmpim::ReturnOp>(loc);
    // setOp.erase();
    return outlinedFunc;
}

static void convertToLaunchFuncOp(hbmpim::SetDeviceConfigOp setOp,
                                  hbmpim::HbmpimFuncOp kernelFunc) {
    OpBuilder builder(setOp);
    // for (auto &use : setOp.getResult().getUses())
    //     use.getOwner()->erase();
    // setOp.erase();
  
    for (auto &use : setOp.getResult().getUses()){
        if (dyn_cast<hbmpim::ExecuteElementwiseOp>(use.getOwner())){
            auto oldOp = dyn_cast<hbmpim::ExecuteElementwiseOp>(use.getOwner());
            auto launchFunc = builder.create<hbmpim::HbmpimLaunchFuncOp>(
                oldOp.getLoc(), kernelFunc, ValueRange{});
            // oldOp->replace(launchFunc);
        }
        // use.getOwner()->erase();

    } 
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
  SmallVector<Operation*> operationsToDelete;
    for (auto func : getOperation().getOps<func::FuncOp>()) {
        func.walk([&](Operation *op) {
            if (auto setOp = dyn_cast<hbmpim::SetDeviceConfigOp>(op)){
                for (auto &use : setOp.getResult().getUses())
                    use.getOwner()->erase(); 
            }
            return WalkResult::advance();
    });
  }
}
} // namespace mlir

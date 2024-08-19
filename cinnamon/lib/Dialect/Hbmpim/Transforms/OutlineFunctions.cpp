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
                                                hbmpim::LaunchOp launchOp, 
                                                SymbolTable& kernelContainer) {
    OpBuilder builder(parent->getContext());
    Location loc = launchOp.getLoc();
    Region &launchOpBody = launchOp.getBody();
    Block &launchOpEntry = launchOpBody.front();
    hbmpim::SetDeviceConfigOp setOp = dyn_cast<hbmpim::SetDeviceConfigOp>(launchOp.getConfig().getDefiningOp());
    auto outlinedFunc = builder.create<hbmpim::HbmpimFuncOp>(
      loc, parent.getName());

    kernelContainer.insert(outlinedFunc);

    IRMapping map;
    Block &outlinedEntryBlock = outlinedFunc.getBody().emplaceBlock();

    builder.setInsertionPointToStart(&outlinedEntryBlock);
    Operation *toClone = setOp;
    builder.clone(*toClone, map);
    for (auto &op : launchOpEntry.without_terminator()) {
        if (dyn_cast<hbmpim::PreloadNoReplacementOp>(op)){
            auto oldOp = dyn_cast<hbmpim::PreloadNoReplacementOp>(op);
            builder.create<hbmpim::SimulatorPreloadNoReplacementOp>(loc,
                 map.lookup(oldOp.getStartRow()), map.lookup(oldOp.getStartCol()));
        } else if (dyn_cast<hbmpim::PreloadGemvOp>(op)){
            auto oldOp = dyn_cast<hbmpim::PreloadGemvOp>(op);
            auto operandShape = dyn_cast<ShapedType>(oldOp.getInBuffer().getType()).getShape();
            builder.create<hbmpim::SimulatorPreloadGemvOp>(loc, 
                 map.lookup(oldOp.getStartRow()), map.lookup(oldOp.getStartCol()), builder.getDenseI64ArrayAttr(operandShape));
        } else if (dyn_cast<hbmpim::ReadResultOp>(op)){
            auto oldOp = dyn_cast<hbmpim::ReadResultOp>(op);
            auto operandShape = dyn_cast<ShapedType>(oldOp.getOutBuffer().getType()).getShape();
            builder.create<hbmpim::SimulatorReadResultOp>(loc, 
                 oldOp.getBankType(), map.lookup(oldOp.getOutDim()), 
                 map.lookup(oldOp.getBaseAddr()), map.lookup(oldOp.getStartRow()),
                 map.lookup(oldOp.getStartCol()), builder.getDenseI64ArrayAttr(operandShape));
        } else if (dyn_cast<hbmpim::ReadDataOp>(op)){
            auto oldOp = dyn_cast<hbmpim::ReadDataOp>(op);
            auto operandShape = dyn_cast<ShapedType>(oldOp.getOutBuffer().getType()).getShape();
            builder.create<hbmpim::SimulatorReadDataOp>(loc, 
                 map.lookup(oldOp.getStartRow()), map.lookup(oldOp.getStartCol()),
                 builder.getDenseI64ArrayAttr(operandShape));
        } else {
            builder.clone(op, map);
        }
    }
  
    builder.create<hbmpim::ReturnOp>(loc);
    return outlinedFunc;
}

static void convertToLaunchFuncOp(hbmpim::LaunchOp launchOp,
                                  hbmpim::HbmpimFuncOp kernelFunc) {
    OpBuilder builder(launchOp);
    auto launchFunc = builder.create<hbmpim::HbmpimLaunchFuncOp>(
        launchOp.getLoc(), kernelFunc, ValueRange{});
    launchOp.erase();
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
    func.walk([&](hbmpim::LaunchOp op) {
      OpBuilder::InsertionGuard guard(builder);
      hbmpim::HbmpimFuncOp outlinedFunc =
          outlineKernelFuncImpl(func, op, kernelModuleSymTable);
      convertToLaunchFuncOp(op, outlinedFunc);
      return WalkResult::advance();
    });
  }
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    Block::iterator insertPt(func->getNextNode());
    func.walk([&](hbmpim::SetDeviceConfigOp op) {
        // for (mlir::OpOperand &use : op.getResult().getUses()){
            // auto op = use.getOwner();
            // op->print(llvm::dbgs());
            // llvm::dbgs() << "one use " << use << "\n";
        // }
        // op.erase();

        return WalkResult::advance();
    });
  }
}
} // namespace mlir

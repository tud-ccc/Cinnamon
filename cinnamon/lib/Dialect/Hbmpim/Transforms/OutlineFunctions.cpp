#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.h"
#include "cinm-mlir/Dialect/Hbmpim/Transforms/Passes.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"

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
// static int getInBufferPosition(hbmpim::LaunchOp op, Value inToSearch){
//   int index = 0;
//   for (auto input : op.getParams()) 
//     if(input == inToSearch) return index;
//     index ++;
//   return -1;
// }
// MemRefType convertTensorToMemref(ShapedType ty) {
//   if (ty.isa<MemRefType>())
//     return ty.cast<MemRefType>();

//   return MemRefType::get(ty.getShape(), ty.getElementType());
// }

// Value getOriginalifUnrealizedCast(Value val){
//   if (dyn_cast<UnrealizedConversionCastOp>(val.getDefiningOp()))
//     return val.getDefiningOp()->getOperand(0);
  
//   return val;
// }


// static hbmpim::HostPreloadGemvOp insertHostPreloadGemvOperand(OpBuilder builder, Location loc, hbmpim::LaunchOp launchOp, hbmpim::PreloadGemvOp op, IRMapping map){
//   ShapedType inputType = dyn_cast<ShapedType>(op.getInBuffer().getType());
//   if(inputType.getShape().size() == 2){
//     // Assumption: the operand with two ranks is in the first operand.
//     auto originalVal = getOriginalifUnrealizedCast(launchOp.getParams()[0]);
//     // ShapedType inputTy = dyn_cast<ShapedType>(orignalVal.getType());
//     // Value inputAsMemref = createOrFoldUnrealizedConversionCast(
//     //           op.getLoc(), builder, convertTensorToMemref(inputTy), orignalVal);
//     // return builder.create<hbmpim::HostPreloadGemvOp>(loc, inputAsMemref, hbmpim::DataDimType::weight_npbst_);
//     return builder.create<hbmpim::HostPreloadGemvOp>(loc, originalVal, hbmpim::DataDimType::weight_npbst_);
//   } else if (inputType.getShape().size() == 1){
//     // Assumption: the operand with one rank is in the second operand.
//     auto originalVal = getOriginalifUnrealizedCast(launchOp.getParams()[1]);
//     // ShapedType inputTy = dyn_cast<ShapedType>(orignalVal.getType());
//     // Value inputAsMemref = createOrFoldUnrealizedConversionCast(
//     //           op.getLoc(), builder, convertTensorToMemref(inputTy), orignalVal);
//     // builder.create<hbmpim::HostPreloadGemvOp>(loc, inputAsMemref, hbmpim::DataDimType::input_npbst_);
//     return builder.create<hbmpim::HostPreloadGemvOp>(loc, originalVal, hbmpim::DataDimType::input_npbst_);
//   }
// }


// static LogicalResult outlineKernelFuncImpl(func::FuncOp parent,
//                                                 hbmpim::LaunchOp launchOp, 
//                                                 SymbolTable& kernelContainer) {
//     OpBuilder builder(parent->getContext());
//     Location loc = launchOp.getLoc();
//     Region &launchOpBody = launchOp.getBody();
//     Block &launchOpEntry = launchOpBody.front();
//     hbmpim::SetDeviceConfigOp setOp = dyn_cast<hbmpim::SetDeviceConfigOp>(launchOp.getConfig().getDefiningOp());
//     auto outlinedFunc = builder.create<hbmpim::HbmpimFuncOp>(
//       loc, parent.getName());

//     kernelContainer.insert(outlinedFunc);

//     IRMapping map;
//     Block &outlinedEntryBlock = outlinedFunc.getBody().emplaceBlock();
//     auto savedPoint = builder.saveInsertionPoint();
//     builder.setInsertionPointToStart(&outlinedEntryBlock);
//     auto wgShape = launchOp.getConfig().getType().getShape();
//     builder.create<hbmpim::SimulatorSetDeviceConfigOp>(loc, wgShape[3], wgShape[2], wgShape[1], wgShape[0]);
//     for (auto &op : launchOpEntry.without_terminator()) {
//         if (dyn_cast<hbmpim::PreloadNoReplacementOp>(op)){
//             // auto oldOp = dyn_cast<hbmpim::PreloadNoReplacementOp>(op);
//             // builder.create<hbmpim::SimulatorPreloadNoReplacementOp>(loc,
//             //      map.lookup(oldOp.getStartRow()), map.lookup(oldOp.getStartCol()));
//         } else if (dyn_cast<hbmpim::PreloadGemvOp>(op)){
//           auto currPoint = builder.saveInsertionPoint();
//           builder.restoreInsertionPoint(savedPoint);
//           auto oldOp = dyn_cast<hbmpim::PreloadGemvOp>(op);
//           auto op = insertHostPreloadGemvOperand(builder, loc, launchOp, oldOp, map);
//           builder.restoreInsertionPoint(currPoint);
//           // auto operandShape = dyn_cast<ShapedType>(oldOp.getInBuffer().getType()).getShape();
//           //   builder.create<hbmpim::SimulatorPreloadGemvOp>(loc, 
//           //        map.lookup(oldOp.getStartRow()), map.lookup(oldOp.getStartCol()), builder.getDenseI64ArrayAttr(operandShape));
//         } else if (dyn_cast<hbmpim::ReadResultOp>(op)){
//             // auto oldOp = dyn_cast<hbmpim::ReadResultOp>(op);
//             // auto operandShape = dyn_cast<ShapedType>(oldOp.getOutBuffer().getType()).getShape();
//             // builder.create<hbmpim::SimulatorReadResultOp>(loc, 
//             //      oldOp.getBankType(), map.lookup(oldOp.getOutDim()), 
//             //      map.lookup(oldOp.getBaseAddr()), map.lookup(oldOp.getStartRow()),
//             //      map.lookup(oldOp.getStartCol()), builder.getDenseI64ArrayAttr(operandShape));
//         } else if (dyn_cast<hbmpim::ReadDataOp>(op)){
//             // auto oldOp = dyn_cast<hbmpim::ReadDataOp>(op);
//             // auto operandShape = dyn_cast<ShapedType>(oldOp.getOutBuffer().getType()).getShape();
//             // builder.create<hbmpim::SimulatorReadDataOp>(loc, 
//             //      map.lookup(oldOp.getStartRow()), map.lookup(oldOp.getStartCol()),
//             //      builder.getDenseI64ArrayAttr(operandShape));
//         } else if (dyn_cast<hbmpim::ExecuteGemvOp>(op)) {
//         } else {
//             builder.clone(op, map);
//         }
//     }
  
//     builder.create<hbmpim::ReturnOp>(loc);
//     auto launchFunc = builder.create<hbmpim::HbmpimLaunchFuncOp>(
//         launchOp.getLoc(), outlinedFunc, ValueRange{});
//     replaceLaunchWith.push_back(launchFunc); 
//     // launchOp.erase();
//     launchOp->replaceAllUsesWith(replaceLaunchWith);
//     return success();
// }

// static void convertToLaunchFuncOp(hbmpim::LaunchOp launchOp,
//                                   hbmpim::HbmpimFuncOp kernelFunc) {
//     OpBuilder builder(launchOp);
    
// }

// } // namespace

//===----------------------------------------------------------------------===//
// struct HbmpimOutlineKernelPass
//     : public impl::HbmpimOutlineKernelPassBase<HbmpimOutlineKernelPass> {
//   using Base::Base;

//   void runOnOperation() final;

//   void getDependentDialects(DialectRegistry &) const override {}
//   hbmpim::HbmpimModuleOp createKernelModule(StringRef moduleName,
//                                           const SymbolTable &parentSymbolTable);
// };

// void HbmpimOutlineKernelPass::runOnOperation() {
//   ModuleOp module = getOperation();
//   SymbolTable symbolTable(module);

//   auto *context = getOperation().getContext();
//   OpBuilder builder(context);
//   builder.setInsertionPointToEnd(&module.getBodyRegion().front());
//   auto kernelModule =
//       builder.create<hbmpim::HbmpimModuleOp>(module.getLoc(), "hbmpim_kernels");
//   kernelModule.getBodyRegion().emplaceBlock();
//   SymbolTable kernelModuleSymTable(kernelModule);
//   builder.setInsertionPointToStart(&kernelModule.getBodyRegion().front());
    
//   for (auto func : getOperation().getOps<func::FuncOp>()) {
//     Block::iterator insertPt(func->getNextNode());
//     func.walk([&](hbmpim::LaunchOp op) {
//       OpBuilder::InsertionGuard guard(builder);
//       // hbmpim::HbmpimFuncOp outlinedFunc =
//       outlineKernelFuncImpl(func, op, kernelModuleSymTable);
//       // convertToLaunchFuncOp(op, outlinedFunc);
//       return WalkResult::advance();
//     });
//   }
  // for (auto func : getOperation().getOps<func::FuncOp>()) {
  //   Block::iterator insertPt(func->getNextNode());
  //   func.walk([&](hbmpim::SetDeviceConfigOp op) {
  //       for (mlir::OpOperand &use : op.getResult().getUses()){
  //           auto op = use.getOwner();
  //           op->print(llvm::dbgs());
  //           llvm::dbgs() << "one use " << use << "\n";
  //       }
  //       op.erase();

  //       return WalkResult::advance();
  //   });
  // }
// }


struct LaunchOpRewriter
    : public OpConversionPattern<hbmpim::LaunchOp> {
public:
  using OpConversionPattern<hbmpim::LaunchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hbmpim::LaunchOp launchOp, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = launchOp->getParentOp()->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(module);
    auto loc = launchOp.getLoc();
    auto *context = rewriter.getContext();
    auto origStartPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToEnd(&module.getBodyRegion().front());
    auto kernelModule =
      rewriter.create<hbmpim::HbmpimModuleOp>(module.getLoc(), "hbmpim_kernels");
    kernelModule.getBodyRegion().emplaceBlock();
    SymbolTable kernelModuleSymTable(kernelModule);
    rewriter.setInsertionPointToStart(&kernelModule.getBodyRegion().front());


    func::FuncOp parent = launchOp->getParentOfType<func::FuncOp>();
    Region &launchOpBody = launchOp.getBody();
    Block &launchOpEntry = launchOpBody.front();
    auto outlinedFunc = rewriter.create<hbmpim::HbmpimFuncOp>(
      loc, parent.getName());
    // OpBuilder builder(parent->getContext());
    
    IRMapping map;
    Block &outlinedEntryBlock = outlinedFunc.getBody().emplaceBlock();
    rewriter.setInsertionPointToStart(&outlinedEntryBlock);


    auto wgShape = launchOp.getConfig().getType().getShape();
    rewriter.create<hbmpim::SimulatorSetDeviceConfigOp>(loc, wgShape[3], wgShape[2], wgShape[1], wgShape[0]);
    for (auto &op : launchOpEntry.without_terminator()) {
      rewriter.clone(op, map);
    }

    kernelModuleSymTable.insert(outlinedFunc);
    rewriter.create<hbmpim::ReturnOp>(loc);
    rewriter.restoreInsertionPoint(origStartPoint);
    auto launchFunc = rewriter.create<hbmpim::HbmpimLaunchFuncOp>(
       loc, outlinedFunc, ValueRange{});
    rewriter.eraseOp(launchOp);
    return success();
  }
};

} // namespace

void populateHbmpimLaunchOpRewritePatterns(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
    patterns.add<LaunchOpRewriter>(patterns.getContext());

}

struct HbmpimOutlineKernelPass
    : public impl::HbmpimOutlineKernelPassBase<HbmpimOutlineKernelPass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    TypeConverter converter;
    // converter.addSourceMaterialization(addUnrealizedCast);
    // converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateHbmpimLaunchOpRewritePatterns(converter, patterns);

    ConversionTarget target(getContext());
    target.addIllegalOp<hbmpim::LaunchOp>();
    
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createHbmpimOutlineKernelPassPass() {
  return std::make_unique<HbmpimOutlineKernelPass>();
}

} // namespace mlir

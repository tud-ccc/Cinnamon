#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

#include <llvm/Support/Regex.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"




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
  Region &launchOpBody = launchOp.getBody();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  SetVector<Value> outsideValues;
  getUsedValuesDefinedAbove(launchOpBody, outsideValues);

//   SmallVector<Type, 4> kernelOperandTypes;
//   kernelOperandTypes.reserve(operands.size());
//   for (Value operand : operands) {
//     kernelOperandTypes.push_back(operand.getType());
//   }
  FunctionType type =
      FunctionType::get(launchOp.getContext(), {}, {});
  auto outlinedFunc = builder.create<upmem::UPMEMFuncOp>(
      loc, kernelFnName, type);
      
      
  outlinedFunc->setAttr(upmem::UPMEMDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());


  IRMapping map;
  Region &outlinedFuncBody = outlinedFunc.getBody();
//   outlinedFuncBody.getBlocks().pop_back();


 // Block &entryBlock = outlinedFuncBody.front();
//   for (const auto &operand : enumerate(operands))
//     map.map(operand.value(), entryBlock.getArgument(operand.index()));


  Block &launchOpEntry = launchOpBody.front();
  builder.setInsertionPointToStart(&outlinedFuncBody.front());

  auto cloneOptions =
    Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  
    for (auto value : outsideValues){
        // auto copy = value.getDefiningOp()->clone(map, cloneOptions);
        auto copy=builder.clone(*value.getDefiningOp());
        map.map(value.getDefiningOp(), copy);
    }
    
  {
//     Block *newBlock = new Block();
//  map.map(&launchOpEntry, newBlock);
//  auto taskletId = builder.create<upmem::TaskletIDOp>(launchOpEntry.getArgument(0)

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    // for (auto arg : block.getArguments())
    //   if (!mapper.contains(arg))
    //     mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));

    // dest->getBlocks().insert(destPos, newBlock);
  }

//   Block *clonedLaunchOpEntry = map.lookup(&launchOpEntry);
  builder.setInsertionPointToEnd(&outlinedFuncBody.front());
//   builder.create<cf::BranchOp>(loc, clonedLaunchOpEntry);
    llvm::dbgs() << "Hello\n";
  outlinedFunc.walk([](upmem::TerminatorOp op) {
    llvm::dbgs() << "Found it\n";
    OpBuilder replacer(op);
    replacer.create<upmem::ReturnOp>(op.getLoc());
    op.erase();
  });
  return outlinedFunc;
}

/// Replace `gpu.launch` operations with an `gpu.launch_func` operation
/// launching `kernelFunc`. The kernel func contains the body of the
/// `gpu.launch` with constant region arguments inlined.
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

// upmem::UPMEMFuncOp mlir::outlineKernelFunc(upmem::LaunchOp launchOp,
//                                        StringRef kernelFnName,
//                                        llvm::SmallVectorImpl<Value> &operands) {
//   DenseSet<Value> inputOperandSet;
//   inputOperandSet.insert(operands.begin(), operands.end());
//   SetVector<Value> operandSet(operands.begin(), operands.end());
//   auto funcOp = outlineKernelFuncImpl(launchOp, kernelFnName, operandSet);
//   for (auto operand : operandSet) {
//     if (!inputOperandSet.count(operand))
//       operands.push_back(operand);
//   }
//   return funcOp;
// }

// /// Replace `gpu.launch` operations with an `gpu.launch_func` operation
// /// launching `kernelFunc`. The kernel func contains the body of the
// /// `gpu.launch` with constant region arguments inlined.
// static void convertToLaunchFuncOp(gpu::LaunchOp launchOp,
//                                   gpu::GPUFuncOp kernelFunc,
//                                   ValueRange operands) {
//   OpBuilder builder(launchOp);
//   // The launch op has an optional dynamic shared memory size. If it doesn't
//   // exist, we use zero.
//   Value asyncToken = launchOp.getAsyncToken();
//   auto launchFunc = builder.create<gpu::LaunchFuncOp>(
//       launchOp.getLoc(), kernelFunc, launchOp.getGridSizeOperandValues(),
//       launchOp.getBlockSizeOperandValues(),
//       launchOp.getDynamicSharedMemorySize(), operands,
//       asyncToken ? asyncToken.getType() : nullptr,
//       launchOp.getAsyncDependencies());
//   launchOp.replaceAllUsesWith(launchFunc);
//   launchOp.erase();
// }




//===----------------------------------------------------------------------===//
struct UPMEMOutlineKernelPass : public impl::UPMEMOutlineKernelPassBase<UPMEMOutlineKernelPass> {
    using Base::Base;

    void runOnOperation() final;

    void getDependentDialects(DialectRegistry &registry) const override {
    }
    upmem::UPMEMModuleOp createKernelModule(upmem::UPMEMFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable);
};

void UPMEMOutlineKernelPass::runOnOperation() {
    SymbolTable symbolTable(getOperation());
    bool modified = false;
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      Block::iterator insertPt(func->getNextNode());
      auto funcWalkResult = func.walk([&](upmem::LaunchOp op) {
        SetVector<Value> operands;
        std::string kernelFnName =
            Twine(op->getParentOfType<func::FuncOp>().getName(), "_kernel")
                .str();

        upmem::UPMEMFuncOp outlinedFunc =
            outlineKernelFuncImpl(op, kernelFnName, operands);

        auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
        symbolTable.insert(kernelModule, insertPt);

    //     // Potentially changes signature, pulling in constants.
        convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
        modified = true;
        return WalkResult::advance();
      });
      if (funcWalkResult.wasInterrupted())
        return signalPassFailure();
    }

    // // If any new module was inserted in this module, annotate this module as
    // // a container module.
    // if (modified)
    //   getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
    //                           UnitAttr::get(&getContext()));

}

upmem::UPMEMModuleOp UPMEMOutlineKernelPass::createKernelModule(upmem::UPMEMFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable) {
    auto *context = getOperation().getContext();
    OpBuilder builder(context);
    auto kernelModule = builder.create<upmem::UPMEMModuleOp>(kernelFunc.getLoc(),
                                                         kernelFunc.getName());

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

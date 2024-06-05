#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

#include <llvm/Support/Regex.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>
#include "mlir/IR/SymbolTable.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_UPMEMOUTLINEKERNELPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
struct UPMEMOutlineKernelPass : public impl::UPMEMOutlineKernelPassBase<UPMEMOutlineKernelPass> {
    using Base::Base;

    void runOnOperation() final;

    void getDependentDialects(DialectRegistry &registry) const override {
    }
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

        // upmem::UPMEMFuncOp outlinedFunc =
        //     outlineKernelFuncImpl(op, kernelFnName, operands);

    //     // Create nested module and insert outlinedFunc. The module will
    //     // originally get the same name as the function, but may be renamed on
    //     // insertion into the parent module.
    //     auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
    //     symbolTable.insert(kernelModule, insertPt);

    //     // Potentially changes signature, pulling in constants.
    //     convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
    //     modified = true;
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


} // namespace mlir

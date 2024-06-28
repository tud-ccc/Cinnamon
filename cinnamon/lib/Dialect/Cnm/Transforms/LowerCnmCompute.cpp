#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/CnmComputeTransforms.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMLOWERCOMPUTEPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;

struct CnmLowerComputePass
    : public cnm::impl::CnmLowerComputePassBase<CnmLowerComputePass> {
  void runOnOperation() override {
    auto fun = getOperation();

    fun.walk([&](cnm::ComputeOp op) {
      OpBuilder rewriter(&getContext());
      cnm::lowerComputeToLaunch(rewriter, op);
    });
  }
};

#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include <cinm-mlir/Conversion/CinmPasses.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/CinmPasses.h.inc>


namespace {

Value convertInputIntoAlloc(Value inputBuf, Value workGroup, ImplicitLocOpBuilder& rewriter) {
  // For each input of the reduce, we need to

  auto inputTy = inputBuf.getType().cast<RankedTensorType>();
  cnm::BufferType bufTy = cnm::BufferType::get(rewriter.getContext(), inputTy.getShape(), inputTy.getElementType(), 0);

  // 1. Allocate a cinm buffer
  Value alloc = rewriter.create<cnm::AllocOp>(bufTy, workGroup);
  // 2. Scatter original buffer into it


  rewriter.create<cnm::ScatterOp>(bufTy, workGroup);

  return alloc;
}

void convertLinalgReduceIntoLaunch(linalg::ReduceOp reduction, cnm::WorkgroupType wgType) {

}


struct ConvertTiledCinmToCnm : public ConvertTiledCinmToCnmBase<ConvertTiledCinmToCnm> {

  void runOnOperation() override {
    
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm::createConvertTiledCinmToCnmPass() {
    return std::make_unique<ConvertTiledCinmToCnm>();
}

void mlir::cinm::registerCinmToCnmPipeline() {
  // todo
}
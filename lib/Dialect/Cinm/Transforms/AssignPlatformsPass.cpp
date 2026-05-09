#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMASSIGNPLATFORMSPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

struct CinmAssignPlatformsPass
    : public impl::CinmAssignPlatformsPassBase<CinmAssignPlatformsPass> {
  using Base::Base;

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    auto platformsAttr =
        func->getAttrOfType<ArrayAttr>(CinmDialect::AVAILABLE_PLATFORMS_NAME);
    if (!platformsAttr)
      return;

    SmallVector<CinmPlatformAttrInterface> platforms;
    for (Attribute attr : platformsAttr) {
      if (auto platform = llvm::dyn_cast<CinmPlatformAttrInterface>(attr))
        platforms.push_back(platform);
    }
    if (platforms.empty())
      return;

    IRRewriter rewriter(func->getContext());

    SmallVector<Operation *> opsToWrap;
    func.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("cinm.op."))
        return;
      if (op->getParentOfType<cinm::ComputeBlockOp>())
        return;
      opsToWrap.push_back(op);
    });

    for (Operation *op : opsToWrap) {
      SmallVector<Attribute> interested;
      for (auto platform : platforms) {
        if (platform.isOffloadingTarget(op))
          interested.push_back(platform);
      }
      if (interested.empty())
        continue;

      ComputeOp computeOp = wrapOperationInCompute(op, rewriter);
      computeOp->setAttr(CinmDialect::AVAILABLE_PLATFORMS_NAME,
                         ArrayAttr::get(func->getContext(), interested));
    }
  }
};

} // namespace mlir::cinm

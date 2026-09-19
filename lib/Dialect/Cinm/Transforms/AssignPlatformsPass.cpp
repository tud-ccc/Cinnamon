#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/Support/Format.h>
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
      if (!op->getName().getStringRef().starts_with("cinm.op.") &&
          op->getName().getDialectNamespace() != "linalg")
        return;
      if (op->getParentOfType<cinm::ComputeBlockOp>())
        return;
      opsToWrap.push_back(op);
    });

    cinm::HostModel host = HostPlatformAttr::getInScope(func).getModel();
    if (hostOpsPerSecond > 0)
      host.opsPerSecond = hostOpsPerSecond;
    if (hostDramBytesPerSecond > 0)
      host.dramBytesPerSecond = hostDramBytesPerSecond;

    for (Operation *op : opsToWrap) {
      // An op the program already put inside a cinm.compute was offloaded on
      // purpose. Capability still has to hold, but profitability is not
      // second-guessed: this is the override for the cases the roofline
      // rejects and the author wants anyway.
      const bool explicitlyRequested = op->getParentOfType<cinm::ComputeOp>();

      SmallVector<Attribute> interested;
      for (auto platform : platforms) {
        if (!platform.isOffloadingTarget(op))
          continue;
        if (requireProfitable && !explicitlyRequested) {
          cinm::OffloadVerdict verdict = platform.evaluateOffload(op, host);
          if (!verdict.profitable) {
            std::string terms;
            llvm::raw_string_ostream os(terms);
            os << llvm::format(
                "work %.3g ops, resident %.3g B, per-call %.3g B, host %.3g s "
                "vs device %.3g s",
                verdict.work, verdict.staticBytes, verdict.dynamicBytes,
                verdict.hostSeconds, verdict.deviceSeconds);
            // Anchored to the location, not the op: a diagnostic that
            // carries the op prints it, and printing verifies and numbers
            // the whole enclosing function first -- once per rejected op,
            // that is quadratic in the program.
            emitRemark(op->getLoc())
                << "not offloaded: " << verdict.reason << " (" << terms << ")";
            continue;
          }
        }
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

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdio>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h"
#include "cinm-mlir/Utils/Scheduling/Schedulers/Alap.h"
#include "cinm-mlir/Utils/Scheduling/Schedulers/Asap.h"
#include "cinm-mlir/Utils/Scheduling/Scheduling.h"

namespace mlir::cim {

  static bool isCimOp(Operation &op) {
    return op.getName().getStringRef().starts_with("cim.op");
  }

  static void scheduleCimOpOnCrossbar(Operation &op, Value crossbar) {
    op.getOperand(0).replaceUsesWithIf(
        crossbar, [&](OpOperand &use) { return use.getOwner() == &op; });
  }

  static Operation *insertBarrierForCimOpResult(PatternRewriter &rewriter, Operation &cimOp) {
    auto future = cimOp.getResult(0);

    auto shapedType = future.getType().cast<ShapedType>();
    auto barrierOp = rewriter.create<BarrierOp>(
        cimOp.getLoc(),
        RankedTensorType::get(shapedType.getShape(), shapedType.getElementType()),
        future);
    future.replaceAllUsesExcept(barrierOp.getResult(), barrierOp);
    return barrierOp;
  }

  static bool cimScheduleCheckDynamicallyLegal(Operation *op) {
    auto acquireDeviceOp = dyn_cast<AcquireDeviceOp>(op);

    if (!acquireDeviceOp)
      return true;

    return acquireDeviceOp.getIsFullyScheduled();
  };

  static ReleaseDeviceOp getReleaseDeviceOp(AcquireDeviceOp op) {
    for (auto *user : op.getResult().getUsers())
      if (auto releaseDeviceOp = dyn_cast<ReleaseDeviceOp>(user))
        return releaseDeviceOp;
    return nullptr;
  }

  static ReleaseCrossbarOp getReleaseCrossbarOp(AcquireCrossbarOp op) {
    for (auto *user : op.getResult().getUsers())
      if (auto releaseCrossbarOp = dyn_cast<ReleaseCrossbarOp>(user))
        return releaseCrossbarOp;
    return nullptr;
  }

  static std::vector<AcquireCrossbarOp>
  getDependentAcquireCrossbarOps(AcquireDeviceOp op) {
    std::vector<AcquireCrossbarOp> acquireCrossbarOps;
    for (auto *user : op.getResult().getUsers())
      if (auto acquireCrossbarOp = dyn_cast<AcquireCrossbarOp>(user))
        acquireCrossbarOps.push_back(acquireCrossbarOp);
    return acquireCrossbarOps;
  }

  static std::vector<Operation *>
  getDependentCimOps(AcquireCrossbarOp op) {
    std::vector<Operation *> ops;
    for (auto *user : op.getResult().getUsers())
      if (isCimOp(*user))
        ops.push_back(user);
    return ops;
  }

  static bool
  hasUsersOutsideAcquiredBlock(Operation *op, Value crossbarId) {
    for (auto *user : op->getUsers()) {
      if (isCimOp(*user) && user->getOperand(0) == crossbarId)
        continue;

      return true;
    }

    return false;
  }

  static std::pair<std::vector<Value>, std::vector<Value>>
  prepareForScheduling(AcquireDeviceOp acquireDeviceOp, PatternRewriter &rewriter) {
    auto releaseDeviceOp = getReleaseDeviceOp(acquireDeviceOp);
    auto acquireCrossbarOps = getDependentAcquireCrossbarOps(acquireDeviceOp);

    // save only one of the potentially multiple crossbar ids
    auto savedCrossbarOp = acquireCrossbarOps.back();
    acquireCrossbarOps.pop_back();

    // delete the rest, rebind their uses to saved crossbar id
    for (auto acquireCrossbarOp : acquireCrossbarOps) {
      auto releaseCrossbarOp = getReleaseCrossbarOp(acquireCrossbarOp);
      acquireCrossbarOp.getResult().replaceAllUsesWith(savedCrossbarOp.getResult());
      acquireCrossbarOp.erase();
      releaseCrossbarOp.erase();
    }

    std::unordered_set<Operation *> discoveredBarriers;

    auto cimOps = getDependentCimOps(savedCrossbarOp);
    for (auto *cimOp : cimOps) {
      for (auto operand : cimOp->getOperands()) {
        // check if operand is a tensor created by a cim.barrier operation
        auto *definingOp = operand.getDefiningOp();
        if (!llvm::isa<TensorType>(operand.getType()) || !definingOp || !llvm::isa<BarrierOp>(definingOp))
          continue;

        discoveredBarriers.insert(definingOp);
      }

      for (auto user : cimOp->getUsers()) {
        if (llvm::isa<BarrierOp>(user))
          discoveredBarriers.insert(user);
      }
    }

    std::vector<Value> crossbarIds;
    std::vector<Value> roots;

    // add saved crossbar id to crossbarIds
    crossbarIds.push_back(savedCrossbarOp.getResult());

    // recreate acquire_crossbar operations, until the number of available crossbars is reached
    while (crossbarIds.size() != acquireDeviceOp.getAvailableCrossbarCount()) {
      rewriter.setInsertionPointAfter(acquireDeviceOp.getOperation());
      auto acquireCrossbarOp =
          rewriter.create<AcquireCrossbarOp>(acquireDeviceOp->getLoc(), acquireDeviceOp.getResult());
      crossbarIds.push_back(acquireCrossbarOp.getResult());

      rewriter.setInsertionPoint(releaseDeviceOp);
      rewriter.create<ReleaseCrossbarOp>(releaseDeviceOp->getLoc(), acquireCrossbarOp.getResult());
    }

    // find all roots for scheduling
    for (auto *barrier : discoveredBarriers) {
      if (hasUsersOutsideAcquiredBlock(barrier, savedCrossbarOp.getResult()))
        roots.push_back(barrier->getOperand(0));

      // replace operand with its cim.future
      barrier->getResult(0).replaceAllUsesWith(barrier->getOperand(0));
      barrier->erase();
    }

    return {crossbarIds, roots};
  }

  template<template<typename> class Scheduler>
  struct CimSchedulePattern : public RewritePattern {
    CimSchedulePattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag{}, 1, context) {}

    LogicalResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const override {
      auto acquireDeviceOp = dyn_cast<AcquireDeviceOp>(op);

      if (!acquireDeviceOp)
        return failure();

      rewriter.startOpModification(op);

      auto [crossbars, roots] = prepareForScheduling(acquireDeviceOp, rewriter);
      Scheduler<Value> scheduler{crossbars};

      cinm::utils::scheduling::SchedulingHooks<Value> hooks{
          .rescheduleOperationFilter = isCimOp,
          .schedulingStrategy =
              std::bind(&Scheduler<Value>::schedule, &scheduler, std::placeholders::_1),
          .operationScheduler = scheduleCimOpOnCrossbar,
          .barrierInserter = insertBarrierForCimOpResult};

      cinm::utils::scheduling::applyScheduling(rewriter, roots, hooks);

      op->setAttr("isFullyScheduled", rewriter.getBoolAttr(true));

      rewriter.finalizeOpModification(op);

      return success();
    }
  };

  template<template<typename> class Scheduler, template<typename> class PassBase>
  struct CimSchedulePassBase
      : public PassBase<CimSchedulePassBase<Scheduler, PassBase>> {
    using PassBase<CimSchedulePassBase<Scheduler, PassBase>>::PassBase;

    void runOnOperation() final {
      auto &ctx = this->getContext();

      ConversionTarget target(ctx);
      target.markUnknownOpDynamicallyLegal(cimScheduleCheckDynamicallyLegal);

      RewritePatternSet patterns(&ctx);
      patterns.add<CimSchedulePattern<Scheduler>>(&ctx);

      auto result =
          applyPartialConversion(this->getOperation(), target, std::move(patterns));

      if (failed(result))
        this->signalPassFailure();
    }
  };

  //===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_CIMSCHEDULEASAPPASS
#define GEN_PASS_DEF_CIMSCHEDULEALAPPASS
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h.inc"

  //===----------------------------------------------------------------------===//

  struct CimScheduleAsapPass : CimSchedulePassBase<cinm::utils::scheduling::AsapScheduler, impl::CimScheduleAsapPassBase> {};
  struct CimScheduleAlapPass : CimSchedulePassBase<cinm::utils::scheduling::AlapScheduler, impl::CimScheduleAlapPassBase> {};

}  // namespace mlir::cim
#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <list>
#include <vector>
namespace mlir::cinm::utils::scheduling {

  struct DependencyGraphNode {
    Operation &operation;
    std::vector<DependencyGraphNode *> dependencies{};
    std::vector<DependencyGraphNode *> dependents{};
    bool scheduled{false};
  };

  using DependencyGraph = std::list<DependencyGraphNode>;
  using DependencyGraphView = DependencyGraph &;

  template<typename ComputeResource>
  struct ScheduledOperation {
    Operation *operation;
    ComputeResource resource;
    size_t dispatchTime, barrierTime;

    std::strong_ordering operator<=>(ScheduledOperation const &other) {
      return dispatchTime <=> other.dispatchTime;
    }

    friend void swap(ScheduledOperation &lhs, ScheduledOperation &rhs) {
      using std::swap;
      swap(lhs.operation, rhs.operation);
      swap(lhs.resource, rhs.resource);
      swap(lhs.dispatchTime, rhs.dispatchTime);
      swap(lhs.barrierTime, rhs.barrierTime);
    }
  };

  template<typename ComputeResource>
  using OperationScheduling = std::list<ScheduledOperation<ComputeResource>>;

  template<typename ComputeResource>
  struct SchedulingHooks {
    using RescheduleOperationFilter = std::function<bool(Operation &op)>;
    using SchedulingStrategy = std::function<OperationScheduling<ComputeResource>(
        DependencyGraphView graph)>;
    using OperationScheduler = std::function<void(
        Operation &operation, ComputeResource schedulingResource)>;
    using BarrierInserter =
        std::function<Operation *(PatternRewriter &rewriter, Operation &op)>;

    RescheduleOperationFilter rescheduleOperationFilter;
    SchedulingStrategy schedulingStrategy;
    OperationScheduler operationScheduler;
    BarrierInserter barrierInserter;
  };

  template<typename ComputeResource>
  void applyScheduling(PatternRewriter &rewriter, ValueRange rootValues,
                       SchedulingHooks<ComputeResource> const &hooks) {
    std::list<DependencyGraphNode *> openOperations{};
    DependencyGraph dependencyGraph{};

    for (auto const &value : rootValues) {
      DependencyGraphNode node{.operation = *value.getDefiningOp()};
      auto it = dependencyGraph.insert(dependencyGraph.end(), std::move(node));
      openOperations.push_back(&*it);
    }

    while (!openOperations.empty()) {
      auto &currentNode = *openOperations.front();
      openOperations.pop_front();

      for (auto value : currentNode.operation.getOperands()) {
        auto *definingOp = value.getDefiningOp();

        if (!definingOp || !hooks.rescheduleOperationFilter(*definingOp))
          continue;

        DependencyGraphNode node{.operation = *definingOp};
        node.dependents.push_back(&currentNode);
        auto it = dependencyGraph.insert(dependencyGraph.end(), std::move(node));

        openOperations.push_back(&*it);
        currentNode.dependencies.push_back(&*it);
      }
    }

    auto schedule = hooks.schedulingStrategy(dependencyGraph);

    size_t currentTime = 0;
    Operation *lastOperation = nullptr;

    static constexpr size_t noTime = std::numeric_limits<size_t>::max();

    while (!schedule.empty()) {
      // First pass: Only op dispatch
      for (auto &scheduledOp : schedule) {
        if (scheduledOp.dispatchTime != currentTime)
          continue;

        scheduledOp.dispatchTime = noTime;
        Operation *op = scheduledOp.operation;
        if (lastOperation)
          op->moveAfter(lastOperation);
        lastOperation = op;
        hooks.operationScheduler(*op, scheduledOp.resource);
      }

      // Second pass: Generate barriers
      for (auto it = schedule.begin(); it != schedule.end();) {
        auto scheduledOp{it++};

        if (scheduledOp->barrierTime != currentTime)
          continue;

        rewriter.setInsertionPointAfter(lastOperation);
        auto *barrierOp = hooks.barrierInserter(rewriter, *scheduledOp->operation);

        if (barrierOp)
          lastOperation = barrierOp;

        schedule.erase(scheduledOp);
      }

      currentTime++;
    }
  }

}  // namespace mlir::cinm::utils::scheduling
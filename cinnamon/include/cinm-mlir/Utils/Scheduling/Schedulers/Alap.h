#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#include "cinm-mlir/Utils/Scheduling/Scheduling.h"

namespace mlir::cinm::utils::scheduling {

  template<typename ComputeResource>
  class AlapScheduler {
   public:
    AlapScheduler(llvm::ArrayRef<ComputeResource> resources)
        : computeResources(std::move(resources)) {}

    OperationScheduling<ComputeResource> schedule(DependencyGraphView graph) {
      OperationScheduling<ComputeResource> schedule;

      bool doneScheduling = false;

      size_t time = std::numeric_limits<size_t>::max();
      for (; !doneScheduling; --time) {
        doneScheduling = true;

        size_t nextFreeResource = 0;
        std::vector<DependencyGraphNode *> scheduledThisTime{};

        for (auto &node : graph) {
          if (node.scheduled)
            continue;

          if (!llvm::all_of(node.dependents, isScheduled))
            continue;

          if (nextFreeResource < computeResources.size()) {
            schedule.push_back(ScheduledOperation<ComputeResource>{
                .operation = &node.operation,
                .resource = computeResources[nextFreeResource++],
                .dispatchTime = time,
                .barrierTime = time});

            scheduledThisTime.push_back(&node);
          }

          doneScheduling = false;
        }

        for (auto *node : scheduledThisTime)
          node->scheduled = true;
      }

      for (auto &scheduledOp : schedule) {
        scheduledOp.dispatchTime = scheduledOp.dispatchTime - time;
        scheduledOp.barrierTime = scheduledOp.barrierTime - time;
      }

      return schedule;
    }

   private:
    static bool isScheduled(DependencyGraphNode const *n) { return n->scheduled; }

    llvm::ArrayRef<ComputeResource> computeResources;
  };
}  // namespace mlir::cinm::utils::scheduling
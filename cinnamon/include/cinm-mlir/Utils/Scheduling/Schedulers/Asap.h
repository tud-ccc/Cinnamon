#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>

#include "cinm-mlir/Utils/Scheduling/Scheduling.h"

namespace mlir::cinm::utils::scheduling {

  template<typename ComputeResource>
  class AsapScheduler {
   public:
    AsapScheduler(llvm::ArrayRef<ComputeResource> resources)
        : computeResources(std::move(resources)) {}

    OperationScheduling<ComputeResource> schedule(DependencyGraphView graph) {
      OperationScheduling<ComputeResource> schedule;

      bool doneScheduling = false;

      for (size_t time = 0; !doneScheduling; ++time) {
        doneScheduling = true;

        size_t nextFreeResource = 0;
        std::vector<DependencyGraphNode *> scheduledThisTime{};

        for (auto &node : graph) {
          if (node.scheduled)
            continue;

          if (!llvm::all_of(node.dependencies, isScheduled))
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

      return schedule;
    }

   private:
    static bool isScheduled(DependencyGraphNode const *n) { return n->scheduled; }

    llvm::ArrayRef<ComputeResource> computeResources;
  };
}  // namespace mlir::cinm::utils::scheduling
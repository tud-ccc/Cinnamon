#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace mlir::upmem {

namespace {

struct OpCountSimulator : UpmemSimulator {
  mlir::cinm::utils::Maybe<double> simulate(Region& module) override {
    double cost = 0.0;
    module.walk([&](mlir::Operation *op) {
      llvm::StringRef name = op->getName().getStringRef();
      // Data-movement is the primary bottleneck on UPMEM.
      if (name.contains("transfer") || name.contains("copy") ||
          name.contains("scatter") || name.contains("gather"))
        cost += 100.0;
      else if (name.contains("launch") || name.contains("call"))
        cost += 10.0;
      else
        cost += 1.0;
    });
    return cost;
  }
};

} // namespace

std::unique_ptr<UpmemSimulator> createOpCountSimulator() {
  return std::make_unique<OpCountSimulator>();
}

} // namespace mlir::upmem

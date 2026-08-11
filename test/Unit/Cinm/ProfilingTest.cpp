//===- ProfilingTest.cpp - Stage A profile extraction ---------------------===//
//
// profileComputeBlock runs the single-op search once per shared-resource menu
// value with the resource pinned, and records L(D) with its argmin
// (docs/GraphOptimizationDesign.md, Stage A). The plugin here is a mock whose
// cost function is known in closed form, so the test can check that the
// profile is the pointwise optimum: the harness (pinning, per-point search,
// harvesting, infeasible-point holes) is what is under test, not any real
// lowering.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include <gtest/gtest.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;
using cinm::utils::Maybe;

namespace {

/// A target whose cost is 1000/dpus + tile: monotone in the resource, so the
/// search must pick tile = 1 everywhere and the profile must be exactly
/// 1000/D + 1 at every feasible menu value.
struct MockPlugin : cinm::InferencePlugin {
  void initializeSpace(cinm::ComputeBlockOp,
                       cinm::SpaceBuilder &space) override {
    dpus_ = space.intRange("dpus", 1, 64);
    tile_ = space.intRange("tile", 1, 4);
  }

  Maybe<cinm::utils::SimCost> evaluate(cinm::TrialInfo &trial) override {
    int64_t dpus = 0, tile = 0;
    for (size_t d = 0; d < trial.space->numDims(); ++d) {
      if (trial.space->dimName(d) == "dpus")
        dpus = trial.config[d];
      else if (trial.space->dimName(d) == "tile")
        tile = trial.config[d];
    }
    return cinm::utils::SimCost::forKernel(1000.0 / double(dpus) +
                                           double(tile));
  }

  std::unique_ptr<cinm::InferencePlugin> clone() const override {
    return std::make_unique<MockPlugin>(*this);
  }

  StringRef sharedResourceParam() const override { return "dpus"; }
  SmallVector<int64_t> sharedResourceMenu() const override {
    // 128 exceeds the declared range: the pinned space is empty there and
    // the profile must simply have no point, not an error.
    return {16, 32, 64, 128};
  }

  cinm::IntVar dpus_, tile_;
};

TEST(Profiling, ProfilesTheMenuPointwise) {
  MLIRContext ctx;
  ctx.loadDialect<cinm::CinmDialect, func::FuncDialect, arith::ArithDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%x: tensor<8xi32>) -> tensor<8xi32> {
      %r = cinm.compute_block (%a = %x : tensor<8xi32>) -> tensor<8xi32> {
        %y = arith.addi %a, %a : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      return %r : tensor<8xi32>
    }
  )mlir",
                                                             &ctx);
  ASSERT_TRUE(module);
  cinm::ComputeBlockOp block;
  module->walk([&](cinm::ComputeBlockOp op) { block = op; });
  ASSERT_TRUE(block);

  MockPlugin plugin;
  cinm::InferenceOptions opts;
  opts.exhaustiveSearch = true; // deterministic: the argmin, not a BO guess
  opts.numWorkers = 1;
  opts.dumpFullPool = false;

  ScopedDiagnosticHandler diagHandler(&ctx, [](Diagnostic &diag) {
    llvm::errs() << "diagnostic: " << diag.str() << "\n";
    return failure();
  });

  auto result = cinm::profileComputeBlock(block, plugin, opts);
  auto *points = std::get_if<SmallVector<cinm::ProfilePoint>>(&result);
  if (auto *fail = std::get_if<DiagnosedSilenceableFailure>(&result))
    FAIL() << "profiling failed (" << fail->getStatusString()
           << "): " << (fail->isSilenceableFailure() ? fail->getMessage() : "");
  ASSERT_TRUE(points) << "profiling failed";

  ASSERT_EQ(points->size(), 3u) << "the out-of-range menu value must be a "
                                   "hole, the rest must have points";
  for (auto [i, expectD] : llvm::enumerate(SmallVector<int64_t>{16, 32, 64})) {
    const cinm::ProfilePoint &p = (*points)[i];
    EXPECT_EQ(p.resource, expectD);
    EXPECT_DOUBLE_EQ(p.costMs, 1000.0 / double(expectD) + 1.0);
    ASSERT_TRUE(p.config.contains("dpus"));
    ASSERT_TRUE(p.config.contains("tile"));
    EXPECT_EQ(p.config.lookup("dpus"), expectD) << "the pin must hold";
    EXPECT_EQ(p.config.lookup("tile"), 1) << "the argmin tile is 1";
  }
}

} // namespace

//===- ProfilingTest.cpp - Cost profile extraction ------------------------===//
//
// profileComputeBlock runs the single-op search once per shared-resource menu
// value with the resource pinned, and records the best cost with its argmin.
// The plugin here is a mock whose cost function is known in closed form, so
// the test can check that the profile is the pointwise optimum: the harness
// (pinning, per-point search, harvesting, infeasible-point holes) is what is
// under test, not any real lowering.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"

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
  int64_t sharedResourceMax() const override { return 128; }
  SmallVector<int64_t> sharedResourceMenu(cinm::ComputeBlockOp) const override {
    // 128 exceeds the declared range: the pinned space is empty there and
    // the profile must simply have no point, not an error.
    return {16, 32, 64, 128};
  }

  /// Counts the block's static operands (staticness must be readable from
  /// inside a trial clone, where the original defs are out of reach) and
  /// reports footprints derived from them, so the test can check the whole
  /// path from `cinm.static` on the original function to the profile point.
  cinm::ResidencyInfo measureResidency(cinm::TrialInfo &trial) override {
    cinm::LevelResidency mem{"mem", 0, 0};
    for (BlockArgument arg : trial.computeBlock.getBodyArguments())
      if (cinm::isStaticValue(arg))
        mem.staticBytes += 100;
      else
        mem.dynBytes += 10;
    cinm::ResidencyInfo out;
    out.weightScatterMs = double(mem.staticBytes) / 100.0;
    out.levels.push_back(std::move(mem));
    return out;
  }

  cinm::IntVar dpus_, tile_;
};

TEST(Profiling, ProfilesTheMenuPointwise) {
  MLIRContext ctx;
  ctx.loadDialect<cinm::CinmDialect, func::FuncDialect, arith::ArithDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%w: tensor<8xi32> {cinm.static}, %x: tensor<8xi32>) -> tensor<8xi32> {
      %r = cinm.compute_block (%a = %w : tensor<8xi32>, %b = %x : tensor<8xi32>) -> tensor<8xi32> {
        %y = arith.addi %a, %b : tensor<8xi32>
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
    // One static operand (%w, via the forwarded cinm.static arg attr) and
    // one dynamic (%x), as the mock's residency model counts them.
    const cinm::LevelResidency *mem = p.residency.find("mem");
    ASSERT_TRUE(mem);
    EXPECT_EQ(mem->staticBytes, 100);
    EXPECT_EQ(mem->dynBytes, 10);
    EXPECT_DOUBLE_EQ(p.residency.weightScatterMs, 1.0);
  }
}

} // namespace

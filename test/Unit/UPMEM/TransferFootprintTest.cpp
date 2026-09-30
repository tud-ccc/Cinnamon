//===- TransferFootprintTest.cpp - Distinct host bytes of a transfer -----===//
//
// uniqueHostElements is what prices a replicated scatter, so the cases are the
// ones the compiler emits: every DPU its own block, the two replicated
// layouts of the gemv outliers (DPU d reads slice d mod 2, and slice
// d floordiv 8), and windows that overlap without coinciding.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTransferFootprint.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

using namespace mlir;

namespace {

class TransferFootprintTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  AffineExpr d = getAffineDimExpr(0, &ctx);
  AffineExpr c(int64_t v) { return getAffineConstantExpr(v, &ctx); }
  AffineMap map(ArrayRef<AffineExpr> results) {
    return AffineMap::get(1, 0, results, &ctx);
  }
};

TEST_F(TransferFootprintTest, DistinctBlocksCoverEveryDpu) {
  // memref<8x1024xi32>, DPU d reads row d.
  auto m = map({d, c(0)});
  EXPECT_EQ(upmem::uniqueHostElements(m, 8, {1024, 1}, 1024), 8 * 1024);
}

TEST_F(TransferFootprintTest, TwoSlicesSharedAlternately) {
  // gemv_512MB s290: memref<32x8x1x64xi32>, DPU d reads 8192 elements from
  // row (d mod 2) * 16 -- two distinct halves of the vector for 2048 DPUs.
  auto m = map({(d % 2) * 16, c(0), c(0), c(0)});
  EXPECT_EQ(upmem::uniqueHostElements(m, 2048, {512, 64, 64, 1}, 8192), 16384);
}

TEST_F(TransferFootprintTest, SlicesSharedByConsecutiveDpus) {
  // gemv_512MB s294: memref<256x4x1x16xi32>, DPU d reads 64 elements from
  // row d floordiv 8 -- 256 slices, each shared by 8 consecutive DPUs.
  auto m = map({d.floorDiv(8), c(0), c(0), c(0)});
  EXPECT_EQ(upmem::uniqueHostElements(m, 2048, {64, 16, 16, 1}, 64), 256 * 64);
}

TEST_F(TransferFootprintTest, OverlappingWindowsCountOnce) {
  // Starts 0, 2, 4, 6 with 4 elements each: [0, 10) in all.
  EXPECT_EQ(upmem::uniqueHostElements(map({d * 2}), 4, {1}, 4), 10);
}

TEST_F(TransferFootprintTest, TouchingWindowsMerge) {
  // Starts 0, 4, 8 with 4 elements each, then a gap to 20: [0, 12) + [20, 24).
  auto m = map({d * 4 + (d.floorDiv(3)) * 8});
  EXPECT_EQ(upmem::uniqueHostElements(m, 4, {1}, 4), 16);
}

TEST_F(TransferFootprintTest, SymbolsAreNotEvaluated) {
  auto m = AffineMap::get(1, 1, {d + getAffineSymbolExpr(0, &ctx)}, &ctx);
  EXPECT_EQ(upmem::uniqueHostElements(m, 4, {1}, 4), std::nullopt);
}

} // namespace

//===- GraphInferenceTest.cpp - Graph collection & program identity -------===//
//
// collectComputeGraphs produces the unit the graph-level optimization
// operates on: connected components of the block-level dataflow,
// canonicalized into program-identity classes (only blocks with identical
// programs may share a device set). These tests pin the two relations: what
// connects blocks into one graph, and what makes two blocks the same
// program.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphInference.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include <gtest/gtest.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;

namespace {

/// Parse `ir` and return the (graph -> class sizes) structure of its host
/// platform compute graphs, e.g. {{2}, {1, 1}} = two graphs, the first with
/// one class of two members, the second with two singleton classes.
static std::vector<std::vector<unsigned>>
graphShape(MLIRContext &ctx, OwningOpRef<ModuleOp> &module,
           llvm::StringRef ir) {
  ctx.loadDialect<cinm::CinmDialect, func::FuncDialect, arith::ArithDialect,
                  tensor::TensorDialect>();
  module = parseSourceString<ModuleOp>(ir, &ctx);
  EXPECT_TRUE(module) << "test IR does not parse";
  if (!module)
    return {};

  std::vector<std::vector<unsigned>> shape;
  for (const cinm::ComputeGraph &graph :
       cinm::collectComputeGraphs(*module, "host")) {
    std::vector<unsigned> classes;
    for (const cinm::BlockClass &cls : graph.classes)
      classes.push_back(cls.size());
    shape.push_back(std::move(classes));
  }
  return shape;
}

TEST(ComputeGraphs, IdenticalBlocksFormOneClass) {
  // Same shapes, same staticness, different weight values: one program.
  MLIRContext ctx;
  OwningOpRef<ModuleOp> m;
  auto shape = graphShape(ctx, m, R"mlir(
    func.func @f(%x: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
        attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %r1 = cinm.compute_block (%a = %x : tensor<8xi32>) -> tensor<8xi32> {
        %w = arith.constant dense<1> : tensor<8xi32>
        %y = arith.addi %a, %w : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      %r2 = cinm.compute_block (%a = %x : tensor<8xi32>) -> tensor<8xi32> {
        %w = arith.constant dense<2> : tensor<8xi32>
        %y = arith.addi %a, %w : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      return %r1, %r2 : tensor<8xi32>, tensor<8xi32>
    }
  )mlir");
  ASSERT_EQ(shape, (std::vector<std::vector<unsigned>>{{2}}));
}

TEST(ComputeGraphs, DifferentShapesAreDifferentClasses) {
  // Chained blocks of different shapes: one graph (dataflow-connected),
  // two classes (different programs).
  MLIRContext ctx;
  OwningOpRef<ModuleOp> m;
  auto shape = graphShape(ctx, m, R"mlir(
    func.func @f(%x: tensor<8xi32>) -> tensor<4xi32>
        attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %r1 = cinm.compute_block (%a = %x : tensor<8xi32>) -> tensor<8xi32> {
        %y = arith.addi %a, %a : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      %s = tensor.extract_slice %r1[0] [4] [1] : tensor<8xi32> to tensor<4xi32>
      %r2 = cinm.compute_block (%a = %s : tensor<4xi32>) -> tensor<4xi32> {
        %y = arith.addi %a, %a : tensor<4xi32>
        cinm.yield %y : tensor<4xi32>
      }
      return %r2 : tensor<4xi32>
    }
  )mlir");
  ASSERT_EQ(shape, (std::vector<std::vector<unsigned>>{{1, 1}}));
}

TEST(ComputeGraphs, StaticnessIsPartOfTheSignature) {
  // Identical bodies and types, but one operand is pinned weights and the
  // other is per-inference data: same binary, different cost profile, so the
  // blocks must not share a class.
  MLIRContext ctx;
  OwningOpRef<ModuleOp> m;
  auto shape = graphShape(ctx, m, R"mlir(
    func.func @f(%w: tensor<8xi32> {cinm.static}, %x: tensor<8xi32>)
        -> (tensor<8xi32>, tensor<8xi32>)
        attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %r1 = cinm.compute_block (%a = %w : tensor<8xi32>) -> tensor<8xi32> {
        %y = arith.addi %a, %a : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      %r2 = cinm.compute_block (%a = %x : tensor<8xi32>) -> tensor<8xi32> {
        %y = arith.addi %a, %a : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      return %r1, %r2 : tensor<8xi32>, tensor<8xi32>
    }
  )mlir");
  ASSERT_EQ(shape, (std::vector<std::vector<unsigned>>{{1, 1}}));
}

TEST(ComputeGraphs, DisconnectedComponentsAreSeparateGraphs) {
  // Two functions sharing nothing: separate grids to allocate, separate
  // graphs — even though the blocks are identical programs.
  MLIRContext ctx;
  OwningOpRef<ModuleOp> m;
  auto shape = graphShape(ctx, m, R"mlir(
    func.func @f(%x: tensor<8xi32>) -> tensor<8xi32>
        attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %r = cinm.compute_block (%a = %x : tensor<8xi32>) -> tensor<8xi32> {
        %y = arith.addi %a, %a : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      return %r : tensor<8xi32>
    }
    func.func @g(%x: tensor<8xi32>) -> tensor<8xi32>
        attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %r = cinm.compute_block (%a = %x : tensor<8xi32>) -> tensor<8xi32> {
        %y = arith.addi %a, %a : tensor<8xi32>
        cinm.yield %y : tensor<8xi32>
      }
      return %r : tensor<8xi32>
    }
  )mlir");
  ASSERT_EQ(shape, (std::vector<std::vector<unsigned>>{{1}, {1}}));
}

} // namespace

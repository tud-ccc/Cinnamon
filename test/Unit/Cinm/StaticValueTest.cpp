//===- StaticValueTest.cpp - Staticness derivation ------------------------===//
//
// isStaticValue decides which operands the graph-level optimization may treat
// as resident (docs/GraphOptimizationDesign.md, "What counts as static"). The
// cases below are the definition, one per rule, plus the sound-rejection
// cases: dynamic indexing and ops that combine two tensors.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"

#include <gtest/gtest.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;

namespace {

/// Parse `ir`, then for every `"test.check"(%v) {expect}` op assert that
/// isStaticValue(%v) == expect, tagging failures with the op's `case` attr.
static void checkStaticness(llvm::StringRef ir) {
  MLIRContext ctx;
  ctx.loadDialect<cinm::CinmDialect, func::FuncDialect, arith::ArithDialect,
                  tensor::TensorDialect>();
  ctx.allowUnregisteredDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  ASSERT_TRUE(module) << "test IR does not parse";

  unsigned nChecks = 0;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "test.check")
      return;
    ++nChecks;
    bool expect = op->hasAttr("expect");
    auto tag = op->getAttrOfType<StringAttr>("case");
    EXPECT_EQ(cinm::isStaticValue(op->getOperand(0)), expect)
        << (tag ? tag.getValue().str() : "<untagged>");
  });
  EXPECT_GT(nChecks, 0u) << "test IR contains no test.check ops";
}

TEST(StaticValue, FunctionArguments) {
  checkStaticness(R"mlir(
    func.func @f(%w: tensor<8x8xi32> {cinm.static}, %x: tensor<8x8xi32>) {
      "test.check"(%w) {expect, case = "annotated arg is static"} : (tensor<8x8xi32>) -> ()
      "test.check"(%x) {case = "plain arg is dynamic"} : (tensor<8x8xi32>) -> ()
      return
    }
  )mlir");
}

TEST(StaticValue, ConstantsAndViews) {
  checkStaticness(R"mlir(
    func.func @f(%w: tensor<8x8xi32> {cinm.static}, %i: index) {
      %cst = arith.constant dense<0> : tensor<8x8xi32>
      "test.check"(%cst) {expect, case = "constant is static"} : (tensor<8x8xi32>) -> ()

      %s0 = tensor.extract_slice %w[0, 0] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      "test.check"(%s0) {expect, case = "static slice of static arg"} : (tensor<4x4xi32>) -> ()

      %s1 = tensor.extract_slice %s0[0, 0] [2, 2] [1, 1] : tensor<4x4xi32> to tensor<2x2xi32>
      "test.check"(%s1) {expect, case = "chained static slice"} : (tensor<2x2xi32>) -> ()

      %s2 = tensor.extract_slice %w[%i, 0] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
      "test.check"(%s2) {case = "dynamically offset slice is dynamic"} : (tensor<4x4xi32>) -> ()

      %ins = tensor.insert_slice %s0 into %cst[0, 0] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
      "test.check"(%ins) {case = "insert_slice is not a pure view"} : (tensor<8x8xi32>) -> ()
      return
    }
  )mlir");
}

TEST(StaticValue, ComputeBlockDelegation) {
  checkStaticness(R"mlir(
    func.func @f(%w: tensor<8xi32> {cinm.static}, %x: tensor<8xi32>) {
      %r = cinm.compute_block (%a = %w : tensor<8xi32>, %b = %x : tensor<8xi32>) -> tensor<8xi32> {
        "test.check"(%a) {expect, case = "block arg of static operand"} : (tensor<8xi32>) -> ()
        "test.check"(%b) {case = "block arg of dynamic operand"} : (tensor<8xi32>) -> ()
        cinm.yield %a : tensor<8xi32>
      }
      return
    }
  )mlir");
}

} // namespace

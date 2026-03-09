#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"
#include "dlib/global_optimization/find_max_global.h"
#include "dlib/global_optimization/global_function_search.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <dlib/global_optimization.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

namespace py = pybind11;
using namespace py::literals;

namespace mlir::cinm {

#define GEN_PASS_DEF_RUNCOSTMODELPASS
#define GEN_PASS_DEF_COSTMODELFINALIZEPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {
struct RunCostModelPass : public impl::RunCostModelPassBase<RunCostModelPass> {
  using Base::Base;

  std::string convertTypeToCostModelType(mlir::Type type) {
    if (type.isBF16())
      return "bf16";
    else if (type.isF16())
      return "f16";
    else if (type.isF32())
      return "f32";
    else if (type.isF64())
      return "f64";
    else if (type.isF80())
      return "f80";
    else if (type.isF128())
      return "f128";
    else if (type.isIndex())
      return "index";
    else if (type.isInteger())
      return "i" + std::to_string(type.getIntOrFloatBitWidth());
    else
      return "<invalid_element_type>";
  }

  FailureOr<float> runCostModelOnOperationWithParameters(
      Operation *op, IRRewriter &rewriter, py::module &cost_model,
      const std::string &cost_model_name,
      const std::vector<std::string> &parameter_names,
      const dlib::matrix<double, 0, 1> &x) {
    std::unordered_map<std::string, double> current_dse_parameters;
    for (size_t i = 0; i < parameter_names.size(); i++) {
      current_dse_parameters[parameter_names[i]] = x(i);
    }

    std::string cost_model_pipeline = "";
    try {
      cost_model_pipeline =
          cost_model
              .attr("get_passes_for_next_run")(py::cast(current_dse_parameters))
              .cast<std::string>();
    } catch (const py::error_already_set &e) {
      // python throws a StopIteration exception when the generator is
      // exhausted, but we don't handle that explicitly and continue with
      // the next operation no matter the exception type
      return failure();
    }

    Region *dst_region = nullptr;
    if (cinm::SelectOp old_select =
            llvm::dyn_cast_or_null<cinm::SelectOp>(op->getParentOp())) {
      rewriter.setInsertionPointAfter(old_select);
      cinm::SelectOp new_select = rewriter.create<cinm::SelectOp>(
          op->getLoc(), old_select.getResultTypes(),
          old_select->getNumRegions() + 1);

      for (size_t i = 0; i < old_select->getNumRegions(); i++) {
        rewriter.inlineRegionBefore(old_select->getRegion(i),
                                    new_select->getRegion(i),
                                    new_select->getRegion(i).begin());
      }

      rewriter.replaceOp(old_select, new_select);
      dst_region = &new_select.getRegions().back();
    } else {
      rewriter.setInsertionPointAfter(op);
      cinm::SelectOp new_select = rewriter.create<cinm::SelectOp>(
          op->getLoc(), op->getResultTypes(), 2);
      rewriter.replaceAllOpUsesWith(op, new_select);
      rewriter.setInsertionPointToStart(
          &new_select.getRegion(0).emplaceBlock());
      cinm::YieldOp yield =
          rewriter.create<cinm::YieldOp>(op->getLoc(), op->getResults());
      rewriter.moveOpBefore(op, yield);
      dst_region = &new_select.getRegion(1);
    }

    Block &block = dst_region->emplaceBlock();
    rewriter.setInsertionPointToStart(&block);
    IRMapping map;
    Operation *copy = rewriter.clone(*op, map);
    cinm::YieldOp yield =
        rewriter.create<cinm::YieldOp>(op->getLoc(), copy->getResults());

    if (cost_model_pipeline != "") {
      if (!copy->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
        emitError(getOperation()->getLoc(),
                  "pass pipeline can only be run on operations with the "
                  "IsIsolatedFromAbove trait");
        return failure();
      }

      PassManager pm(&getContext());

      if (llvm::failed(parsePassPipeline(StringRef(cost_model_pipeline),
                                         *(OpPassManager *)&pm))) {
        emitError(getOperation()->getLoc(), "invalid pass pipeline");
        return failure();
      }

      if (llvm::failed(runPipeline(pm, copy))) {
        return failure();
      }
    }

    std::string ir;
    llvm::raw_string_ostream os(ir);
    copy->print(os, OpPrintingFlags().printGenericOpForm());

    std::string locStr;
    llvm::raw_string_ostream locOs(locStr);
    op->getLoc().print(locOs);

    float cost = 0.0f;
    try {
      cost = cost_model
                 .attr("run")(py::str(ir.c_str(), ir.size()),
                              py::str(locStr.c_str(), locStr.size()))
                 .cast<float>();
    } catch (py::error_already_set &e) {
      return failure();
    }

    yield->setAttr("cinm_cost_model_data",
                   CostModelDataAttr::get(
                       &getContext(),
                       StringAttr::get(&getContext(), cost_model_name),
                       FloatAttr::get(Float32Type::get(&getContext()), cost)));
    return cost;
  }

  void runCostModelOnOperation(Operation *op, py::module &cost_model,
                               const std::string &cost_model_name,
                               const py::dict &dse_parameter_bounds,
                               size_t max_function_calls) {
    IRRewriter rewriter(&getContext());

    std::vector<std::string> parameter_names;
    dlib::matrix<double, 0, 1> parameter_bounds_min;
    dlib::matrix<double, 0, 1> parameter_bounds_max;
    std::vector<bool> parameter_is_integer_variable;

    std::unordered_map<std::string, std::tuple<double, double, bool>>
        dse_parameter_bounds2 = dse_parameter_bounds.cast<std::unordered_map<
            std::string, std::tuple<double, double, bool>>>();
    parameter_bounds_min.set_size(dse_parameter_bounds2.size());
    parameter_bounds_max.set_size(dse_parameter_bounds2.size());
    size_t i = 0;
    for (const auto &[key, value] : dse_parameter_bounds2) {
      const auto [min, max, is_integer] = value;
      parameter_names.push_back(key);
      parameter_bounds_min(i) = min;
      parameter_bounds_max(i) = max;
      parameter_is_integer_variable.push_back(is_integer);
      i++;
    }

    dlib::find_min_global(
        [&](const dlib::matrix<double, 0, 1> &x) -> double {
          const FailureOr<float> result = runCostModelOnOperationWithParameters(
              op, rewriter, cost_model, cost_model_name, parameter_names, x);
          return result.value_or(INFINITY);
        },
        parameter_bounds_min, parameter_bounds_max,
        parameter_is_integer_variable,
        dlib::max_function_calls(max_function_calls));
  }

  void runOnOperation() final {
    py::scoped_interpreter guard{};
    py::module sys = py::module::import("sys");

    std::string costModelModuleDir = ".";
    std::string costModelModuleName = "cost_model_test";
    if (!costModelPath.empty()) {
      std::filesystem::path p = costModelPath.getValue();
      costModelModuleName = p.stem().string();
      costModelModuleDir = std::filesystem::absolute(p).parent_path().string();
      sys.attr("path").attr("insert")(0, costModelModuleDir);
    }

    py::module cost_model;
    try {
      cost_model = py::module::import(costModelModuleName.c_str());
    } catch (py::error_already_set &e) {
      emitError(getOperation()->getLoc(), "failed to load cost-model (" +
                                              costModelModuleDir + " / " +
                                              costModelModuleName + ")");
      emitError(getOperation()->getLoc(), e.what());
      return;
    }

    std::string cost_model_name = cost_model.attr("name").cast<std::string>();
    auto operation_names = cost_model.attr("operations");
    auto dse_parameters = cost_model.attr("dse_parameters");
    size_t max_function_calls =
        cost_model.attr("dse_max_iterations").cast<size_t>();

    std::vector<Operation *> operations;
    getOperation()->walk([&](Operation *op) {
      if (!operation_names.contains(op->getName().getStringRef().str())) {
        return;
      }

      if (llvm::dyn_cast_or_null<cinm::SelectOp>(op->getParentOp())) {
        cinm::YieldOp yield = llvm::dyn_cast<cinm::YieldOp>(op->getNextNode());
        if (yield->hasAttr("cinm_cost_model_data")) {
          return;
        } else {
          operations.push_back(op);
        }
      } else {
        operations.push_back(op);
      }
    });

    for (Operation *op : operations) {
      runCostModelOnOperation(op, cost_model, cost_model_name, dse_parameters,
                              max_function_calls);
    }
  }
};

struct ApplySelectOpPattern : OpConversionPattern<cinm::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::SelectOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Region *min_cost_region = nullptr;
    float min_cost = INFINITY;

    for (Region &region : op.getRegions()) {
      cinm::YieldOp yield =
          llvm::dyn_cast<cinm::YieldOp>(region.front().back());
      if (cinm::CostModelDataAttr data =
              yield->getAttrOfType<cinm::CostModelDataAttr>(
                  "cinm_cost_model_data")) {
        const float cost = data.getCost().getValue().convertToFloat();
        if (cost < min_cost) {
          min_cost_region = &region;
          min_cost = cost;
        }
      }
    }

    if (!min_cost_region) {
      return failure();
    }

    cinm::YieldOp yield =
        llvm::dyn_cast<cinm::YieldOp>(min_cost_region->front().back());
    SmallVector<Value> results = yield.getOperands();
    if (min_cost_region->hasOneBlock()) {
      rewriter.inlineBlockBefore(&min_cost_region->front(), op);
    } else {
      emitError(op->getLoc(),
                "TODO: implement inlining regions with multiple blocks");
      return failure();
    }

    rewriter.replaceOp(op, results);
    rewriter.eraseOp(yield);

    return success();
  }
};

struct CostModelFinalizePass
    : public impl::CostModelFinalizePassBase<CostModelFinalizePass> {
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.insert<ApplySelectOpPattern>(&getContext());
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addIllegalOp<cinm::SelectOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cinm

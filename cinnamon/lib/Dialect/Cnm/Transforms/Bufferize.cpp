//===- Bufferize.cpp - Bufferization for `cnm` dialect ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of `cnm` dialect ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/ValueRange.h>

namespace mlir {
namespace cnm {
#define GEN_PASS_DEF_CNMBUFFERIZEPASS
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc"
} // namespace cnm
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {

struct ScatterOpInterface
    : public BufferizableOpInterface::ExternalModel<ScatterOpInterface,
                                                    cnm::ScatterOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto scatter = cast<cnm::ScatterOp>(op);
    FailureOr<Value> v = getBuffer(rewriter, scatter.getInput(), options);
    if (failed(v))
      return failure();

    replaceOpWithNewBufferizedOp<cnm::ScatterOp>(
        rewriter, op, *v, scatter.getBuffer(), scatter.getWg(),
        scatter.getScatterMap());
    return success();
  }
};

struct GatherOpInterface
    : public BufferizableOpInterface::ExternalModel<GatherOpInterface,
                                                    cnm::GatherOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &,
                                      const AnalysisState &) const {
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto gather = cast<cnm::GatherOp>(op);
    FailureOr<Value> v = getBuffer(rewriter, gather.getOutputBuf(), options);
    if (failed(v))
      return failure();

    rewriter.create<cnm::GatherOp>(op->getLoc(), gather.getBuffer(),
                                   gather.getWg(), gather.getGatherMap(), *v);
    replaceOpWithBufferizedValues(rewriter, op, ValueRange{*v});
    return success();
  }
};

struct CnmBufferizePass
    : public cnm::impl::CnmBufferizePassBase<CnmBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<cnm::CnmDialect>();

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    cnm::CnmDialect, scf::SCFDialect, arith::ArithDialect>();
  }
};
} // namespace

void cnm::registerCnmBufferizationExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, cnm::CnmDialect *) {
    cnm::ScatterOp::attachInterface<ScatterOpInterface>(*ctx);
    cnm::GatherOp::attachInterface<GatherOpInterface>(*ctx);
  });
}
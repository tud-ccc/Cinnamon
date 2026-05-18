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
#include "cinm-mlir/Utils/CinmUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
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
                          const BufferizationOptions &options,
                          const BufferizationState &state) const {
    auto scatter = cast<cnm::ScatterOp>(op);
    FailureOr<Value> v =
        getBuffer(rewriter, scatter.getInput(), options, state);
    if (failed(v))
      return failure();

    Value input = *v;
    auto bufShape = scatter.getBuffer().getType().getShape();
    // cnm.scatter requires the input to be contiguous in the suffix of
    // dimensions that correspond to the per-DPU buffer shape. When the
    // input is a non-contiguous subview (e.g. produced by expand_shape on
    // a strided subview), insert an alloc+copy to make it contiguous first.
    if (auto mr = dyn_cast<MemRefType>(input.getType())) {
      if (!mlir::scatteredMemrefIsContiguous(
              cast<TypedValue<ShapedType>>(input), bufShape)) {
        auto contiguousTy = MemRefType::get(mr.getShape(), mr.getElementType());
        Value alloc =
            memref::AllocOp::create(rewriter, op->getLoc(), contiguousTy);
        memref::CopyOp::create(rewriter, op->getLoc(), input, alloc);
        input = alloc;
      }
    }

    replaceOpWithNewBufferizedOp<cnm::ScatterOp>(
        rewriter, op, input, scatter.getBuffer(), scatter.getWg(),
        scatter.getScatterMap());
    return success();
  }
};

struct GatherOpInterface
    : public DstBufferizableOpInterfaceExternalModel<GatherOpInterface,
                                                    cnm::GatherOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    // Outputs of gather is not read, just written to.
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          const BufferizationState &state) const {
    auto gather = cast<cnm::GatherOp>(op);
    FailureOr<Value> v =
        getBuffer(rewriter, gather.getOutputBuf(), options, state);
    if (failed(v))
      return failure();

    cnm::GatherOp::create(rewriter, op->getLoc(), gather.getBuffer(),
                                   gather.getWg(), gather.getGatherMap(), *v);
    replaceOpWithBufferizedValues(rewriter, op, ValueRange{*v});
    return success();
  }
};

} // namespace

void cnm::registerCnmBufferizationExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, cnm::CnmDialect *) {
    cnm::ScatterOp::attachInterface<ScatterOpInterface>(*ctx);
    cnm::GatherOp::attachInterface<GatherOpInterface>(*ctx);
  });
}
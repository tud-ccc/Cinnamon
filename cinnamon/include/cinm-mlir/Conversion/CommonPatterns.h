#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

void populateFinalBufferizationPatterns(RewritePatternSet &set);

Value createOrFoldUnrealizedConversionCast(Location loc, OpBuilder &builder,
                                           Type dstType, Value value);

using BodyBuilderCallback = function_ref<SmallVector<Value>(
    OpBuilder &, Location, ValueRange, ValueRange)>;

SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<int64_t> loopSizes,
                                              ArrayRef<int64_t> loopSteps,
                                              ValueRange iterArgInit,
                                              BodyBuilderCallback bodyBuilder);

struct ConvertCnmSetZeroToAffine : public OpConversionPattern<cnm::SetZeroOp> {
  using OpConversionPattern<cnm::SetZeroOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(cnm::SetZeroOp, OpAdaptor,
                                ConversionPatternRewriter &) const override;
};

} // namespace mlir

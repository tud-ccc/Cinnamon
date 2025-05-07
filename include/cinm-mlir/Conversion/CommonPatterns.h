#pragma once

#include <cinm-mlir/Dialect/Cinm/IR/CinmUtils.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/AffineMap.h>
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
struct ConvertCnmSetZeroToAffine : public OpConversionPattern<cnm::SetZeroOp> {
  using OpConversionPattern<cnm::SetZeroOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(cnm::SetZeroOp, OpAdaptor,
                                ConversionPatternRewriter &) const override;
};

SmallVector<Value> createAffineApply(OpBuilder &builder, Location loc,
                                     AffineMap map, ValueRange values);

void createMemrefSubviewCopy(OpBuilder &builder, Location loc, Value src,
                             Value dst, ArrayRef<int64_t> sliceShape,
                             ValueRange srcOffsets, ValueRange dstOffsets);

} // namespace mlir

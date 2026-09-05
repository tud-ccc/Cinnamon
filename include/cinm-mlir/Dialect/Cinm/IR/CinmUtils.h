#pragma once

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

namespace mlir::cinm {

/// Whether `value` is known to hold the same data on every inference, so
/// that pinning it on an accelerator amortizes its transfer over the serving
/// lifetime. A value is static iff it is
///  - a function argument carrying the `cinm.static` arg attribute
///    (CinmDialect::STATIC_ATTR_NAME) -- the serving contract, declared by
///    the frontend;
///  - a compile-time constant;
///  - a view of a static value taken at compile-time-constant offsets,
///    sizes and strides (`tensor.extract_slice`, `memref.subview`) -- the
///    same window of the same tensor each time. A dynamically-indexed view
///    of static data is *not* static: the data moved per inference varies;
///  - produced by an op carrying `cinm.static` itself. A buffer is filled by
///    an op that writes it, not produced by one, so a pass that packs static
///    data into a fresh allocation records that on the allocation -- the
///    derivation cannot rediscover it, and the writing op may belong to a
///    dialect this library does not depend on.
/// A `cinm.compute_block` region argument delegates to the corresponding
/// outer operand, so operands may be classified from inside the isolated
/// body.
bool isStaticValue(Value value);

using BodyBuilderCallback = function_ref<SmallVector<Value>(
    OpBuilder &, Location, ValueRange, ValueRange)>;

// Overload for fully static loop sizes.
SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<int64_t> loopSizes,
                                              ArrayRef<int64_t> loopSteps,
                                              ValueRange iterArgInit,
                                              BodyBuilderCallback bodyBuilder);

// Overload for mixed static/dynamic loop sizes. Static sizes are represented
// as IntegerAttr, dynamic sizes as Values. Loop steps remain static.
SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<OpFoldResult> loopSizes,
                                              ArrayRef<int64_t> loopSteps,
                                              ValueRange iterArgInit,
                                              BodyBuilderCallback bodyBuilder);
} // namespace mlir::cinm

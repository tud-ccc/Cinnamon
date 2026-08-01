
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

#include <optional>

namespace mlir {

SmallString<20> getUniqueFunctionName(ModuleOp &moduleOp, StringRef prefix);

/// Check that the memref is contiguous in the dimensions corresponding to the
/// bufShape, which is a suffix of the shape of the input tensor/memref.
bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value,
                                 llvm::ArrayRef<int64_t> bufShape);

/// Returns the number of trailing elements of `type` that are guaranteed to
/// be laid out contiguously in memory (i.e. the largest suffix of dimensions
/// that is packed row-major), or -1 if this cannot be determined statically
/// (dynamic shape/strides, or an unsupported layout). This is the same
/// criterion upmem::ScatterOp/GatherOp::verify() uses to reject transfers
/// that wouldn't be safe as a single flat memcpy per DPU.
int64_t getContiguousSuffixSize(MemRefType type);

/// Simplify an affine map given static upper bounds on the inputs.
/// This is used to simplify even more the affine maps on the CNM and UPMEM
/// levels, given knowledge of the workgroup shape. That makes the generated code
/// simpler, and gives more opportunities for broadcasting.
AffineMap simplifyAffineMapWithBounds(AffineMap map,
                                      llvm::ArrayRef<int64_t> dimSizes);

// Turn an index in the index space of the given shape into a linear index.
AffineExpr linearizeIndices(MLIRContext *ctx, ArrayRef<int64_t> shape);

// inflate a linear index into the given shape
void structureIndex(AffineExpr index, ArrayRef<int64_t> shape,
                    SmallVectorImpl<AffineExpr> &map);

/// The value that every element of the shaped value \p v is statically known
/// to hold, or nullopt if that is not known. Recognizes the three spellings a
/// uniform value takes in this compiler:
///  - a splat `arith.constant dense<...>` (or anything folding to one),
///  - a `linalg.fill` with a constant scalar input,
///  - a `memref.get_global` of a constant global with a splat initializer.
///
/// This only looks at SSA definitions. In particular it says nothing about a
/// memref that some earlier op filled in place: proving that needs alias and
/// effect analysis, so callers that care about such cases must run before
/// bufferization.
std::optional<TypedAttr> getUniformValue(Value v);

/// Returns true if \p v folds to a splat-zero tensor or memref constant.
/// Beyond matching a direct `arith.constant dense<0>`, this also tries to fold
/// the defining op with whatever constant operands are available, so it catches
/// patterns like `linalg.fill(0, tensor.empty())`.
bool isZeroSplatFoldable(Value v);

/// Create a tensor.reshape for a fully static tensor shape
TypedValue<ShapedType> reshapeStatic(OpBuilder &, Location loc, Value value,
                                     ShapedType type,
                                     llvm::ArrayRef<int64_t> newShape);

/// Create a tensor.reshape for a fully static tensor shape
TypedValue<ShapedType> reshapeStatic(OpBuilder &b, Location loc,
                                     TypedValue<ShapedType> value,
                                     llvm::ArrayRef<int64_t> newShape);

} // namespace mlir

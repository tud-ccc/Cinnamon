
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {

SmallString<20> getUniqueFunctionName(ModuleOp &moduleOp, StringRef prefix);

/// Check that the memref is contiguous in the dimensions corresponding to the
/// bufShape, which is a suffix of the shape of the input tensor/memref.
bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value,
                                 llvm::ArrayRef<int64_t> bufShape);

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

/// Create a tensor.reshape for a fully static tensor shape
TypedValue<ShapedType> reshapeStatic(OpBuilder &, Location loc, Value value,
                                     ShapedType type,
                                     llvm::ArrayRef<int64_t> newShape);

/// Create a tensor.reshape for a fully static tensor shape
TypedValue<ShapedType> reshapeStatic(OpBuilder &b, Location loc,
                                     TypedValue<ShapedType> value,
                                     llvm::ArrayRef<int64_t> newShape);

} // namespace mlir


#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/TilingInterface.h>

#include <optional>

namespace mlir {

SmallString<20> getUniqueFunctionName(ModuleOp &moduleOp, StringRef prefix);

/// Whether \p ty 's elements occupy one unbroken run of memory, i.e. whether
/// it can be moved by a single contiguous copy.
///
/// Unit dimensions address nothing and so constrain nothing; every other
/// dimension's stride must equal the product of the sizes below it. A dynamic
/// size, a dynamic stride or a layout that is not strided all count as not
/// contiguous, since none of them can be shown to be at compile time.
bool memrefIsContiguous(MemRefType ty);

/// Check that the memref is contiguous in the dimensions corresponding to the
/// bufShape, which is a suffix of the shape of the input tensor/memref.
bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value,
                                 llvm::ArrayRef<int64_t> bufShape);

/// Returns the number of trailing elements of `type` that are guaranteed to
/// be laid out contiguously in memory (i.e. the largest suffix of dimensions
/// that is packed row-major), or -1 if this cannot be determined statically
/// (dynamic shape/strides, or an unsupported layout). This is the same
/// criterion the upmem transfer ops' verifiers use to reject transfers that
/// wouldn't be safe as a flat memcpy.
int64_t getContiguousSuffixSize(MemRefType type);

/// How many trailing dimensions of `type` make up that contiguous suffix, or
/// -1 in the cases getContiguousSuffixSize returns -1. The memref is then a
/// regular grid of contiguous runs, one per index of the leading dimensions.
int64_t getContiguousSuffixRank(MemRefType type);

/// Linearizes `map` -- which must have one result per dimension of `type` --
/// into a single element-offset expression over the map's own dimensions,
/// using `type`'s layout. A strided layout's base offset is dropped: only
/// relative positions matter to the callers. Fails if the layout is neither
/// the identity nor a static StridedLayoutAttr.
FailureOr<AffineExpr> linearizeToElementOffset(AffineMap map, MemRefType type);

/// The exact largest value `expr` takes over the box `[0, extents)`, or
/// nullopt when that cannot be computed. Floordiv and mod by a constant are
/// handled, but no dimension may appear twice: interval arithmetic treats
/// occurrences as independent, and a bound that is merely an
/// over-approximation is no grounds for rejecting anything.
std::optional<int64_t> getAffineUpperBound(AffineExpr expr,
                                           ArrayRef<int64_t> extents);

/// Whether `expr` takes a different value at every point of the box
/// `[0, extents)`, or nullopt when that cannot be decided. Both answers are
/// conclusive; the undecided case is common.
std::optional<bool> isAffineExprInjective(AffineExpr expr,
                                          ArrayRef<int64_t> extents);

/// Simplify an affine map given static upper bounds on the inputs.
/// This is used to simplify even more the affine maps on the CNM and UPMEM
/// levels, given knowledge of the workgroup shape. That makes the generated
/// code simpler, and gives more opportunities for broadcasting.
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

/// Tile \p op with `scf::tileUsingSCF`, but build the inter-tile loops with
/// `affine.for` instead of `scf.for`.
///
/// Worth doing whenever affine passes run on the result: the tile is addressed
/// through the loop induction variables, and only an `affine.for`'s is a valid
/// affine dimension. With `scf.for` the offsets inside the nest stay opaque, so
/// --affine-raise-from-memref, scalar replacement and affine LICM see nothing
/// to work with there.
///
/// \p options is used as given except for its loop type, which is overridden.
/// Fails if \p op is not one `canTileUsingAffineFor` accepts.
FailureOr<scf::SCFTilingResult>
tileUsingAffineFor(RewriterBase &rewriter, TilingInterface op,
                   scf::SCFTilingOptions options,
                   llvm::ArrayRef<int64_t> tileSizes);

/// Whether `tileUsingAffineFor` can tile \p op with \p tileSizes, i.e. whether
/// the op has buffer semantics -- an `affine.for` nest yields no tile back --
/// and every dimension being tiled has a static extent to use as a loop bound.
/// Callers that cannot guarantee this should fall back to `scf::tileUsingSCF`.
bool canTileUsingAffineFor(TilingInterface op,
                           llvm::ArrayRef<int64_t> tileSizes);

} // namespace mlir

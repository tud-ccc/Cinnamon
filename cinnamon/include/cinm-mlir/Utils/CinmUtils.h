
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {

SmallString<20> getUniqueFunctionName(SymbolTable moduleOp, StringRef prefix);

/// Check that the memref is contiguous in the dimensions corresponding to the
/// bufShape, which is a suffix of the shape of the input tensor/memref.
bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value,
                                 llvm::ArrayRef<int64_t> bufShape);
} // namespace mlir

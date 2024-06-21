
#include <cinm-mlir/Utils/CinmUtils.h>
#include <llvm/ADT/SmallString.h>

namespace mlir {

SmallString<20> getUniqueFunctionName(SymbolTable moduleOp, StringRef prefix) {
  // Get a unique global name.
  unsigned stringNumber = 0;
  size_t prefixLen = prefix.size();
  assert(20 > 3 + prefixLen); // make sure this is bigger than the prefix
                              // (prefixes are literals)
  SmallString<20> name(prefix);
  do {
    name.truncate(prefixLen);
    name.append(std::to_string(stringNumber++));
  } while (moduleOp.lookup(name));
  return name;
}

/// Check that the memref is contiguous in the dimensions corresponding to the
/// bufShape, which is a suffix of the shape of the input tensor/memref.
bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value,
                                 llvm::ArrayRef<int64_t> bufShape) {
  if (value.getType().isa<MemRefType>()) {
    auto type = value.getType().cast<MemRefType>();
    if (!type.hasStaticShape())
      return false;

    SmallVector<int64_t> strides;
    int64_t offset; // offset may be dynamic, we don't
    if (failed(getStridesAndOffset(type, strides, offset)))
      return false;

    // MemRef is contiguous if outer dimensions are size-1 and inner
    // dimensions have unit strides.
    int64_t runningStride = 1;
    int64_t curDim = strides.size() - 1;
    int64_t lastDimToCheck = strides.size() - bufShape.size();
    // Finds all inner dimensions with unit strides.
    while (curDim >= lastDimToCheck && strides[curDim] == runningStride) {
      runningStride *= type.getDimSize(curDim);
      --curDim;
    }

    // Check if other dimensions are size-1.
    while (curDim >= lastDimToCheck && type.getDimSize(curDim) == 1) {
      --curDim;
    }

    // All dims are unit-strided or size-1.
    return curDim < lastDimToCheck;
  }
  return true;
}
} // namespace mlir
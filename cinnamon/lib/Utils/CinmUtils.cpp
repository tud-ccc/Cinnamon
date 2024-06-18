
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
} // namespace mlir
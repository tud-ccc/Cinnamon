
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {

SmallString<20> getUniqueFunctionName(SymbolTable moduleOp, StringRef prefix);
}

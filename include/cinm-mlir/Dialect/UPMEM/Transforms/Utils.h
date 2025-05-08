#include "mlir/IR/BuiltinOps.h"
#include <algorithm>

namespace mlir::upmem {

static llvm::FailureOr<SymbolRefAttr> getSymbolPath(SymbolTable fromTable,
                                                    SymbolOpInterface target) {
  if (!fromTable.getOp()->isAncestor(target))
    return failure();

  StringAttr rootPath = target.getNameAttr();
  llvm::SmallVector<FlatSymbolRefAttr, 4> path;
  Operation *table = SymbolTable::getNearestSymbolTable(target->getParentOp());
  while (table != fromTable.getOp()) {
    if (auto asSymbol = llvm::dyn_cast_or_null<SymbolOpInterface>(table)) {
      auto next = SymbolTable::getNearestSymbolTable(table->getParentOp());
      path.push_back(FlatSymbolRefAttr::get(rootPath));
      rootPath = asSymbol.getNameAttr();
      table = next;
    } else {
      return failure();
    }
  }
  std::reverse(path.begin(), path.end());
  return SymbolRefAttr::get(rootPath, path);
}

} // namespace mlir::upmem
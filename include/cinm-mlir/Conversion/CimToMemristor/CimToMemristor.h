

#include <cinm-mlir/Dialect/Cim/IR/CimDialect.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cim {
std::unique_ptr<Pass> createConvertCimToMemristorPass();
} // namespace mlir::cim
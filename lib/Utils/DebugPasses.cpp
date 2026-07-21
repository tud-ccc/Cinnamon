/// Generic, dialect-independent debug passes.
///
/// @file

#include "cinm-mlir/Utils/DebugPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// Prints the current module IR to stdout, unchanged. Useful for inserting
/// at an arbitrary point of a `-pass-pipeline` to inspect intermediate IR
/// without needing `-mlir-print-ir-after`/`-before` to match pass names.
struct PrintModuleIRPass
    : public PassWrapper<PrintModuleIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintModuleIRPass)

  StringRef getArgument() const override { return "print-module-ir"; }

  StringRef getDescription() const override {
    return "Debug pass: prints the current module IR to stdout, unchanged";
  }

  void runOnOperation() override {
    getOperation()->print(llvm::outs());
    llvm::outs() << "\n";
  }
};

} // namespace

void mlir::cinm::registerPrintModuleIRPass() {
  PassRegistration<PrintModuleIRPass>();
}

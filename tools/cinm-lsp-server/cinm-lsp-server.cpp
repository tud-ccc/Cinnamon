/// Main entry point for the cinm-mlir MLIR language server.

#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cim/IR/CimDialect.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmDialect.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorDialect.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"

#ifdef CINM_TORCH_MLIR_ENABLED
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#endif

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

static int asMainReturnCode(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char *argv[]) {
  DialectRegistry registry;
  registerAllDialects(registry);

  registry.insert<cinm::CinmDialect>();
  registry.insert<cim::CimDialect>();
  registry.insert<cnm::CnmDialect>();
  registry.insert<upmem::UPMEMDialect>();
  registry.insert<memristor::MemristorDialect>();

#ifdef CINM_TORCH_MLIR_ENABLED
  registry.insert<torch::Torch::TorchDialect>();
  registry.insert<torch::TorchConversion::TorchConversionDialect>();
#endif

  return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}

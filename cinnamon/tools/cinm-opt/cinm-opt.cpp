/// Main entry point for the cinm-mlir optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)

#include "cinm-mlir/Conversion/CimPasses.h"
#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Conversion/CnmPasses.h"
#include "cinm-mlir/Conversion/MemristorPasses.h"
#include "cinm-mlir/Conversion/UPMEMPasses.h"
#include "cinm-mlir/Conversion/UPMEMToLLVM/UPMEMToLLVM.h"
#include "cinm-mlir/Dialect/Cim/IR/CimDialect.h"
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmDialect.h"
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorDialect.h"
#include "cinm-mlir/Dialect/Memristor/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#ifdef CINM_TORCH_MLIR_ENABLED
#include "cinm-mlir/Conversion/CinmFrontendPasses.h" // Does the TorchToCinm pass
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/RefBackend/Passes.h"
#endif

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllExtensions.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char *argv[]) {
  DialectRegistry registry;
  registerAllDialects(registry);

  registry.insert<cinm::CinmDialect,           //
                  cim::CimDialect,             //
                  cnm::CnmDialect,             //
                  memristor::MemristorDialect, //
                  upmem::UPMEMDialect>();

#ifdef CINM_TORCH_MLIR_ENABLED
  registry.insert<torch::Torch::TorchDialect>();
  registry.insert<torch::TorchConversion::TorchConversionDialect>();

  torch::registerTorchConversionPasses();
  torch::registerConversionPasses();
  torch::registerTorchPasses();
  torch::RefBackend::registerRefBackendPasses();
#endif

  registerAllPasses();
  registerAllExtensions(registry);
#ifdef CINM_TORCH_MLIR_ENABLED
  registerCinmFrontendConversionPasses();
#endif
  registerCinmConversionPasses();
  registerCimConversionPasses();
  registerCnmConversionPasses();
  registerMemristorConversionPasses();
  cim::registerCimTransformsPasses();
  cnm::registerCnmBufferizationExternalModels(registry);
  cnm::registerCnmTransformsPasses();
  cinm::registerCinmTransformsPasses();
  memristor::registerMemristorTransformsPasses();
  upmem::registerConvertUpmemToLLvmInterface(registry);

  registerUPMEMTransformsPasses();
  registerUPMEMConversionPasses();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "cinm-mlir optimizer driver\n", registry));
}

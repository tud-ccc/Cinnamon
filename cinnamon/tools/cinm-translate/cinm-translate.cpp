/// Main entry point for the cinm-mlir optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)
/// @author      Hamid Farzaneh (hamid.farzaneh@tu-dresden.de)
#include "cinm-mlir/Target/UPMEMCpp/UPMEMCppEmitter.h"

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;


int main(int argc, char **argv) {
  registerAllTranslations();
  upmem_emitc::registerUPMEMCppTranslation();
  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}



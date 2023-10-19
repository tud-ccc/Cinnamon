/// Main entry point for the cinm-mlir optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "cinm-mlir/Dialect/Closure/IR/ClosureDialect.h"
//#include "cinm-mlir/Dialect/cinm/IR/cinmDialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "cinm-mlir/Conversion/ClosurePasses.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
//#include "cinm-mlir/Conversion/cinmPasses.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<closure::ClosureDialect
                    //, cinm::cinmDialect
                    >();

    registerAllPasses();
    registerClosureConversionPasses();
    // registercinmConversionPasses();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "cinm-mlir optimizer driver\n", registry));
}

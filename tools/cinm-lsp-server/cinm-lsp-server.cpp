/// Main entry point for the cinm-mlir MLIR language server.

#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

static int asMainReturnCode(LogicalResult r)
{
    return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<cinm::CinmDialect>();

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}

/// Implements the ${dialectNameUpper} dialect ops.
///
/// @file

#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/APFloat.h"

#define DEBUG_TYPE "${dialectNs}-ops"

using namespace mlir;
using namespace mlir::${dialectNs};

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// ${dialectNameUpper}Dialect
//===----------------------------------------------------------------------===//

void ${dialectNameUpper}Dialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Ops.cpp.inc"
        >();
}

// parsers/printers

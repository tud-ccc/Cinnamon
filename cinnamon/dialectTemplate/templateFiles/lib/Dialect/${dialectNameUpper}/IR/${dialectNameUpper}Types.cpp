/// Implements the ${dialectNameUpper} dialect types.
///
/// @file

#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "${dialectNs}-types"

using namespace mlir;
using namespace mlir::${dialectNs};

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ${dialectNameUpper}Dialect
//===----------------------------------------------------------------------===//

void ${dialectNameUpper}Dialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Types.cpp.inc"
        >();
}

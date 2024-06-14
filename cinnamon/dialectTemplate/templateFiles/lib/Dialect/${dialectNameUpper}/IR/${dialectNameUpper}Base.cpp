/// Implements the ${dialectNameUpper} dialect base.
///
/// @file

#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Base.h"

#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Dialect.h"

#define DEBUG_TYPE "${dialectNs}-base"

using namespace mlir;
using namespace mlir::${dialectNs};

//===- Generated implementation -------------------------------------------===//

#include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Base.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ${dialectNameUpper}Dialect
//===----------------------------------------------------------------------===//

void ${dialectNameUpper}Dialect::initialize()
{
    registerOps();
    registerTypes();
}

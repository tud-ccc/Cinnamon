/// Implements the UPMEM dialect base.
///
/// @file

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h"

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/TypeSwitch.h"


#define DEBUG_TYPE "upmem-base"

using namespace mlir;
using namespace mlir::upmem;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.cpp.inc"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// UPMEMDialect
//===----------------------------------------------------------------------===//

void UPMEMDialect::initialize()
{
    registerOps();
    registerTypes();
    addAttributes<MemcpyDirOpAttr>();
}



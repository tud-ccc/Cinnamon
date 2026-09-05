#pragma once

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cnm {

#define GEN_PASS_DECL_CONVERTCNMTOUPMEMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

void populateCnmToUPMEMFinalTypeConversions(TypeConverter &typeConverter);
std::unique_ptr<Pass> createConvertCnmToUPMEMPass();
std::unique_ptr<Pass> createConvertCnmToUPMEMPass(ConvertCnmToUPMEMPassOptions);
} // namespace mlir::cnm

#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {

// The generated pass base lives in `mlir`, so its options struct does too.
#define GEN_PASS_DECL_CONVERTLINALGTOCNMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace cnm {

std::unique_ptr<Pass> createConvertLinalgToCnmPass();
std::unique_ptr<Pass>
createConvertLinalgToCnmPass(ConvertLinalgToCnmPassOptions);

} // namespace cnm
} // namespace mlir

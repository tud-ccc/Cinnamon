
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformOps.h>
#include <cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformPass.h>
#include <mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace cnm {
#define GEN_PASS_DEF_CNMAPPLYTRANSFORMSCRIPTPASS
#include "cinm-mlir/Dialect/Cnm/TransformOps/TransformPass.h.inc"
} // namespace cnm
} // namespace mlir

using namespace mlir;

struct CnmTransformDialectInterpreterPass
    : public transform::TransformInterpreterPassBase<
          CnmTransformDialectInterpreterPass,
          cnm::impl::CnmApplyTransformScriptPassBase> {

  CnmTransformDialectInterpreterPass() = default;
  CnmTransformDialectInterpreterPass(
      const CnmTransformDialectInterpreterPass &pass)
      : TransformInterpreterPassBase(pass) {

    debugTransformRootTag = pass.debugTransformRootTag;
    debugPayloadRootTag = pass.debugPayloadRootTag;
    disableExpensiveChecks = pass.disableExpensiveChecks;
    transformFileName = pass.transformFileName;
    entryPoint = pass.entryPoint;
    transformLibraryPaths = pass.transformLibraryPaths;
  }
  CnmTransformDialectInterpreterPass(
      const cnm::CnmApplyTransformScriptPassOptions &options) {
    debugTransformRootTag = options.debugTransformRootTag;
    debugPayloadRootTag = options.debugPayloadRootTag;
    disableExpensiveChecks = options.disableExpensiveChecks;
    transformFileName = options.transformFileName;
    entryPoint = options.entryPoint;
    transformLibraryPaths = options.transformLibraryPaths;
  }
};

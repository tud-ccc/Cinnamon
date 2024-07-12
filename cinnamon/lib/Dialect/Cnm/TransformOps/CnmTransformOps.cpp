
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/CnmComputeTransforms.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Transform/IR/TransformInterfaces.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#define GET_OP_CLASSES
#include <cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformOps.cpp.inc>

using namespace mlir;
using namespace mlir::transform;

// Implementation of our Transform dialect operation.
// This operation returns a tri-state result that can be one of:
// - success when the transformation succeeded;
// - definite failure when the transformation failed in such a way that
//   following transformations are impossible or undesirable, typically it could
//   have left payload IR in an invalid state; it is expected that a diagnostic
//   is emitted immediately before returning the definite error;
// - silenceable failure when the transformation failed but following
//   transformations are still applicable, typically this means a precondition
//   for the transformation is not satisfied and the payload IR has not been
//   modified. The silenceable failure additionally carries a Diagnostic that
//   can be emitted to the user.
DiagnosedSilenceableFailure CnmExpandDimOp::applyToOne(TransformRewriter &,
                                                       cnm::ComputeOp compute,
                                                       ApplyToEachResultList &,
                                                       TransformState &) {

  if (failed(cnm::expandWorkshoupDim(compute, getDim(), getFactor()))) {
    DiagnosedSilenceableFailure diag = emitDefaultSilenceableFailure(compute);
    diag.attachNote() << "Transform failed";
  }

  return DiagnosedSilenceableFailure::success();
}

void CnmExpandDimOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  // Indicate that the `call` handle is only read by this operation because the
  // associated operation is not erased but rather modified in-place, so the
  // reference to it remains valid.
  onlyReadsHandle(getTarget(), effects);

  // Indicate that the payload is modified by this operation.
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure CnmPeelRightOp::applyToOne(TransformRewriter &,
                                                       cnm::ComputeOp compute,
                                                       ApplyToEachResultList &,
                                                       TransformState &) {

  auto res = cnm::peelRight(compute);
  if (failed(res)) {
    DiagnosedSilenceableFailure diag = emitDefaultSilenceableFailure(compute);
    diag.attachNote() << "Transform failed";
  }

  return DiagnosedSilenceableFailure::success();
}

void CnmPeelRightOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure CnmSwapDimsOp::applyToOne(TransformRewriter &,
                                                      cnm::ComputeOp compute,
                                                      ApplyToEachResultList &,
                                                      TransformState &) {

  auto res = cnm::swapWorkgroupDims(compute, getDim0(), getDim1());
  if (failed(res)) {
    DiagnosedSilenceableFailure diag = emitDefaultSilenceableFailure(compute);
    diag.attachNote() << "Transform failed";
  }
  return DiagnosedSilenceableFailure::success();
}

void CnmSwapDimsOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  modifiesPayload(effects);
}

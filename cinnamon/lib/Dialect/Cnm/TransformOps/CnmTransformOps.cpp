
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/CnmComputeTransforms.h>
#include <mlir/IR/Builders.h>
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
DiagnosedSilenceableFailure CnmExpandDimOp::apply(
    // The rewriter that should be used when modifying IR.
    TransformRewriter &rewriter,
    // The list of payload IR entities that will be associated with the
    // transform IR values defined by this transform operation. In this case, it
    // can remain empty as there are no results.
    TransformResults &results,
    // The transform application state. This object can be used to query the
    // current associations between transform IR values and payload IR entities.
    // It can also carry additional user-defined state.
    TransformState &state) {

  // First, we need to obtain the list of payload operations that are associated
  // with the operand handle.
  auto payload = state.getPayloadOps(getTarget());

  // Then, we iterate over the list of operands and call the actual IR-mutating
  // function. We also check the preconditions here.
  for (Operation *payloadOp : payload) {
    auto compute = dyn_cast<::mlir::cnm::ComputeOp>(payloadOp);
    if (!compute) {
      return emitDefaultSilenceableFailure(payloadOp);
    }

    if (failed(cnm::expandWorkshoupDim(compute, getDim(), getFactor()))) {
      DiagnosedSilenceableFailure diag =
          emitDefaultSilenceableFailure(payloadOp);
      diag.attachNote() << "Transform failed";
    }
  }

  // If everything went well, return success.
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
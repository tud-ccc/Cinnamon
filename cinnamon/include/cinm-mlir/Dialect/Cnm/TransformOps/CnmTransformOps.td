

include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def CnmExpandDimOp : Op<Transform_Dialect, "cnm.expand_dim",
    [FunctionalStyleTransformOpTrait,
     TransformOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::cnm::ComputeOp op,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];

  let arguments = (ins Transform_ConcreteOpType<"cnm.compute">:$target,
                       I64Attr:$dim,
                       I64Attr:$factor);
  let results = (outs);
  let assemblyFormat =
      "$target `dim` $dim `by` `factor` $factor attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::cnm::ComputeOp op,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def CnmSwapDimsOp : Op<Transform_Dialect, "cnm.swap_dims",
    [FunctionalStyleTransformOpTrait,
     TransformOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{

  }];

  let arguments = (ins Transform_ConcreteOpType<"cnm.compute">:$target, 
                       I64Attr:$dim0,
                       I64Attr:$dim1);
  let results = (outs);
  let assemblyFormat =
      "$target `,` $dim0 `,` $dim1 attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::cnm::ComputeOp op,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def CnmPeelRightOp : Op<Transform_Dialect, "cnm.peel_right",
    [FunctionalStyleTransformOpTrait,
     TransformOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{

  }];

  let arguments = (ins Transform_ConcreteOpType<"cnm.compute">:$target);
  let results = (outs);
  let assemblyFormat =
      "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::cnm::ComputeOp op,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
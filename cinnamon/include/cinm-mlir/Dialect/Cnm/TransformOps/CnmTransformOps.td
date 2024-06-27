

include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def CnmExpandDimOp : Op<Transform_Dialect, "cnm.expand_dim",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{

  }];

  let arguments = (ins TransformHandleTypeInterface:$target, 
                       I64Attr:$dim,
                       I64Attr:$factor);
  let results = (outs);
  let assemblyFormat =
      "$target `on` $dim `by` `factor` $factor attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
  }];
}


def CnmPeelRightOp : Op<Transform_Dialect, "cnm.peel_right",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{

  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat =
      "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
  }];
}
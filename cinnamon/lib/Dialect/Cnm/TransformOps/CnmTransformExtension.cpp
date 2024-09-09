
// In CnmTransformExtension.cpp.
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include <cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformOps.h>

// Define a new Transform dialect extension. This uses the CRTP idiom to
// identify extensions.
class CnmTransformExtension
    : public ::mlir::transform::TransformDialectExtension<
          CnmTransformExtension> {
public:
  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect  definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void CnmTransformExtension::init() {
  // Similarly to dialects, an extension can declare a dependent dialect. This
  // dialect will be loaded along with the extension and, therefore, along with
  // the Transform  dialect. Only declare as dependent the dialects that contain
  // the attributes or types used by transform operations. Do NOT declare as
  // dependent the dialects produced during the transformation.
  //
  // declareDependentDialect<MyDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  // In the following chapter, we will be add operations that generate function
  // calls and structured control flow operations, so let's declare the
  // corresponding dialects as generated.
  declareGeneratedDialect<::mlir::cnm::CnmDialect>();
  declareGeneratedDialect<::mlir::memref::MemRefDialect>();

  // Finally, we register the additional transform operations with the dialect.
  registerTransformOps<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cnm/TransformOps/CnmTransformOps.cpp.inc"
      >();
}

void mlir::cnm::registerTransformDialectExtension(
    ::mlir::DialectRegistry &registry) {
  registry.addExtensions<CnmTransformExtension>();
}
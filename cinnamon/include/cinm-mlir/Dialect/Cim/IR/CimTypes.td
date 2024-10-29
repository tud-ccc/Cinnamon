//===- Types.td - Cim dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Cim dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_TYPES
#define CIM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "CimBase.td"

def DeviceIdType : Cim_Type<"DeviceId"> {
  let mnemonic = "deviceId";
  let summary = "Represents an acquired cim device.";
}

def CrossbarIdType : Cim_Type<"CrossbarId"> {
  let mnemonic = "crossbarId";
  let summary = "Represents an acquired cim crossbar.";
}

def FutureType : Cim_Type<"Future", [ShapedTypeInterface]> {
  let mnemonic = "future";
  let summary = "Result of a cim operation";

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "Type":$elementType);

  let hasCustomAssemblyFormat = 1;

  let builders = [TypeBuilderWithInferredContext<
      (ins "ArrayRef<int64_t>":$shape, "Type": $elementType),
      [{ return $_get(elementType.getContext(), shape, elementType); }]
  >];

  let extraClassDeclaration = [{
    bool hasRank() const { return true; }

    ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> newShape, ::mlir::Type elementType) {
      auto shape = newShape.value_or(getShape());
      return FutureType::get(getContext(), shape, elementType);
    }

    int64_t getItemCount() {
      auto shape = getShape();
      return shape.empty() ? 1 : mlir::computeProduct(shape);
    }

    int64_t getSizeInBytes() {
      return getItemCount() * getElementTypeBitWidth() / 8;
    }
  }];
}

#endif

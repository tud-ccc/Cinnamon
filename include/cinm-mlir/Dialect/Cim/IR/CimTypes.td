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

    ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> newShape,
                        ::mlir::Type elementType) {
      auto shape = newShape.value_or(getShape());
      return FutureType::get(getContext(), shape, elementType);
    }

    int64_t getItemCount() {
      auto shape = getShape();
      if (shape.empty()) return 1;
      int64_t prod = 1;
      for (int64_t d : shape) {
        if (mlir::ShapedType::isDynamic(d)) return mlir::ShapedType::kDynamic;
        prod *= d;
      }
      return prod;
    }

    int64_t getSizeInBytes() {
      auto items = getItemCount();
      if (items == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
      return items * getElementTypeBitWidth() / 8;
    }
}];
}

#endif

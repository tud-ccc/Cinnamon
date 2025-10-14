//===- Types.td - Cnm dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_TYPES
#define CNM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "CnmBase.td"

def WorkgroupType : Cnm_Type<"Workgroup"> {
  let mnemonic = "workgroup";
  let summary = "";

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    int64_t getNumElements() {
      auto wgShape = getShape();
      return std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
    }
  }];
}

def BufferType : Cnm_Type<"Buffer", [ShapedTypeInterface]> {
  let mnemonic = "buffer";
  let summary = "";

  let parameters = (
    ins ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$workgroupShape,
    "int64_t":$level
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins   
    "ArrayRef<int64_t>":$shape, 
    "Type": $elementType,
    "ArrayRef<int64_t>":$workgroupShape,
    CArg<"int64_t", "0">:$level
    ), [{
      return $_get(elementType.getContext(), shape, elementType, workgroupShape, level);
    }]>
  ];

  let extraClassDeclaration = [{
    bool hasRank() { return true; }
    ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> newShape, ::mlir::Type elementType) {
      auto shape = newShape.value_or(getShape());
      return BufferType::get(getContext(), shape, elementType, getWorkgroupShape(), getLevel());
    }
    int64_t getItemCount() {
      auto bufShape = getShape();
      return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
    }

    int64_t getSizeInBytes() {
      return getItemCount() * getElementTypeBitWidth() / 8;
    }
  }];
}


#endif

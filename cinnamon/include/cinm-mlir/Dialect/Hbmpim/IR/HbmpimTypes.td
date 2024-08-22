//===- Types.td - Hbmpim dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Hbmpim dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef HBMPIM_TYPES
#define HBMPIM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "HbmpimBase.td"
include "HbmpimAttributes.td"

// def PIMCommandsType : Hbmpim_Type<"PIMCMDs"> {
//   let mnemonic = "PIMCMDs";
//   let summary = "";

//   let parameters = (ins ArrayRefParameter<"PIMCMD">:$shape);
//   let hasCustomAssemblyFormat = 1;
//   let extraClassDeclaration = [{
//   }];
// }

def DeviceConfiguration : Hbmpim_Type<"DeviceConfiguration"> {
  let mnemonic = "configuration";
  let summary = "";

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    int64_t getNumElements() {
      auto wgShape = getShape();
      return std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
    }
    int64_t getNumDims() { return getShape().size();}

    int64_t getNumBanks() {
      return getShape()[0];
    }
    int64_t getNumChannels() {
      return getShape()[1];
    }
    int64_t getNumRanks() {
      return getShape()[2];
    }
    int64_t getNumGrf() {
      return getShape()[3];
    }
  }];
}

def PIMCommandsVectorType : Hbmpim_Type<"PIMCMDVec"> {
  let mnemonic = "PIMCMDVec";
  let summary = "";

  let parameters = (ins "PimKernelType":$kerType);
  let extraClassDeclaration = [{
  }];
  let hasCustomAssemblyFormat = 1;
}

def NumpyBurstType : Hbmpim_Type<"NumpyBurstType"> {
  let mnemonic = "NumpyBurstType";
  let summary = "";

  // let parameters = (ins "PimKernelType":$kerType);
  let extraClassDeclaration = [{
  }];
  // let hasCustomAssemblyFormat = 1;
}

def Burst: Hbmpim_Type<"Burst"> {
  let mnemonic = "Burst";
  let summary = "";

  let parameters = (ins "DeviceBurstType":$burstType);
  let extraClassDeclaration = [{
  }];
  let hasCustomAssemblyFormat = 1;
}

def OperandDataDim: Hbmpim_Type<"OperandDataDim"> {
  let mnemonic = "OperandDataDim";
  let summary = "";

  let parameters = (ins "DataDimType":$dataDim);
  let extraClassDeclaration = [{
  }];
  let hasCustomAssemblyFormat = 1;
}

def OperandToken: Hbmpim_Type<"OperandToken"> {
  let mnemonic = "OperandToken";
  let summary = "";

  let extraClassDeclaration = [{
  }];
}

// def BufferType : Hbmpim_Type<"Buffer", [ShapedTypeInterface]> {
//   let mnemonic = "buffer";
//   let summary = "";

//   let parameters = (
//     ins ArrayRefParameter<"int64_t">:$shape,
//     "Type":$elementType,
//     ArrayRefParameter<"int64_t">:$workgroupShape,
//     "int64_t":$level
//   );

//   let hasCustomAssemblyFormat = 1;

//   let builders = [
//     TypeBuilderWithInferredContext<(ins   
//     "ArrayRef<int64_t>":$shape, 
//     "Type": $elementType,
//     "ArrayRef<int64_t>":$workgroupShape,
//     CArg<"int64_t", "0">:$level
//     ), [{
//       return $_get(elementType.getContext(), shape, elementType, workgroupShape, level);
//     }]>
//   ];

//   let extraClassDeclaration = [{
//     bool hasRank() { return true; }
//     ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> newShape, ::mlir::Type elementType) {
//       auto shape = newShape.value_or(getShape());
//       return BufferType::get(getContext(), shape, elementType, getWorkgroupShape(), getLevel());
//     }
//     int64_t getItemCount() {
//       auto bufShape = getShape();
//       return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
//     }

//     int64_t getSizeInBytes() {
//       return getItemCount() * getElementTypeBitWidth() / 8;
//     }
//   }];
// }


#endif

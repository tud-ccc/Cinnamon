//===- Attributes.td - Cim dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the Cim dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_ATTRIBUTES
#define CIM_ATTRIBUTES

include "cinm-mlir/Dialect/Cim/IR/CimBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"

// Quantization rounding mode 
def Cim_RoundingMode : I32EnumAttr<
    "RoundingMode", "quantization rounding mode", [
  I32EnumAttrCase<"Nearest",     0, "nearest">,
  I32EnumAttrCase<"TowardsZero", 1, "towards_zero">
]> {
  let cppNamespace = "::mlir::cim";
}

def Cim_ActivationKindAttr : I64EnumAttr<
    "ActivationKind", "",
    [
      I64EnumAttrCase<"RELU",    0, "relu">,
      I64EnumAttrCase<"SIGMOID", 1, "sigmoid">,
      I64EnumAttrCase<"TANH",    2, "tanh">,
      I64EnumAttrCase<"GELU",    3, "gelu">
    ]> {
  let cppNamespace = "::mlir::cim";
}

#endif

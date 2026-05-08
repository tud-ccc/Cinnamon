//===- Attributes.td - Cinm dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_OP_ATTRIBUTES
#define CINM_OP_ATTRIBUTES

include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"

def Cinm_ScanMethodAttr : I64EnumAttr<
    "ScanMethod", "",
    [
      I64EnumAttrCase<"ADD", 0, "add">,
      I64EnumAttrCase<"MUL", 1, "mul">
    ]> {
  let cppNamespace = "::mlir::cinm";
}

def Cinm_ReduceMethodAttr : I64EnumAttr<
    "ReduceMethod", "",
    [
      I64EnumAttrCase<"ADD", 0, "add">,
      I64EnumAttrCase<"MUL", 1, "mul">,

      I64EnumAttrCase<"MAXSI", 2, "maxsi">,
      I64EnumAttrCase<"MAXUI", 3, "maxui">,
      I64EnumAttrCase<"MAXIMUMF", 4, "maximumf">,
      I64EnumAttrCase<"MAXNUMF", 5, "maxnumf">,

      I64EnumAttrCase<"MINSI", 6, "minsi">,
      I64EnumAttrCase<"MINUI", 7, "minui">,
      I64EnumAttrCase<"MINIMUMF", 8, "minimumf">,
      I64EnumAttrCase<"MINNUMF", 9, "minnumf">,
    ]> {
  let cppNamespace = "::mlir::cinm";
}

def Cinm_ElementwiseKindAttr : I64EnumAttr<
    "ElementwiseKind", "",
    [
      // Int / Float Operations
      I64EnumAttrCase<"Add", 0, "add">,
      I64EnumAttrCase<"Sub", 1, "sub">,
      I64EnumAttrCase<"Mul", 2, "mul">,
      I64EnumAttrCase<"Div", 3, "div">,
      I64EnumAttrCase<"Mod", 4, "mod">,
      // unaries
      I64EnumAttrCase<"Neg", 5, "neg">,
      I64EnumAttrCase<"Abs", 6, "abs">,
      I64EnumAttrCase<"Ceil", 7, "ceil">,
      I64EnumAttrCase<"Erf", 8, "erf">,
      I64EnumAttrCase<"Exp", 9, "exp">,
      I64EnumAttrCase<"Floor", 10, "floor">,
      I64EnumAttrCase<"Log", 11, "log">,
      I64EnumAttrCase<"Reciprocal", 12, "reciprocal">,
      I64EnumAttrCase<"Round", 13, "round">,
      I64EnumAttrCase<"Rsqrt", 14, "rsqrt">,
      I64EnumAttrCase<"Sqrt", 15, "sqrt">,
      I64EnumAttrCase<"Square", 16, "square">,
      I64EnumAttrCase<"Tanh", 17, "tanh">,

      // Bitwise Operations
      I64EnumAttrCase<"And", 18, "and">,
      I64EnumAttrCase<"Or", 19, "or">,
      I64EnumAttrCase<"Xor", 20, "xor">,
      I64EnumAttrCase<"Not", 21, "not">
    ]> {
  let cppNamespace = "::mlir::cinm";
}


def Cinm_SimilarityMetricAttr : I64EnumAttr<
    "SimilarityMetric", "",
    [
      // dot product similarity
      I64EnumAttrCase<"DOT", 0, "dot">,
      // cosine similarity
      I64EnumAttrCase<"COS", 1, "cos">
    ]> {
  let cppNamespace = "::mlir::cinm";
}

//def Cinm_SimilarityMetricAttr : EnumAttr<Cinm_Dialect, Cinm_SimilarityMetric, "metric">;
//def Cinm_ReduceMethodAttr : EnumAttr<Cinm_Dialect, Cinm_ReduceMethod, "method">;
//def Cinm_ScanMethodAttr : EnumAttr<Cinm_Dialect, Cinm_ScanMethod, "method">;

def Cinm_ActivationKindAttr : I64EnumAttr<
    "ActivationKind", "",
    [
      I64EnumAttrCase<"RELU",    0, "relu">,
      I64EnumAttrCase<"SIGMOID", 1, "sigmoid">,
      I64EnumAttrCase<"TANH",    2, "tanh">,
      I64EnumAttrCase<"GELU",    3, "gelu">
    ]> {
  let cppNamespace = "::mlir::cinm";
}

//===----------------------------------------------------------------------===//
// Quantization
//===----------------------------------------------------------------------===//

def Cinm_RoundingMode : I32EnumAttr<
    "RoundingMode", "quantization rounding mode", [
  I32EnumAttrCase<"Nearest",     0, "nearest">,
  I32EnumAttrCase<"TowardsZero", 1, "towards_zero">
]> {
  let cppNamespace = "::mlir::cinm";
}

#endif

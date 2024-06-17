//===- Attributes.td - Cinm dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_ATTRIBUTES
#define CINM_ATTRIBUTES

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
      I64EnumAttrCase<"MAX", 2, "max">,
      I64EnumAttrCase<"MIN", 3, "min">,
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

#endif

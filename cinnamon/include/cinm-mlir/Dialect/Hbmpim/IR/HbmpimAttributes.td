//===- Attributes.td - Hbmpim dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the Hbmpim dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef HBMPIM_ATTRIBUTES
#define HBMPIM_ATTRIBUTES

include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"

// def PIMCMDAttr : I64EnumAttr<
//     "PIMCMD", "",
//     [
//       I64EnumAttrCase<"NOP", 0, "nop">,
//       I64EnumAttrCase<"ADD", 1, "add">,
//       I64EnumAttrCase<"MUL", 2, "mul">,
//       I64EnumAttrCase<"MAC", 3, "mac">,
//       I64EnumAttrCase<"MAD", 4, "mad">,
//       I64EnumAttrCase<"REV0", 5, "rev0">,
//       I64EnumAttrCase<"REV1", 6, "rev1">,
//       I64EnumAttrCase<"REV2", 7, "rev2">,
//       I64EnumAttrCase<"MOV",  8, "mov">,
//       I64EnumAttrCase<"FILL", 9, "fill">,
//       I64EnumAttrCase<"REV3", 10, "rev3">,
//       I64EnumAttrCase<"REV4", 11, "rev4">,
//       I64EnumAttrCase<"REV5", 12, "rev5">,
//       I64EnumAttrCase<"REV6", 13, "rev6">,
//       I64EnumAttrCase<"JUMP", 14, "jump">,
//       I64EnumAttrCase<"EXIT", 16, "exit">
//     ]> {
//   let cppNamespace = "::mlir::hbmpim";
// }

def PimKernelTypeAttr : I64EnumAttr<
    "PimKernelType", "",
    [
      I64EnumAttrCase<"RELU", 0, "RELU">,
      I64EnumAttrCase<"MUL", 1, "MUL">,
      I64EnumAttrCase<"ADD", 2, "ADD">,
      I64EnumAttrCase<"GEMV", 3, "GEMV">
    ]> {
  let cppNamespace = "::mlir::hbmpim";
}

def DataTypeAttr : I64EnumAttr<
    "DataType", "",
    [
      I64EnumAttrCase<"INT8", 0, "INT8">,
      I64EnumAttrCase<"FP16", 1, "FP16">,
      I64EnumAttrCase<"FP32", 2, "FP32">
    ]> {
  let cppNamespace = "::mlir::hbmpim";
}



def DeviceBurstTypeAttr : I64EnumAttr<
    "DeviceBurstType", "",
    [
      I64EnumAttrCase<"null_bst_", 0, "null_bst_">,
      I64EnumAttrCase<"bst_hab_pim_", 1, "bst_hab_pim_">,
      I64EnumAttrCase<"bst_hab_", 2, "bst_hab_">,
      I64EnumAttrCase<"crf_bst_", 3, "crf_bst_">,
      I64EnumAttrCase<"srf_bst_", 4, "srf_bst_">,
      I64EnumAttrCase<"result", 5, "result">
    ]> {
  let cppNamespace = "::mlir::hbmpim";
}

def DataDimTypeAttr : I64EnumAttr<
    "DataDimType", "",
    [
      I64EnumAttrCase<"input_npbst_", 0, "input_npbst_">,
      I64EnumAttrCase<"input1_npbst_", 1, "input1_npbst_">,
      I64EnumAttrCase<"weight_npbst_", 2, "weight_npbst_">,
      I64EnumAttrCase<"output_npbst_", 3, "output_npbst_">
    ]> {
  let cppNamespace = "::mlir::hbmpim";
}


def PimBankTypeAttr : I64EnumAttr<
    "PimBankType", "",
    [
      I64EnumAttrCase<"EVEN_BANK", 0, "EVEN_BANK">,
      I64EnumAttrCase<"ODD_BANK", 1, "ODD_BANK">,
      I64EnumAttrCase<"ALL_BANK", 2, "ALL_BANK">
    ]> {
  let cppNamespace = "::mlir::hbmpim";
}


def DRAMModeAttr : I64EnumAttr<
    "DRAMMode", "",
    [
      I64EnumAttrCase<"SB", 0, "SB">,
      I64EnumAttrCase<"HAB", 1, "HAB">,
      I64EnumAttrCase<"HAB_PIM", 2, "HAB_PIM">
    ]> {
  let cppNamespace = "::mlir::hbmpim";
}

#endif

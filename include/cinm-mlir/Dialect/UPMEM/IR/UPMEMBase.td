//===- Base.td - UPMEM dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_BASE
#define UPMEM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"


def UPMEM_Dialect : Dialect {
    let name = "upmem";
    let cppNamespace = "::mlir::upmem";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    let useDefaultTypePrinterParser = 1;
    let hasOperationAttrVerify = 1;
    // let useFoldAPI = kEmitFoldAdaptorFolder;


    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
        void registerAttributes();
    public:
    }];
    let dependentDialects = ["cinm::CinmDialect"];
    let useDefaultAttributePrinterParser = 1;
    // let useDefaultTypePrinterParser = 1;
}

class Upmem_Op<string mnemonic, list<Trait> traits = []>
        : Op<UPMEM_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// UPMEM Enums.
//===----------------------------------------------------------------------===//

class UPMEM_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::upmem";
}
class UPMEM_I32EnumAttr<string mnemonic, UPMEM_I32Enum enumInfo> :
    EnumAttr<UPMEM_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}



class UPMEM_Attr<string name, list<Trait> traits = []>
        : AttrDef<UPMEM_Dialect, name, traits>;

class UPMEM_Type<string name, list<Trait> traits = []>
        : TypeDef<UPMEM_Dialect, name, traits>;





//===----------------------------------------------------------------------===//
// UPMEM Interfaces.
//===----------------------------------------------------------------------===//


#endif

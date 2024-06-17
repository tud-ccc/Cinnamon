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
    public:
        static StringRef getContainerModuleAttrName() {
            return "upmem.container_module";
        }
        static StringRef getKernelFuncAttrName() { return "upmem.kernel"; }
        static bool isKernel(Operation *op);
        static unsigned getNumWorkgroupDimensions() { return 1; }
        // static AddressSpace getWorkgroupAddressSpace() { return AddressSpace::Workgroup; }
        // static AddressSpace getPrivateAddressSpace() { return AddressSpace::Private; }
    }];
    // let dependentDialects = ["arith::ArithDialect"];
    let useDefaultAttributePrinterParser = 1;
    // let useDefaultTypePrinterParser = 1;
}

class UPMEM_Op<string mnemonic, list<Trait> traits = []>
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




class UPMEM_Type<string name, list<Trait> traits = []>
        : TypeDef<UPMEM_Dialect, name, traits>;





//===----------------------------------------------------------------------===//
// UPMEM Interfaces.
//===----------------------------------------------------------------------===//


#endif

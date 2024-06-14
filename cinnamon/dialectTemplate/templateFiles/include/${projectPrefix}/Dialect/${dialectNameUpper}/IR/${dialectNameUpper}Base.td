//===- Base.td - ${dialectNameUpper} dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the ${dialectNameUpper} dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef ${dialectNameAllCaps}_BASE
#define ${dialectNameAllCaps}_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def ${dialectNameUpper}_Dialect : Dialect {
    let name = "${dialectNs}";
    let cppNamespace = "::mlir::${dialectNs}";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    let useDefaultTypePrinterParser = 1;
    let useFoldAPI = kEmitFoldAdaptorFolder;


    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
// class ${dialectNameUpper}_Attr<string name, list<Trait> traits = []>
//         : AttrDef<${dialectNameUpper}_Dialect, name, traits>;
// Template for ops.
class ${dialectNameUpper}_Op<string mnemonic, list<Trait> traits = []>
        : Op<${dialectNameUpper}_Dialect, mnemonic, traits>;
// Template for types.
class ${dialectNameUpper}_Type<string name, list<Trait> traits = []>
        : TypeDef<${dialectNameUpper}_Dialect, name, traits>;

#endif

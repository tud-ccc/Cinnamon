//===- Base.td - Cinm dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_BASE
#define CINM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Cinm_Dialect : Dialect {
    let name = "cinm";
    let cppNamespace = "::mlir::cinm";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    let useDefaultTypePrinterParser = 1;


    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
// class Cinm_Attr<string name, list<Trait> traits = []>
//         : AttrDef<Cinm_Dialect, name, traits>;
// Template for ops.
class Cinm_Op<string mnemonic, list<Trait> traits = []>
        : Op<Cinm_Dialect, mnemonic, traits>;
// Template for types.
class Cinm_Type<string name, list<Trait> traits = []>
        : TypeDef<Cinm_Dialect, name, traits>;

#endif

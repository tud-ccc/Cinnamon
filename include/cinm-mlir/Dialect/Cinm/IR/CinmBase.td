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

    let usePropertiesForAttributes = 0;
//    let useDefaultAttributePrinterParser = 1;
//    let useDefaultTypePrinterParser = 1;
    let hasOperationAttrVerify = 1;


    code extraClassDeclaration = [{
    constexpr static StringRef NOTILE_NAME = "cinm.notile";
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
 class Cinm_Attr<string name, list<Trait> traits = []>
         : AttrDef<Cinm_Dialect, name, traits>;
// Template for ops.
class Cinm_Base_Op<string mnemonic, list<Trait> traits = []>
        : Op<Cinm_Dialect, mnemonic, traits>;

class Cinm_Op<string mnemonic, list<Trait> traits = []>
        : Cinm_Base_Op<mnemonic, !listconcat(traits, [
            // we want an ancestor not a parent
            // HasParent<"ComputeOp">
        ])>;
// Template for types.
class Cinm_Type<string name, list<Trait> traits = []>
        : TypeDef<Cinm_Dialect, name, traits>;

#endif

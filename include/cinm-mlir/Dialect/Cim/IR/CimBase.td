//===- Base.td - Cim dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Cim dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_BASE
#define CIM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Cim_Dialect : Dialect {
    let name = "cim";
    let cppNamespace = "::mlir::cim";

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
class Cim_Attr<string name, list<Trait> traits = []>
        : AttrDef<Cim_Dialect, name, traits>;

// Template for ops.
class Cim_Op<string mnemonic, list<Trait> traits = []>
        : Op<Cim_Dialect, mnemonic, traits>;

// Template for types.
class Cim_Type<string name, list<Trait> traits = []>
        : TypeDef<Cim_Dialect, name, traits>;

#endif

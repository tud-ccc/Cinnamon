//===- Base.td - Hbmpim dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Hbmpim dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef HBMPIM_BASE
#define HBMPIM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Hbmpim_Dialect : Dialect {
    let name = "hbmpim";
    let cppNamespace = "::mlir::hbmpim";
    

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
class Hbmpim_Attr<string name, list<Trait> traits = []>
        : AttrDef<Hbmpim_Dialect, name, traits>;

// Template for ops.
class Hbmpim_Op<string mnemonic, list<Trait> traits = []>
        : Op<Hbmpim_Dialect, mnemonic, traits>;

// Template for types.
class Hbmpim_Type<string name, list<Trait> traits = []>
        : TypeDef<Hbmpim_Dialect, name, traits>;

#endif

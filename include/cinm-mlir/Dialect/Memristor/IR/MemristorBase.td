//===- Base.td - Memristor dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Memristor dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef MEMRISTOR_BASE
#define MEMRISTOR_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Memristor_Dialect : Dialect {
    let name = "memristor";
    let cppNamespace = "::mlir::memristor";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    code extraClassDeclaration = [{
    private:
        void registerOps();
    }];
}

// Template for attributes.
class Memristor_Attr<string name, list<Trait> traits = []>
        : AttrDef<Memristor_Dialect, name, traits>;

// Template for ops.
class Memristor_Op<string mnemonic, list<Trait> traits = []>
        : Op<Memristor_Dialect, mnemonic, traits> {
    code libraryCallName = [{
      std::string getLibraryCallName() {
        return generateLibraryCallName(getOperation());
      }
    }];

    let extraClassDeclaration = libraryCallName;
}

// Template for types.
class Memristor_Type<string name, list<Trait> traits = []>
        : TypeDef<Memristor_Dialect, name, traits>;

#endif

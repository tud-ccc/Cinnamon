//===- Base.td - Alpine dialect base ------------------------*- tablegen -*-===//

#ifndef ALPINE_BASE
#define ALPINE_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Alpine_Dialect : Dialect {
    let name = "alpine";
    let cppNamespace = "::mlir::alpine";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    code extraClassDeclaration = [{
    private:
        void registerOps();
    }];
    let usePropertiesForAttributes = 0;

}

// Template for attributes.
class Alpine_Attr<string name, list<Trait> traits = []>
        : AttrDef<Alpine_Dialect, name, traits>;

// Template for ops.
class Alpine_Op<string mnemonic, list<Trait> traits = []>
        : Op<Alpine_Dialect, mnemonic, traits> {
    code libraryCallName = [{
      std::string getLibraryCallName() {
        return generateLibraryCallName(getOperation());
      }
    }];

    let extraClassDeclaration = libraryCallName;
}

// Template for types.
class Alpine_Type<string name, list<Trait> traits = []>
        : TypeDef<Alpine_Dialect, name, traits>;

#endif

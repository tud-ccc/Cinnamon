//===- Base.td - Cnm dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_BASE
#define CNM_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Cnm_Dialect : Dialect {
    let name = "cnm";
    let cppNamespace = "::mlir::cnm";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    let useDefaultTypePrinterParser = 1;

    code extraClassDeclaration = [{
        /// Discardable attribute on a linalg op, read by
        /// `--convert-linalg-to-cnm`: one *block size* per iteration
        /// dimension of the op, i.e. the extent of that dimension inside a
        /// single workgroup leaf. The tile count is derived as
        /// `dimSize / blockSize`, and the product of the tile counts must
        /// equal the number of leaves in the workgroup.
        ///
        /// Note the unit: block sizes, like `cinm.tile_sizes`, *not* tile
        /// counts. See docs/CnmMemoryLevelsDesign.md §G2.
        constexpr static StringRef TILE_SIZES_NAME = "cnm.tile_sizes";

    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class Cnm_Attr<string name, list<Trait> traits = []>
        : AttrDef<Cnm_Dialect, name, traits>;

// Template for ops.
class Cnm_Op<string mnemonic, list<Trait> traits = []>
        : Op<Cnm_Dialect, mnemonic, traits>;

// Template for types.
class Cnm_Type<string name, list<Trait> traits = []>
        : TypeDef<Cnm_Dialect, name, traits>;

#endif

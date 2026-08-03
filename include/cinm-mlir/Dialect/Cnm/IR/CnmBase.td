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

        /// Discardable attributes on a linalg op, read by
        /// `--convert-linalg-to-cnm`: which iteration dimension occupies
        /// which workgroup axis, outermost (slowest-varying across leaves)
        /// first. Two forms of the same choice, at most one of which may be
        /// present:
        ///
        /// - `cnm.workgroup_dim_order`: a `DenseI64ArrayAttr` permutation of
        ///   the op's iteration dimensions, stated over the dimensions it has
        ///   once any reduction split has prepended one.
        /// - `cnm.workgroup_dim_order_index`: an `IntegerAttr` naming the
        ///   same order by its rank in lexicographic order, counting only the
        ///   dimensions actually spread over the workgroup. Index 0 is the
        ///   default rule. This is the form the search stamps.
        ///
        /// Either overrides the pass's own `workgroup-dim-order` /
        /// `workgroup-dim-order-index` options for this op. Unlike
        /// `cnm.tile_sizes` these are *not* per-dimension lists, so splitting
        /// a reduction carries them across unchanged. See
        /// docs/CnmMemoryLevelsDesign.md §G3.
        constexpr static StringRef WORKGROUP_DIM_ORDER_NAME =
            "cnm.workgroup_dim_order";
        constexpr static StringRef WORKGROUP_DIM_ORDER_INDEX_NAME =
            "cnm.workgroup_dim_order_index";

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


#ifndef CINM_BASE_ATTRIBUTES
#define CINM_BASE_ATTRIBUTES

include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

def Cinm_LevelDefAttr: Cinm_Attr<"CinmLevelDef", [OpAsmAttrInterface]> {
  let assemblyFormat = "`<` struct(params) `>`";
  let summary = "Definition of a level of a memory hierarchy";
  let description = [{
    This attributes describes the properties of a level of the memory hierarchy
    of a given platform. Its properties for now are:
    - a name;
    - a size, which represents the size of the level on any specific tile of the platform;
    - an alignment (in bytes), which asserts that buffer allocation and transfers must use this alignment (but not the individual elements within a buffer);
  }];
  let mnemonic = "level";
  let parameters = (ins
    "StringAttr":$name,
    "int64_t":$size_in_bytes,
    "int64_t":$alignment
  );
  let extraClassDeclaration = [{

    int64_t getSizeInElements(Type t) const {
      return getSizeInBytes() * 8 / t.getIntOrFloatBitWidth();
    }

    //===------------------------------------------------------------------===//
    // OpAsmAttrInterface Methods
    //===------------------------------------------------------------------===//

    /// Get a name to use when generating an alias for this attribute.
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const {
      os << getName().getValue();
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];
}

def Cinm_LevelArrayAttr : ArrayOfAttr<Cinm_Dialect, "CinmLevelArray", "level_array",
                            Cinm_LevelDefAttr.cppClassName> {}

//===----------------------------------------------------------------------===//
// Cost-Model
//===----------------------------------------------------------------------===//

def Cinm_CostModelData : Cinm_Attr<"CostModelData"> {
    let parameters = (ins
        "StringAttr": $name,
        "FloatAttr": $cost
    );

    let mnemonic = "cost_model_data";
    let assemblyFormat = "`<` struct($name, $cost) `>`";
}

#endif

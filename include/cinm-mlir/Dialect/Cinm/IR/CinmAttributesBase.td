
#ifndef CINM_BASE_ATTRIBUTES
#define CINM_BASE_ATTRIBUTES

include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

def Cinm_VarDefAttr: Cinm_Attr<"CinmVarDef", []> {
  let mnemonic = "vardef";
  let description = [{
    A variable declaration in an accelerator descriptor. This represents a bounded design parameter.
    Such an attribute defines two variables:
    - A _bound variable_, eg `S`, that represent the size of a dimension.
    - An _index variable_, eg `s`, that represents a specific index within the range 0..S.

    Index variables are given special treatment in the context of memory space handling. Only
    index variables may appear in the indexing tuple of a MemSpaceAttr. Index variables are 
    expected to be bound to the iteration variables of a tiled loop.
  }];
  let parameters = (ins 
    "StringAttr":$indexVarName,
    "StringAttr":$boundVarName,
    "long":$lower_bound_inclusive,
    "long":$upper_bound_inclusive
  );
  let builders = [
    AttrBuilderWithInferredContext<
      (ins 
        "StringAttr":$indexVarName,
        "StringAttr":$boundVarName,
        "long":$lower,
        "long":$upper
      ), [{
        return Base::get(indexVarName.getContext(), indexVarName, boundVarName, lower, upper);
      }]
    >
  ];
  let extraClassDeclaration = [{
    // /// Return the bound variable as a SymExpr.
    // SymVariableExpr getBoundVar() const {
    //   return getSymVariableExpr(getBoundVarName());
    // }

    // /// Return the index variable as a SymExpr.
    // SymVariableExpr getIndexVar() const {
    //   return getSymVariableExpr(getIndexVarName());
    // }

    /// Set the bounds of the bound variable.
    CinmVarDefAttr withBounds(long lowerInclusive, long upperInclusive) const {
      return get(getContext(), getIndexVarName(), getBoundVarName(), lowerInclusive, upperInclusive);
    }

    /// Set a value of the bound variable.
    CinmVarDefAttr withValue(long value) const {
      return withBounds(value, value);
    }

    std::optional<long> asInt() const {
      if (getLowerBoundInclusive() == getUpperBoundInclusive())
        return getLowerBoundInclusive();
      return std::nullopt;
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def Cinm_LevelDefAttr: Cinm_Attr<"CinmLevelDef", [OpAsmAttrInterface]> {
  let assemblyFormat = "`<` struct(params) `>`";
  let summary = "Definition of a level of a memory hierarchy";
  let description = [{
    This attributes describes the properties of a level of the memory hierarchy
    of a given platform. Its properties for now are:
    - a name, which is used by the CinmMemSpaceAttr;
    - a size, which represents the size of the level on any specific tile of the platform;
    - an alignment, which asserts that buffer allocation and transfers must use this alignment
    (but not the individual elements within a buffer);
    - an indexing specification.

    The indexing strategy is a way to define parametric levels. A grid platform
    that has a 2D structure may define a level with a 2D indexing scheme, to designate
    the level of a specific tile. These coordinates must be specified explicitly in 
    a given instance of a CinmMemSpace. 
    
    For instance upmem is a 2D grid platform of (ranks * dpu). Each tile defines two memory
    levels that are not shared between tiles. Each of those levels therefore has a 2D indexing
    scheme. If the platform supported a memory level shared between all the DPUs in a rank, that would
    likely have a 1D indexing scheme, that only uses the rank index variable of the accelerator.
  }];
  let mnemonic = "level";
  let parameters = (ins 
    "StringAttr":$name,
    "int64_t":$size_in_bytes,
    "int64_t":$alignment,
    "uint32_t":$arity
  );
  let extraClassDeclaration = [{

    // tilefirst::CinmMemSpaceAttr toMemSpace(ArrayRef<SymExpr> indices = {}) const {
    //   return tilefirst::CinmMemSpaceAttr::get(
    //       getName(),
    //       ShapeExprAttr::get(tilefirst::getShapeTupleExpr(indices, getContext())));
    // }

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

def Cinm_VarDefArrayAttr : ArrayOfAttr<Cinm_Dialect, "CinmVarDefArray", "vardef_array",
                            Cinm_VarDefAttr.cppClassName> {}

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

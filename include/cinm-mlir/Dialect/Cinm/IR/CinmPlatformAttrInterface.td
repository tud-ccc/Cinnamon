
#ifndef CINM_PLATFORM_ATTR_INTERFACE
#define CINM_PLATFORM_ATTR_INTERFACE

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"

def CinmPlatformAttrInterface : AttrInterface<"CinmPlatformAttrInterface", []> {
  let description = [{
    Interface for attributes that can be used as a platform descriptor for Cinm scheduling.
    A platform description defines levels of the memory hierarchy, and design parameters
    that should be solved during scheduling.

    Available platforms are expected to be defined on a function containing the code to schedule.
  }];

  let cppNamespace = "::mlir::cinm";

  let methods = [
    InterfaceMethod<
      /*desc=*/"A readable name for the platform.",
      /*retTy=*/"llvm::StringRef",
      /*methodName=*/"getName",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Return the levels of the memory hierarchy for this platform.",
      /*retTy=*/"::llvm::ArrayRef<::mlir::cinm::CinmLevelDefAttr>",
      /*methodName=*/"getLevels",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Get the level descriptor that has the given name. The result may be falsy if the level does not exist.",
      /*retTy=*/"::mlir::cinm::CinmLevelDefAttr",
      /*methodName=*/"getLevel",
      /*args=*/(ins "::llvm::StringRef":$name),
      [{
        for (auto level : $_self.getLevels()) {
          if (level.getName() == name) return level;
        }
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Translate a cinm memspace to memref memspace (could be string or int or whatever).",
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getMemrefMemspace",
      /*args=*/(ins "::mlir::cinm::CinmLevelDefAttr":$level),
      "",
      /*default=*/"return {};"
    >,

    InterfaceMethod<
      /*desc=*/"Return true if this platform wants to handle the given op. This is a hook for the --cinm-assign-platforms pass.",
      /*retTy=*/"bool",
      /*methodName=*/"wantsToHandle",
      /*args=*/(ins "::mlir::Operation*":$op),
      "",
      /*default=*/"return false;"
    >

  ];
  let extraClassDeclaration = [{

    /// Get a name to use when generating an alias for this attribute.
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const {
      os << getName();
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }

  }];
}

def CinmAcceleratorAttrInterface : AttrInterface<"CinmAcceleratorAttrInterface", []> {
  let description = [{
    Interface for attributes that can be used as an accelerator declaration for TF scheduling.
    An accelerator is an instance, or subset, or a platform. For instance, a grid-like platform
    may define the shape and size of the grid. An accelerator represents a specific set of tiles
    on that grid.
  }];

  let cppNamespace = "::mlir::cinm";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the platform descriptor of this accelerator.",
      /*retTy=*/"::mlir::cinm::CinmPlatformAttrInterface",
      /*methodName=*/"getPlatform",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      // note: you can use cinm::VarDefArrayAttr to implement this one.
      /*desc=*/"Return the design parameters of this accelerator.",
      /*retTy=*/"::llvm::ArrayRef<::mlir::cinm::CinmVarDefAttr>",
      /*methodName=*/"getDesignParams",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Return a new accelerator descriptor of the same type with new design parameter values.",
      /*retTy=*/"::mlir::cinm::CinmAcceleratorAttrInterface",
      /*methodName=*/"instantiateDesignParams",
      /*args=*/(ins "const llvm::MapVector<StringRef, long>&":$instantiations)
    >,

    InterfaceMethod<
      /*desc=*/"Compute tiling factors for the given op based on this accelerator's hardware parameters. Factors are appended to tilingFactors. Returns silenceableFailure if sizes cannot be determined automatically.",
      /*retTy=*/"::mlir::DiagnosedSilenceableFailure",
      /*methodName=*/"computeTilingFactors",
      /*args=*/(ins "::mlir::Operation*":$op, "::llvm::SmallVectorImpl<int64_t>&":$tilingFactors),
      "",
      [{
        return emitSilenceableFailure(op->getLoc()) << "unimplemented";
      }]
    >,

  ];

  let extraClassDeclaration = [{

    mlir::cinm::CinmVarDefAttr getBoundVar(StringRef name) const {
      for (auto v : getDesignParams()) {
        if (v.getBoundVarName() == name) return v;
      }
      return mlir::cinm::CinmVarDefAttr{};
    }

    mlir::cinm::CinmVarDefAttr getIndexVar(StringRef name) const {
      for (auto v : getDesignParams()) {
        if (v.getIndexVarName() == name) return v;
      }
      return mlir::cinm::CinmVarDefAttr{};
    }

    // /// Return whether a given memory space belongs to this accelerator.
    // bool ownsMemSpace(::mlir::cinm::CinmMemSpaceAttr space) const {
    //   if (space.isAbsentMemspace() || space.isHostMemspace()) 
    //       return false;

    //   auto level = getPlatform().getLevel(space.getName());
    //   if (!level)
    //     return false;
    
    //   return space == getGenericMemSpace(level);
    // }

    // /// Return whether all the operands of the given op (minus those of unknown memspace) are buffers belonging to this accelerator.
    // bool ownsOperation(::mlir::Operation* op) const {
    //   for (auto operand : op->getOperands()) {
    //     if (auto buf = llvm::dyn_cast_or_null<::mlir::cinm::CinmBufferType>(operand.getType())) {
    //       auto space = buf.getMemorySpace();
    //       if (space.isAbsentMemspace())
    //         continue;
    //       if (!ownsMemSpace(space))
    //         return false;
    //     }
    //   }
    //   return true;
    // }
  }];
}


#endif

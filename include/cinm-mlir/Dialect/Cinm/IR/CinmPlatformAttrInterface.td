
#ifndef CINM_PLATFORM_ATTR_INTERFACE
#define CINM_PLATFORM_ATTR_INTERFACE

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"

def CinmLevelAttrInterface : AttrInterface<"CinmLevelAttrInterface", []> {
  let description = [{
    Interface for attributes that identify a level of a platform's memory
    hierarchy. Such an attribute is what a `cnm.buffer` carries as its level,
    and what the memrefs of that level carry as their memory space, so it must
    be usable as a memref memory space: in particular it must not be
    parameterized by anything that varies between instances of the same
    platform (a level's capacity, for instance, differs between UPMEM v1A and
    v1B, and baking it in would make `memref<64xi32, wram>` two distinct types
    on the two boards).

    Consequently this interface carries *identity only*. The capacity,
    alignment and other quantitative properties of the level live in the
    platform's `CinmLevelDefAttr` for that name, reachable with
    `platform.getLevel(attr.getLevelName())`.
  }];

  let cppNamespace = "::mlir::cinm";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        The name of the level this attribute denotes. Must match the name of a
        `CinmLevelDefAttr` in the platform this level belongs to.
      }],
      /*retTy=*/"llvm::StringRef",
      /*methodName=*/"getLevelName",
      /*args=*/(ins)
    >
  ];
}

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
      /*desc=*/[{
        Translate a level descriptor of this platform into the attribute used
        as the memref memory space (and as the `cnm.buffer` level) for that
        level. Falsy if this platform has no such level.
      }],
      /*retTy=*/"::mlir::cinm::CinmLevelAttrInterface",
      /*methodName=*/"getMemrefMemspace",
      /*args=*/(ins "::mlir::cinm::CinmLevelDefAttr":$level),
      "",
      /*default=*/"return {};"
    >,
    InterfaceMethod<
      /*desc=*/[{
        The inverse of `getMemrefMemspace`: recover the level descriptor a
        memref memory space (or `cnm.buffer` level) refers to. Falsy if the
        attribute does not denote a level of this platform.
      }],
      /*retTy=*/"::mlir::cinm::CinmLevelDefAttr",
      /*methodName=*/"getLevelOfMemspace",
      /*args=*/(ins "::mlir::Attribute":$memspace),
      [{
        auto space = ::llvm::dyn_cast_or_null<::mlir::cinm::CinmLevelAttrInterface>(memspace);
        if (!space)
          return {};
        for (auto level : $_self.getLevels()) {
          if (level.getName() == space.getLevelName()) return level;
        }
        return {};
      }]
    >,

    InterfaceMethod<
      /*desc=*/"Return true if this platform wants to handle the given op. This is a hook for the --cinm-assign-platforms pass.",
      /*retTy=*/"bool",
      /*methodName=*/"isOffloadingTarget",
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

  }];
}


#endif

//===- Ops.td - Btfl dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Btfl dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEMX_PLATFORM
#define UPMEMX_PLATFORM

include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td"
// include "cinm-mlir/Dialect/Btfl/IR/BtflAttributes.td"

include "UPMEMBase.td"

def UpmemPlatformAttr: UPMEM_Attr<"UpmemPlatform", [
    OpAsmAttrInterface,CinmPlatformAttrInterface
  ]> {
  
  let summary = "Platform description for upmem";
  let mnemonic = "platform";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins 
    "::mlir::cinm::CinmLevelArrayAttr":$levels,
    "bool":$isV1a,
    "int":$maxNumRanks,
    "int":$maxNumDpusPerRank,
    "int":$maxNumTasklets
  );

  let extraClassDeclaration = [{

    static UpmemPlatformAttr getDefault(MLIRContext* ctx);

    // void collectSchedulingStartingPoints(::mlir::cinm::CinmSchedulingStarter &driver) const;

    Attribute getMemrefMemspace(mlir::cinm::CinmLevelDefAttr level) const;

    ::llvm::StringRef getName() const {
      return "upmem";
    }
   
    ::mlir::cinm::CinmLevelDefAttr getMramLevel() const {
      return getLevels()[0];
    }

    ::mlir::cinm::CinmLevelDefAttr getWramLevel() const {
      return getLevels()[1];
    }

    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const {
      os << getName();
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];

}


def UpmemAcceleratorAttr: UPMEM_Attr<"UpmemAccelerator", [
          DeclareAttrInterfaceMethods<CinmAcceleratorAttrInterface, ["getThreadCountVarDef"]>,
          // DeclareAttrInterfaceMethods<BtflAcceleratorAttrInterfaceExtension, ["lowerAcceleratorOp", "lowerOpTopDown"]>
    ]> {
  let summary = "Accelerator description for upmem";
  let hasCustomAssemblyFormat = 1;
  let mnemonic = "array";

  // let skipDefaultBuilders = 1;
  let parameters = (ins 
    "UpmemPlatformAttr":$platform,
    "::mlir::cinm::CinmVarDefArrayAttr":$designParams
  );
  let builders=[
    AttrBuilderWithInferredContext<(ins
      "::mlir::cinm::CinmVarDefAttr":$ranks,
      "::mlir::cinm::CinmVarDefAttr":$dpus,
      "::mlir::cinm::CinmVarDefAttr":$tasklets
    ), [{
      auto platformDef = UpmemPlatformAttr::getDefault(ranks.getContext());
      return get(std::move(platformDef), ranks, dpus, tasklets);
    }]>,

    AttrBuilderWithInferredContext<(ins
      "UpmemPlatformAttr":$platform,
      "::mlir::cinm::CinmVarDefAttr":$ranks,
      "::mlir::cinm::CinmVarDefAttr":$dpus,
      "::mlir::cinm::CinmVarDefAttr":$tasklets
    ), [{
      auto designParms = mlir::cinm::CinmVarDefArrayAttr::get(platform.getContext(), {ranks, dpus, tasklets});
      return Base::get(platform.getContext(), platform, std::move(designParms));
    }]>
  ];

  let extraClassDeclaration = [{
   
    ::mlir::cinm::CinmLevelDefAttr getMramLevel() const {
      return getPlatform().getMramLevel();
    }
    ::mlir::cinm::CinmLevelDefAttr getWramLevel() const {
      return getPlatform().getWramLevel();
    }

    ::mlir::cinm::CinmVarDefAttr getNumRanks() const {
      return getDesignParams().getValue()[0];
    }
    ::mlir::cinm::CinmVarDefAttr getNumDpusPerRank() const {
      return getDesignParams().getValue()[1];
    }
    ::mlir::cinm::CinmVarDefAttr getNumTaskletsPerDpu() const {
      return getDesignParams().getValue()[2];
    }
  }];

}


// todo scatter/gather loop scheduler

// def UpmemTaskletSchedulerAttr: Upmemx_Attr<"UpmemTaskletScheduler", [DeclareAttrInterfaceMethods<BtflLoopSchedulerAttrInterface>]> {
//   let summary = "upmem tasklets parallel scheduler";
//   let assemblyFormat = "";
//   let mnemonic = "tasklets";

//   let extraClassDeclaration = [{

//   }];

// }


#endif

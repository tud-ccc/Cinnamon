//===- Ops.td - Btfl dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Btfl dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEMX_PLATFORM
#define UPMEMX_PLATFORM

include "tilefirst-mlir/Dialect/TileFirst/IR/TileFirstPlatformAttrInterface.td"
include "tilefirst-mlir/Dialect/Btfl/IR/BtflAttributes.td"

include "UPMEMBase.td"

def UpmemPlatformAttr: Upmem_Attr<"UpmemPlatform", [
    OpAsmAttrInterface,TfPlatformAttrInterface
  ]> {
  
  let summary = "Platform description for upmem";
  let assemblyFormat = "`<` `levels` `=` $levels `,` `dimensions` `=` `(` $maxNumRanks `x` $maxNumDpusPerRank `x` $maxNumTasklets `)` `>`";
  let mnemonic = "platform";
  let parameters = (ins 
    "::mlir::tilefirst::TfLevelArrayAttr":$levels,
    "int":$maxNumRanks,
    "int":$maxNumDpusPerRank,
    "int":$maxNumTasklets
  );

  let extraClassDeclaration = [{

    static UpmemPlatformAttr getDefault(MLIRContext* ctx);

    void collectSchedulingStartingPoints(::mlir::tilefirst::TfSchedulingStarter &driver) const;

    Attribute getMemrefMemspace(mlir::tilefirst::TfLevelDefAttr level) const;

    ::llvm::StringRef getName() const {
      return "upmem";
    }
   
    ::mlir::tilefirst::TfLevelDefAttr getMramLevel() const {
      return getLevels()[0];
    }

    ::mlir::tilefirst::TfLevelDefAttr getWramLevel() const {
      return getLevels()[1];
    }

    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const {
      os << getName();
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];

}

def UpmemAcceleratorAttr: Upmem_Attr<"UpmemAccelerator", [
          DeclareAttrInterfaceMethods<TfAcceleratorAttrInterface, ["getThreadCountVarDef"]>,
          DeclareAttrInterfaceMethods<BtflAcceleratorAttrInterfaceExtension>
    ]> {
  let summary = "Accelerator description for upmem";
  let hasCustomAssemblyFormat = 1;
  let mnemonic = "array";

  // let skipDefaultBuilders = 1;
  let parameters = (ins 
    "UpmemPlatformAttr":$platform,
    "::mlir::tilefirst::TfVarDefArrayAttr":$designParams
  );
  let builders=[
    AttrBuilderWithInferredContext<(ins
      "::mlir::tilefirst::TfVarDefAttr":$ranks,
      "::mlir::tilefirst::TfVarDefAttr":$dpus,
      "::mlir::tilefirst::TfVarDefAttr":$tasklets
    ), [{
      auto platformDef = UpmemPlatformAttr::getDefault(ranks.getContext());
      return get(std::move(platformDef), ranks, dpus, tasklets);
    }]>,

    AttrBuilderWithInferredContext<(ins
      "UpmemPlatformAttr":$platform,
      "::mlir::tilefirst::TfVarDefAttr":$ranks,
      "::mlir::tilefirst::TfVarDefAttr":$dpus,
      "::mlir::tilefirst::TfVarDefAttr":$tasklets
    ), [{
      auto designParms = mlir::tilefirst::TfVarDefArrayAttr::get(platform.getContext(), {ranks, dpus, tasklets});
      return Base::get(platform.getContext(), platform, std::move(designParms));
    }]>
  ];

  let extraClassDeclaration = [{
   
    ::mlir::tilefirst::TfLevelDefAttr getMramLevel() const {
      return getPlatform().getMramLevel();
    }
    ::mlir::tilefirst::TfLevelDefAttr getWramLevel() const {
      return getPlatform().getWramLevel();
    }

    ::mlir::tilefirst::TfVarDefAttr getNumRanks() const {
      return getDesignParams().getValue()[0];
    }
    ::mlir::tilefirst::TfVarDefAttr getNumDpusPerRank() const {
      return getDesignParams().getValue()[1];
    }
    ::mlir::tilefirst::TfVarDefAttr getNumTaskletsPerDpu() const {
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

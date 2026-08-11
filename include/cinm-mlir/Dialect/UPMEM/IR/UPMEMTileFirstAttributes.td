//===- Ops.td - Btfl dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Btfl dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEMX_PLATFORM
#define UPMEMX_PLATFORM

include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.td"
// include "cinm-mlir/Dialect/Btfl/IR/BtflAttributes.td"

include "UPMEMBase.td"

def UpmemPlatformAttr: UPMEM_Attr<"UpmemPlatform", [
    OpAsmAttrInterface,
    DeclareAttrInterfaceMethods<CinmPlatformAttrInterface, ["isOffloadingTarget"]>
  ]> {

  let summary = "Platform description for upmem";
  let description = [{
    The device as the allocator sees it: a pool of `maxDpus` DPUs running up
    to `maxTasklets` tasklets each. Ranks are not a dimension of anything --
    the SDK hands out DPU counts and places them on ranks opaquely (some
    ranks even have defective DPUs), and transfer cost is a function of the
    DPU count alone.
  }];
  let mnemonic = "platform";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins
    "::mlir::cinm::CinmLevelArrayAttr":$levels,
    "bool":$isV1a,
    "int":$maxDpus,
    "int":$maxNumTasklets
  );

  let extraClassDeclaration = [{

    static UpmemPlatformAttr getDefault(MLIRContext* ctx);

    // void collectSchedulingStartingPoints(::mlir::cinm::CinmSchedulingStarter &driver) const;

    ::mlir::cinm::CinmLevelAttrInterface getMemrefMemspace(mlir::cinm::CinmLevelDefAttr level) const;

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
          DeclareAttrInterfaceMethods<CnmAcceleratorAttrInterface, [ "getWorkgroupMemoryLevels", "getWorkgroupShape", "bufferSizeOfLeaf"]>,
          DeclareAttrInterfaceMethods<CinmAcceleratorAttrInterface, ["computeTilingFactors", "getDesignParams", "instantiateDesignParams"]>,
          OpAsmAttrInterface
    ]> {
  let summary = "Accelerator description for upmem";
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let mnemonic = "array";

  // let skipDefaultBuilders = 1;
  let parameters = (ins
    "UpmemPlatformAttr":$platform,
    ArrayRefParameter<"int64_t">:$workgroupShape
    // "::mlir::cinm::CinmVarDefArrayAttr":$designParams,
  );
  let builders=[
    AttrBuilder<(ins
      "int64_t":$dpus,
      "int64_t":$tasklets
    ), [{
      auto platformDef = UpmemPlatformAttr::getDefault($_ctxt);
      return get(std::move(platformDef), dpus, tasklets);
    }]>,

    AttrBuilderWithInferredContext<(ins
      "UpmemPlatformAttr":$platform,
      "int64_t":$dpus,
      "int64_t":$tasklets
    ), [{
      return get(platform.getContext(), platform, {dpus, tasklets});
    }]>
  ];

  let extraClassDeclaration = [{

    ::mlir::cinm::CinmLevelDefAttr getMramLevel() const {
      return getPlatform().getMramLevel();
    }
    ::mlir::cinm::CinmLevelDefAttr getWramLevel() const {
      return getPlatform().getWramLevel();
    }

    int64_t getNumDpus() const {
      return getWorkgroupShape()[0];
    }
    int64_t getNumTaskletsPerDpu() const {
      return getWorkgroupShape()[1];
    }
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "upmem_" << getNumDpus() << "_" << getNumTaskletsPerDpu();
      return ::mlir::OpAsmAliasResult::OverridableAlias;
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

//===- Passes.td - Cnm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_TRANSFORM_PASSES
#define CNM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def CnmSPIRVAttachTargetAttributePass: Pass<"cnm-spirv-attach-target-attribute", ""> {
  let summary = "Attaches an SPIR-V target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules and attaches an SPIR-V target attribute.
  }];

  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"spirvVersion", "ver", "std::string",
           /*default=*/ "\"v1.0\"",
           "SPIR-V Version.">,
    ListOption<"spirvCapabilities", "caps", "std::string",
           "List of supported SPIR-V Capabilities">,
    ListOption<"spirvExtensions", "exts", "std::string",
           "List of supported SPIR-V Extensions">,
    Option<"clientApi", "client_api", "std::string",
           /*default=*/ "\"Unknown\"",
           "Client API">,
    Option<"deviceVendor", "vendor", "std::string",
           /*default=*/ "\"Unknown\"",
           "Device Vendor">,
    Option<"deviceType", "device_type", "std::string",
           /*default=*/ "\"Unknown\"",
           "Device Type">,
    Option<"deviceId", "device_id", "uint32_t",
           /*default=*/ "",
           "Device ID">,
  ];
}

def CnmSPIRVAttachKernelEntryPointAttributePass: Pass<"cnm-spirv-attach-kernel-entry-point-attribute", ""> {
  let summary = "Attaches an SPIR-V entry point attribute to a GPU kernel.";
  let description = [{
    This pass searches for all GPU Kernels and attaches an SPIR-V entry_point_abi attribute.
  }];

  let options = [
    Option<"kernelMatcher", "kernel", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the kernels to attach the attribute to.">,
  ];
}


def CnmBufferizePass : Pass<"cnm-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the `cnm` dialect";
  let description = [{

  }];
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "memref::MemRefDialect"];
}


def CnmHoistWorkgroupsPass : Pass<"cnm-hoist-workgroups", "func::FuncOp"> {
  let summary = "Hoist worgroup creation outside of loops";
  let description = [{

  }];
  let dependentDialects = [];
}


#endif // CNM_TRANSFORM_PASSES

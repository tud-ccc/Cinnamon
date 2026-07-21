//===- Passes.td - Cnm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_TRANSFORM_PASSES
#define CNM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def CnmSPIRVAttachTargetAttributePass: Pass<"cnm-spirv-attach-target-attribute"> {
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

def CnmSPIRVAttachKernelEntryPointAttributePass: Pass<"cnm-spirv-attach-kernel-entry-point-attribute"> {
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

def CnmHoistWorkgroupsPass : Pass<"cnm-hoist-workgroups"> {
  let summary = "Hoist worgroup creation outside of loops";
  let description = [{

  }];
  let dependentDialects = [];
}

def CnmEnsureScatterGatherContiguousPass : Pass<"cnm-ensure-scatter-gather-contiguous"> {
  let summary = "Ensure cnm.scatter/cnm.gather transfer contiguous host memrefs";
  let description = [{
    Lowering a `cnm.scatter`/`cnm.gather` to `upmem.scatter`/`upmem.gather`
    performs a single flat memcpy per DPU, computed from a start offset and an
    element count. This is only correct if the transferred elements are
    actually contiguous in the host memref. This pass detects host memrefs
    that are not contiguous (e.g. because they are subviews of a larger
    tensor) and inserts an intermediate contiguous buffer: a `memref.alloc`
    plus a `memref.copy` before a scatter (or after a gather), so that the
    memref handed to the op is always safe to transfer with a flat memcpy.
  }];
  let dependentDialects = ["memref::MemRefDialect"];
}

#endif // CNM_TRANSFORM_PASSES

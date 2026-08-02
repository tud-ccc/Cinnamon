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

def CnmScatterOptimizationsPass : Pass<"cnm-scatter-optimizations"> {
  let summary = "Shrink cnm.scatter transfers that carry redundant data";
  let description = [{
    A `cnm.scatter` names, for every workgroup element, the region of the host
    buffer that element receives. When several workgroup elements provably
    receive the same bytes, the host side of the transfer holds more data than
    the workgroup can distinguish, and it can be replaced by a smaller one.

    Currently implemented: *uniform sources*. If every element of the scattered
    value holds the same constant -- it is a splat `arith.constant`, a
    `linalg.fill` of a constant, or a `memref.get_global` of a constant global
    with a splat initializer -- then the scatter map is irrelevant, since every
    workgroup element receives that same constant whichever region it reads.
    The scatter is rewritten to broadcast a single buffer-shaped tile:

    ```mlir
    %big = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x512x1x8xi32>) -> ...
    cnm.scatter %big into %buf[(d0, d1, d2) -> (d1 floordiv 64, ...)] of %wg
      : tensor<32x512x1x8xi32> into !cnm.buffer<1x8xi32 on ...>
    // becomes
    %tile = arith.constant dense<0> : tensor<1x8xi32>
    cnm.scatter %tile into %buf[(d0, d1, d2) -> ()] of %wg
      : tensor<1x8xi32> into !cnm.buffer<1x8xi32 on ...>
    ```

    This is a pure CNM-to-CNM rewrite: it says nothing about how the backend
    realizes the transfer, only that the data it has to move shrinks from the
    whole workgroup's worth to one tile.

    Also implemented: *device-side initialization*. A uniform source is a
    constant every leaf could write for itself, so when the buffer feeds a
    single `cnm.launch` the transfer is dropped altogether and a `linalg.fill`
    of that constant is prepended to the launch body:

    ```mlir
    cnm.scatter %tile into %buf[(d0, d1, d2) -> ()] of %wg
    cnm.launch %wg outs(%arg0 = %buf : <1x8xi32, #upmem.mram>) { ... }
    // becomes
    cnm.launch %wg outs(%arg0 = %buf : <1x8xi32, #upmem.mram>) {
      %c0 = arith.constant 0 : i32
      linalg.fill ins(%c0 : i32) outs(%arg0 : memref<1x8xi32, #upmem.mram>)
      ...
    }
    ```

    The leaves write in parallel and no host bandwidth is spent. The fill is
    emitted on the launch parameter, wherever that lives, so the rewrite stands
    on its own; a backend that stages the parameter into a faster level can
    additionally fold the fill into the staging buffer and stop loading a
    buffer it is about to overwrite, which is what `--upmem-tile-mram-buffers`
    does.

    The rewrite is dropped when the buffer reaches more than one launch, when
    the scatter and the launch are in different blocks -- a scatter hoisted out
    of a loop around the launch seeds an accumulator once, and re-seeding it
    per launch is a different program -- or when anything reads the buffer in
    between.

    The recognition is purely SSA-based, so a memref that some earlier op
    filled in place is *not* matched (that would need alias and effect
    analysis). Run the pass before bufferization to catch those.
  }];
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect",
                           "memref::MemRefDialect"];
}

#endif // CNM_TRANSFORM_PASSES

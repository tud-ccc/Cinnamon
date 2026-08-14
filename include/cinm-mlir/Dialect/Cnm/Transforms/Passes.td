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

def CnmFuseLaunchesPass : Pass<"cnm-fuse-launches"> {
  let summary = "Drop a gather/scatter round trip and merge the launches";
  let description = [{
    When a distributed op's consumer is distributed the same way, the value
    leaves the workgroup and comes straight back to the leaf it left: a
    `cnm.gather` whose result feeds a `cnm.scatter` with the same map into a
    buffer of the same type. Both transfers are redundant, so the consumer is
    pointed at the producer's buffer and they are dropped; the two launches
    then run on one workgroup and are merged into one.

    Whether the two schedules line up that way is decided by whoever chose the
    tile sizes -- see docs/LaunchFusionDesign.md. This pass only recognises
    that they did, syntactically, and leaves everything else alone. In
    particular a producer whose reduction was split across the workgroup
    gathers partial results and merges them on the host, so its map and buffer
    shape do not match the consumer's and nothing here fires.

    ```mlir
    cnm.launch %wg ins(...) outs(%p : <64xi32>) { ... }
    %h = cnm.gather %p[#m] of %wg into %e : <64xi32> into tensor<1024xi32>
    cnm.scatter %h into %c[#m] of %wg : tensor<1024xi32> into <64xi32>
    cnm.launch %wg ins(%c : <64xi32>) outs(...) { ... }
    // becomes one launch reading %p directly
    ```
  }];
  let options = [
    Option<"mergeLaunchBodies", "merge-launch-bodies", "bool", /*default=*/"true",
           "Also splice the two launch bodies into a single launch. Turning "
           "this off keeps the transfers elided but leaves two kernels, which "
           "is useful for isolating the two effects when measuring.">,
  ];
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
    plus a `cnm.compact_buffer` before a scatter (or a `memref.copy` after a
    gather), so that the memref handed to the op is always safe to transfer
    with a flat memcpy.

    Whether packing is the right answer is a cost question, not a correctness
    one -- `upmem.scatter_blocks` can transfer a non-contiguous host buffer as
    several blocks per DPU instead. Packing wins when the operand is
    `cinm.static`, because then the repack happens once per workload lifetime
    and its cost amortizes away; on a per-inference operand the several-blocks
    transfer is usually cheaper than paying for a copy every call. See
    `static-only`.
  }];
  let options = [
    Option<"packFragmented", "pack-fragmented", "bool", "false",
           "Also pack an operand whose transfer is contiguous but arrives as "
           "several blocks per leaf, so that it becomes one whole-buffer block "
           "instead. The repack reorders the host value into workgroup x "
           "buffer order, which is what makes a leaf's share a single run.\n\n"
           "Whether that is worth it is a cost question: many small transfers "
           "per leaf against one repack. It is unconditionally right for a "
           "backend that has no several-blocks transfer at all -- CINM 1's "
           "code generation had only the whole-array form -- which is what "
           "this option reproduces.">,
    Option<"staticOnly", "static-only", "bool", "false",
           "Only pack operands whose data is the same on every inference "
           "(cinm.static), leaving the rest for the backend to transfer as "
           "several blocks per DPU. Off by default, which packs whatever is "
           "not contiguous -- always correct, and what callers that do not "
           "reason about cost should want.">,
  ];
  let dependentDialects = ["memref::MemRefDialect", "cnm::CnmDialect"];
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

def CnmIsolateLinalgCapturesPass
    : Pass<"cnm-isolate-linalg-captures"> {
  let summary = "Turn a linalg op's captured scalars into explicit operands";
  let description = [{
    `cnm.launch`'s body is `IsolatedFromAbove`, and `--convert-linalg-to-cnm`
    clones a distributed op's region into it verbatim -- so an op that reads a
    value defined outside itself cannot be distributed at all, silently: it
    has no tile sizes to reject, so it is simply skipped, and the enclosing
    compute block falls back to a host loop nest with no diagnostic.

    Such a capture is not exotic. Upstream's `tensor.splat` lowers to a
    `linalg.generic` with no `ins` at all -- the splatted scalar stays a
    captured value inside the body rather than becoming an operand, since
    nothing about a lone splat needs it to be one. Fusing that generic into a
    consumer (`--linalg-fuse-elementwise-ops`) carries the capture into the
    fused body unchanged, because upstream fusion has no reason to expect
    `IsolatedFromAbove` downstream.

    This pass closes exactly that gap. For each capture -- a value used inside
    the op's region but defined outside it -- it materializes a rank-0 tensor
    (`tensor.from_elements`), appends it as a new `ins` operand with a
    zero-result indexing map, and replaces the capture inside the region with
    the new block argument:

    ```mlir
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<1024xi32>) outs(%e : tensor<1024xi32>) {
    ^bb0(%in: i32, %out: i32):
      %r = arith.muli %in, %c : i32   // %c defined outside
      linalg.yield %r : i32
    }
    // becomes
    %c_t = tensor.from_elements %c : tensor<i32>
    linalg.generic {indexing_maps = [#map, #map0, #map], iterator_types = ["parallel"]}
      ins(%x, %c_t : tensor<1024xi32>, tensor<i32>) outs(%e : tensor<1024xi32>) {
    ^bb0(%in: i32, %c_arg: i32, %out: i32):
      %r = arith.muli %in, %c_arg : i32
      linalg.yield %r : i32
    }
    ```

    A zero-result map ties the new operand to none of the loop dimensions, so
    `--convert-linalg-to-cnm` reads it exactly like any other operand: rank 0
    means a `cnm.buffer` of one element, and a scatter map with no results
    sends that one host element to every leaf -- the same shape
    `--cnm-scatter-optimizations` already produces when it shrinks a
    uniform-valued scatter to a broadcast.

    Only scalar (non-shaped) captures are handled; a captured tensor or memref
    is reported as an error rather than silently left in place, since
    upstream fusion is not expected to produce one and a silent miss here is
    exactly the failure mode this pass exists to close.
  }];
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"];
}

#endif // CNM_TRANSFORM_PASSES

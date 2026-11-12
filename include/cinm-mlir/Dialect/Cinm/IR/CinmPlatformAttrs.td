
#ifndef CINM_PLATFORM_ATTRS
#define CINM_PLATFORM_ATTRS

include "CinmBase.td"
include "CinmPlatformAttrInterface.td"

// A host rate or cost: an integer or a float literal, positive (the
// attribute's verifier checks that). Stored as an APFloat (in double
// semantics) because attribute storage hashes its parameters, and a double
// has no llvm::hash_value; getModel() reads them back as doubles.
class Cinm_HostParameter<string desc> : AttrParameter<"::llvm::APFloat", desc> {
  let parser = "::mlir::cinm::parseHostParameter($_parser)";
}

def CinmHostPlatformAttr: Cinm_Attr<"HostPlatform", [
    CinmPlatformAttrInterface
  ]> {
  let summary = "Host platform description";
  let mnemonic = "host_platform";
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let description = [{
    The host the accelerators hang off, and what it costs to run code on it.
    Two consumers read it: the offload roofline (--cinm-assign-platforms),
    which compares an op on an accelerator against the same op left here,
    and the cost model, which prices the host code around each offloaded
    block -- repacks of operands, and the loops that combine partial results.

    Every parameter is required, in any order, and printed, so that IR
    states the machine it is priced against rather than leaving it to
    whichever defaults the reader was built with:

    ```mlir
    #cinm.host_platform<ops_per_second = 2.1e12, dram_bytes_per_second = 21.7e9,
                        scalar_op_ns = 3.0, vector_op_ns = 0.5, vector_bytes = 64,
                        stream_bytes_per_second = 10.8e9,
                        copy_bytes_per_second = 0.63e9>
    ```

    A program rarely writes one: --cinm-attach-host-platform adds it to every
    function's `cinm.available_platforms` from its pass options, whose
    defaults are the bench machine and whose description says where each
    number comes from.

    The host description a piece of IR is priced against is the nearest one
    in a `cinm.available_platforms` list of the op or an enclosing op
    (`getInScope`). There is none when no list in scope names a host, and a
    pass that has host code to price then fails rather than guess.

    Parameters:

    - `ops_per_second` (roofline): arithmetic throughput summed over every
      core, a multiply-add counting as two ops, at the width the host code
      computes in: a host that widens int8 to i32 states its int32 rate.
    - `dram_bytes_per_second` (roofline): DRAM bandwidth to those cores, for
      the mix of two reads and a write this term's byte count implies.
    - `scalar_op_ns`: one scalar integer add in straight-line host code.
    - `vector_op_ns`, `vector_bytes`: one vector instruction, and the width
      it operates on. A host loop nest is priced as vectorized: an op on a
      `b`-byte element costs `vector_op_ns * b / vector_bytes`. The width is
      the one the host pipeline generates, not the one the machine could
      run.
    - `stream_bytes_per_second`: what one core streams through a host loop
      nest walked in contiguous order. A loop nest costs the larger of its
      arithmetic and its memory traffic at this rate.
    - `copy_bytes_per_second`: the rate of the repacks between a host
      buffer's layout and a transfer's (cnm.compact_buffer, its inverse
      cnm.expand_buffer, and memref.copy).
  }];

  let parameters = (ins
    Cinm_HostParameter<"arithmetic throughput, ops/s">:$ops_per_second,
    Cinm_HostParameter<"DRAM bandwidth, bytes/s">:$dram_bytes_per_second,
    Cinm_HostParameter<"scalar add, ns">:$scalar_op_ns,
    Cinm_HostParameter<"vector instruction, ns">:$vector_op_ns,
    Cinm_HostParameter<"vector width, bytes">:$vector_bytes,
    Cinm_HostParameter<"one core's streaming rate, bytes/s">:$stream_bytes_per_second,
    Cinm_HostParameter<"strided repack rate, bytes/s">:$copy_bytes_per_second
  );

  let builders = [
    AttrBuilder<(ins "const ::mlir::cinm::HostModel &":$model), [{
      auto f = [](double v) { return ::llvm::APFloat(v); };
      return $_get($_ctxt, f(model.opsPerSecond), f(model.dramBytesPerSecond),
                   f(model.scalarOpNs), f(model.vectorOpNs),
                   f(model.vectorBytes), f(model.streamBytesPerSecond),
                   f(model.copyBytesPerSecond));
    }]>
  ];

  let extraClassDeclaration = [{
    llvm::StringRef getName() const {
      return "host";
    }

    /// The parameters as one struct, which the roofline and the cost model
    /// take, so that both read the same values.
    ::mlir::cinm::HostModel getModel() const;
    llvm::ArrayRef<cinm::CinmLevelDefAttr> getLevels() const {
      return {};
    }

    /// The host description `op` is priced against: the nearest one listed
    /// in a `cinm.available_platforms` of `op` or an enclosing op, or null
    /// when none is. Unlike accelerator lookup, a list that
    /// names no host does not stop the search -- the host is the machine the
    /// whole program runs on, so a block that lists only its accelerator
    /// still runs on the host its function declared.
    static HostPlatformAttr getInScope(::mlir::Operation *op);
  }];

}


def CinmAvailablePlatformsAttr : ArrayOfAttr<Cinm_Dialect, "CinmPlatformArray", "platforms",
                            CinmPlatformAttrInterface.storageType> {}

#endif

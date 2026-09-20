
#ifndef CINM_PLATFORM_ATTRS
#define CINM_PLATFORM_ATTRS

include "CinmBase.td"
include "CinmPlatformAttrInterface.td"

def CinmHostPlatformAttr: Cinm_Attr<"HostPlatform", [
    CinmPlatformAttrInterface
  ]> {
  let summary = "Host platform description";
  let hasCustomAssemblyFormat = 1;
  let mnemonic = "host_platform";
  let description = [{
    The host the accelerators hang off, and what it costs to run code on it.
    Two consumers read it: the offload roofline (--cinm-assign-platforms),
    which compares an op on an accelerator against the same op left here,
    and the cost model, which prices the host code around each offloaded
    block -- repacks of operands, and the loops that combine partial results.

    Every parameter is optional and defaults to the bench machine, an Intel
    R2312WFTZSR holding 2x Xeon Silver 4216 (16 Cascade Lake cores each at
    2.1 GHz) alongside 20 PIM modules; `#cinm.host_platform` alone is that
    machine. Only the parameters that differ from it are printed:

    ```mlir
    #cinm.host_platform<ops_per_second = 8.6e12>
    ```

    The host description a piece of IR is priced against is the nearest one
    in a `cinm.available_platforms` list of the op or an enclosing op
    (`getInScope`), or the default when none is listed. Listing it is how a
    program states that it runs on a different machine.

    Parameters:

    - `ops_per_second` (roofline): arithmetic throughput summed over every
      core. It depends on the element type, by a factor of four: an int32
      multiply-add uses 16 AVX-512 lanes, while AVX512-VNNI's vpdpbusd does
      64 int8 multiply-accumulates in one instruction, so the machine
      sustains ~2.1e12 int32 ops/s but ~8.6e12 int8 ops/s. The default is
      the int32 figure; an int8 program has to state the other one.
    - `dram_bytes_per_second` (roofline): DRAM bandwidth to those cores. The
      PIM modules occupy DIMM slots: 4x 64 GB DDR4 populate 2 slots per
      socket, so 4 of the 12 channels carry host DRAM, and the 4216 caps
      DDR4 at 2400 MT/s -- 76.8 GB/s of peak against the ~230 GB/s the same
      sockets reach fully populated. The default is a STREAM triad
      (two reads and a write, the mix this term's byte count implies)
      measured under the performance governor with each thread reading
      pages it faulted itself: 23 GB/s, saturating from 16 threads on. A
      pure read reaches 30-32. The figure matters: the gate offloads `va`
      below 20 GB/s and rejects it above, which is where the 15 GB/s
      cpu_baseline.py quotes used to put it.
    - `scalar_op_ns`: one scalar integer add in straight-line host code.
      Between 1.6 and 10 ns on the bench machine, lower the more iterations
      of the enclosing loop amortize it.
    - `vector_op_ns`, `vector_bytes`: one vector instruction, and the width
      it operates on. A host loop nest is priced as vectorized: an op on a
      `b`-byte element costs `vector_op_ns * b / vector_bytes`. The width is
      the one the host pipeline generates, not the one the machine could
      run -- `--affine-super-vectorize` picks the element count and `llc`'s
      `-mcpu` decides what that legalises to (HOST_VECTOR_SIZE and
      HOST_MCPU in the experiments Makefile). At 16 elements on
      `cascadelake` an i32 loop emits zmm, so 64 bytes; the same pass at 8
      elements with llc's default `generic` emitted xmm, 16 bytes.
    - `stream_bytes_per_second`: what one core streams through a host loop
      nest. A loop nest costs the larger of its arithmetic and its memory
      traffic at this rate. The partial-sum loops that follow a split
      reduction are bound by the latter; walked in the contiguous order the
      lowering emits, they measure 10.8-11.3 GB/s on one core (0.35 ns per
      i32 add), against 7.7 for a bare streaming read and 9.6 for a triad.
      Walked across the partials instead the same loop costs 1.3-3.0 ns per
      add, since each access lands on a different page and the trip count
      outruns the 64-entry L1 dTLB -- the knee is at the 4 KiB stride, not
      at any cache size. This rate describes the contiguous case only.
    - `copy_bytes_per_second`: the rate of the repacks between a host
      buffer's layout and a transfer's (cnm.compact_buffer, its inverse
      cnm.expand_buffer, and memref.copy). These are strided permutations,
      an order of magnitude slower than streaming: over the 41k repacks the
      bench sample measured, time is linear in bytes (exponent 0.98) at a
      median 0.63 GB/s, with the layout moving it between 0.23 and 2.0 GB/s
      (10th to 90th percentile).
  }];

  // One struct rather than a parameter per number, so that the roofline and
  // the cost model take the same value, and its member defaults are the one
  // place the bench machine is written down.
  let parameters = (ins "::mlir::cinm::HostModel":$model);

  let builders = [
    AttrBuilder<(ins), [{
      return $_get($_ctxt, ::mlir::cinm::HostModel{});
    }]>
  ];

  let extraClassDeclaration = [{
    llvm::StringRef getName() const {
      return "host";
    }
    llvm::ArrayRef<cinm::CinmLevelDefAttr> getLevels() const {
      return {};
    }

    /// The host description `op` is priced against: the nearest one listed
    /// in a `cinm.available_platforms` of `op` or an enclosing op, or the
    /// default machine when none is. Unlike accelerator lookup, a list that
    /// names no host does not stop the search -- the host is the machine the
    /// whole program runs on, so a block that lists only its accelerator
    /// still runs on the host its function declared.
    static HostPlatformAttr getInScope(::mlir::Operation *op);
  }];

}


def CinmAvailablePlatformsAttr : ArrayOfAttr<Cinm_Dialect, "CinmPlatformArray", "platforms",
                            CinmPlatformAttrInterface.storageType> {}

#endif

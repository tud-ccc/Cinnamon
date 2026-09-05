//===- CinmWorkgroupTypeInterface.td - workgroup handles --*- tablegen -*-===//
//
// The type interface for handles to a piece of an accelerator.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_WORKGROUP_TYPE_INTERFACE
#define CINM_WORKGROUP_TYPE_INTERFACE

include "mlir/IR/Interfaces.td"

def CinmWorkgroupTypeInterface : TypeInterface<"WorkgroupTypeInterface"> {
  let description = [{
    A handle to the piece of an accelerator a program runs on: a grid of
    leaves, of some shape. `!cnm.workgroup` implements it, and so do the
    target-level handles it lowers into (`!upmem.hierarchy`).

    The interface exists for workgroups that cross a region boundary: a
    whole-program schedule allocates a group's device set ONCE, outside the
    compute blocks that use it -- necessarily at the allocator's own dialect
    level -- and forwards the value into each member block. Target-agnostic
    code (the compute_block canonicalizer, the graph-level allocation stage)
    and lowerings that meet the forwarded value must recognize it and read
    its shape without naming any target dialect, and this interface is
    exactly that contract. It lives in the cinm dialect because that is the
    layer everything else already depends on.
  }];

  let cppNamespace = "::mlir::cinm";

  let methods = [
    InterfaceMethod<
      /*desc=*/
      "Return the shape of the workgroup this value is a handle to, one "
      "extent per level of the hierarchy (for UPMEM: DPUs, then tasklets).",
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getWorkgroupShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Number of leaves of the workgroup (product of the shape).",
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumLeaves",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t product = 1;
        for (int64_t extent : $_type.getWorkgroupShape())
          product *= extent;
        return product;
      }]
    >,
  ];
}

#endif

//===- Ops.td - UPMEM dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef Upmem_OpS
#define Upmem_OpS


include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.td"
include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.td"
include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.td"


include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"



class IndexOp<string mnemonic, list<Trait> traits = []> :
    Upmem_Op<mnemonic, !listconcat(traits, [Pure])> {
  let results = (outs Index:$result);
  let assemblyFormat = "`(` `)` attr-dict";
}

def RankDimOp : IndexOp<"rank_dim"> {
  let summary = "returns the rank index of the currently executing DPU";
  let description = [{
    Returns the zero-based index of the rank that the currently executing DPU
    belongs to. In the UPMEM hardware model, DPUs are organized into ranks; this
    op is the outermost of the three index dimensions (rank, dpu, tasklet).

    ```mlir
    %rankId = upmem.rank_dim()
    ```
  }];
}


def DPUDimOp : IndexOp<"dpu_dim"> {
  let summary = "returns the DPU index within the current rank";
  let description = [{
    Returns the zero-based index of the currently executing DPU within its rank.
    This is the middle dimension of the three-level (rank, dpu, tasklet) hierarchy.

    ```mlir
    %dpuId = upmem.dpu_dim()
    ```
  }];
}


def BarrierOp : Upmem_Op<"barrier", []> {
  let summary = "add a barrier that will wait for all threads to reach this point before releasing";
  let results = (outs);
  let arguments = (ins);
  let assemblyFormat = "`(` `)` attr-dict";
  let description = [{

    ```mlir
    upmem.barrier()
    ```
  }];
}

def TaskletDimOp : IndexOp<"tasklet_dim", [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "returns the tasklet index within the current DPU";
  let description = [{
    Returns the zero-based index of the currently executing tasklet within its DPU.
    This is the innermost of the three (rank, dpu, tasklet) index dimensions.
    The upper bound is the `numTasklets` attribute of the enclosing `upmem.dpu_program`.

    ```mlir
    %taskletId = upmem.tasklet_dim()
    ```
  }];
}


def PrivateWRAMAllocOp : Upmem_Op<"pwram_alloc", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {

  let summary = "allocate private WRAM scratch space for the current tasklet";
  let description = [{
    Allocates a region of WRAM (fast on-chip scratchpad) for use by the current
    tasklet. Each tasklet gets its own address range: the allocator partitions WRAM
    across tasklets automatically. No hardware enforcement prevents cross-tasklet
    access, but the intent is that each tasklet only uses its own allocation.

    This op is the UPMEM analogue of `memref.alloc`, and produces a memref with
    the `"wram"` memory space.

    ```mlir
    %buf = upmem.pwram_alloc() : memref<64xf32, "wram">
    ```
  }];

  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$buffer);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "MemRefType":$bufferTy), [{
      $_state.addTypes({
        bufferTy
      });
    }]>
  ];

  let extraClassDeclaration = [{
    MemRefType getType() { return ::llvm::cast<MemRefType>(getBuffer().getType()); }

  }];

  let assemblyFormat = [{
     `(` `)` attr-dict `:` type($buffer)
  }];
}


def StaticAllocOp : Upmem_Op<"static_alloc", [//Symbol,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "static buffer allocation in wram or mram";
  let description = [{
    The `upmem.static_alloc` op declares a buffer in wram or mram.
    The buffer is shared between all threads, and the allocation happens statically.
    Statically allocated MRAM buffers can be named with a symbol.
    In that case, they can be referred-to from the host program
    in a scatter or gather operation.
    These allocations need to happen within a dpu_program scope.

    Example 1:

    ```mlir
    upmem.static_alloc(wram) : memref<100x100xf32, "wram">
    upmem.static_alloc @bufname(mram) : memref<100x100xf32, "mram">
    ```
  }];

  let arguments = (ins
        UPMEM_DpuMemSpaceAttr:$memSpace, 
        DefaultValuedAttr<UnitAttr, "false">:$noinit,
        DefaultValuedAttr<UnitAttr, "false">:$zeroinit,
        OptionalAttr<SymbolNameAttr>:$sym_name);

  let results = (outs Res<AnyMemRef, "structured view of the source buffer",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$buffer);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "MemRefType":$structuredType,
      "upmem::DpuMemSpace":$memSpace,
      CArg<"StringRef", "{}">:$sym_name,
      CArg<"bool", "false">:$noinit,
      CArg<"bool", "false">:$zeroinit
    )>
  ];
  let assemblyFormat = "($sym_name^)? `(` $memSpace `)` (`noinit` $noinit^)? (`zeroinit` $zeroinit^)? attr-dict `:` type($buffer)";

  let extraClassDeclaration = [{

    bool isOptionalSymbol() { return true; }
  
    bool isMram() {
      return getMemSpace() == DpuMemSpace::MRAM;
    }

    bool isWram() {
      return getMemSpace() == DpuMemSpace::WRAM;
    }

  }];

}


def LocalTransferOp : Upmem_Op<"local_transfer"> {
  let summary = "transfer between memory spaces (wram or mram)";
  let description = [{
    The `upmem.local_transfer` op copies a buffer into another buffer.
    Both buffers should have same shape and element type, but the memory
    spaces can differ (they must match the declared direction of the transfer).
    When transferring part of a buffer, a memref.subview should be used
    as the source or destination operand.

    ```mlir
    upmem.local_transfer (wram to mram) %source into %dest : memref<2048xf32, "wram"> to memref<2048xf32, "mram">
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "the memref to copy from",
                           [MemReadAt<0, FullEffect>]>:$source,
                       Arg<AnyMemRef, "the memref to copy to",
                           [MemWriteAt<0, FullEffect>]>:$target);

  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $source `into` $target attr-dict `:` type($source) `to` type($target)
  }];

}

def DpuResource : Resource<"DpuSetResource">;

// The transfer ops form one family at three levels of specificity, from most
// to least general:
//
//   scatter_blocks / gather_blocks  n blocks per DPU, each from its own place
//   scatter_on_array / gather_on_array   one block per DPU
//   broadcast                       the same block to every DPU
//
// `transferCount` is the size of one block in every one of them, so the only
// thing that changes going down the family is how many blocks there are and
// where they may sit. --upmem-specialize-transfers narrows an op to a lower
// level when its map permits.
class ScatterGatherLike<string mnemonic, string kw, MemoryEffect effect, MemoryEffect dpuEffect>
  : Upmem_Op<mnemonic, [MemoryEffects<[effect, dpuEffect]>,
                        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let arguments = (ins Arg<AnyMemRef, "", [effect]>:$hostBuffer,
                  FlatSymbolRefAttr:$dpuBufRef,
                  I64Attr:$transferCount,
                  AffineMapAttr:$scatterMap,
                  DeviceHierarchyType:$hierarchy);

  let results = (outs);

  let description = [{
    Transfers data between a host buffer and the MRAM buffers of a DPU array,
    one contiguous block per DPU.

    `dpuBufRef` is the symbol name of a `upmem.static_alloc` declared inside the
    body of the `upmem.dpu_program` loaded by the `hierarchy`. The verifier checks
    that the symbol exists and refers to a `upmem.static_alloc`.

    `transferCount` is the number of elements (not bytes) to copy per DPU.

    `scatterMap` is an affine map `(rank, dpu) -> (i, ...)` that computes the
    starting index in `hostBuffer` for each DPU. It must have exactly two
    dimensions (rank and dpu) and as many results as the rank of `hostBuffer`.
    The verifier checks that the block starting there stays inside one
    contiguous run of `hostBuffer`.
  }];

  let extraClassDeclaration = [{
    int64_t getDpuBufferSizeInBits() {
      return getTransferCount() * getHostBuffer().getType().getElementTypeBitWidth();
    }

    int64_t getDpuBufferSizeInBytes() {
      return getDpuBufferSizeInBits() / 8;
    }

    /// One block per DPU, so always 1. Present so that generic code can treat
    /// this op and its `_blocks` counterpart alike.
    int64_t getNumBlocksPerDpu() { return 1; }

    /// Return the DpuProgramOp loaded on this hierarchy, or null.
    DpuProgramOp getDpuProgram();
  }];

  let hasVerifier = 1;
  let assemblyFormat =
     "$hostBuffer `[` $transferCount `elts` `,` $scatterMap `]` `" # kw # "` $dpuBufRef `of` $hierarchy attr-dict"
     # "`:` type($hostBuffer) `" # kw # "` qualified(type($hierarchy))";
}

def ScatterOnArrayOp: ScatterGatherLike<"scatter_on_array", "onto", MemRead, MemWrite<DpuResource>> {
  let summary = "copy one contiguous slice of a host buffer into the MRAM of each DPU";
  let description = [{
    For each DPU in `hierarchy`, copies `transferCount` elements from `hostBuffer`
    into the DPU's MRAM buffer named `dpuBufRef`. The slice of `hostBuffer`
    assigned to a given DPU is determined by evaluating `scatterMap(rank, dpu)`,
    which yields the starting multi-dimensional index into `hostBuffer`. The
    copy happens as a single flat transfer per DPU, so the `transferCount`
    elements must be contiguous in `hostBuffer` -- one block per DPU, spread
    across the array, which is what the name says.

    See `upmem.scatter_blocks` for the general form, which gives a DPU several
    blocks that need not be contiguous with one another.

    ```mlir
    // scatter 1024 elements into @input_buf on each DPU;
    // DPU (r, d) reads from hostBuf[d, 0]
    upmem.scatter_on_array %hostBuf[1024 elts, affine_map<(r, d) -> (d, 0)>] onto @input_buf of %dpus
        : memref<128x1024xi32> onto !upmem.hierarchy<1x128x16>
    ```
  }];

}


def GatherOnArrayOp: ScatterGatherLike<"gather_on_array", "from", MemWrite, MemRead<DpuResource>> {
  let summary = "collect one contiguous MRAM slice from each DPU into a host buffer";
  let description = [{
    For each DPU in `hierarchy`, copies `transferCount` elements from the DPU's
    MRAM buffer named `dpuBufRef` into `hostBuffer`. The destination slice in
    `hostBuffer` for a given DPU is determined by evaluating `scatterMap(rank, dpu)`.

    This is the inverse of `upmem.scatter_on_array` and is typically used to
    retrieve computation results after `upmem.wait_for` completes.

    ```mlir
    // gather 128 result elements per DPU back into hostBuf[d, 0]
    upmem.gather_on_array %hostBuf[128 elts, affine_map<(r, d) -> (d, 0)>] from @output_buf of %dpus
        : memref<128x128xi32> from !upmem.hierarchy<1x128x16>
    ```
  }];

}

class BlockTransferLike<string mnemonic, string kw, MemoryEffect effect, MemoryEffect dpuEffect>
  : Upmem_Op<mnemonic, [MemoryEffects<[effect, dpuEffect]>,
                        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let description = [{
    Transfers `numBlocksPerDpu` blocks of `transferCount` elements each between
    `hostBuffer` and the MRAM buffer named `dpuBufRef` of every DPU in
    `hierarchy`, using the UPMEM SDK's scatter/gather transfer API
    (`dpu_push_sg_xfer`), which locates each block independently.

    `dpuBufRef` is the symbol name of a `upmem.static_alloc` declared inside the
    body of the `upmem.dpu_program` loaded by the `hierarchy`. The verifier checks
    that the symbol exists and refers to a `upmem.static_alloc`.

    `transferCount` is the number of elements (not bytes) in a single block,
    and is the same for every block. Each block must be contiguous in
    `hostBuffer`, but different blocks need not be contiguous with one another.

    `scatterMap` is an affine map `(rank, dpu, block) -> (i, ...)` that
    computes the starting index in `hostBuffer` of one block. It must have
    exactly three dimensions and as many results as the rank of `hostBuffer`.

    `numBlocksPerDpu` gives the number of blocks transferred per DPU, i.e. the
    extent of the `block` dimension. This is independent of the number of
    tasklets declared by `hierarchy`: blocks are units of transfer, and a
    single tasklet's data may well arrive as several of them.
  }];

  let arguments = (ins Arg<AnyMemRef, "", [effect]>:$hostBuffer,
                  FlatSymbolRefAttr:$dpuBufRef,
                  I64Attr:$transferCount,
                  AffineMapAttr:$scatterMap,
                  DeviceHierarchyType:$hierarchy,
                  I64Attr:$numBlocksPerDpu);

  let results = (outs);

  let extraClassDeclaration = [{
    int64_t getDpuBufferSizeInBits() {
      return getTransferCount() * getHostBuffer().getType().getElementTypeBitWidth();
    }

    int64_t getDpuBufferSizeInBytes() {
      return getDpuBufferSizeInBits() / 8;
    }

    /// Return the DpuProgramOp loaded on this hierarchy, or null.
    DpuProgramOp getDpuProgram();
  }];

  let hasVerifier = 1;
  let assemblyFormat =
    "$hostBuffer `[` $transferCount `elts` `,` $scatterMap `,` $numBlocksPerDpu `blocks` `]` `" # kw # "` $dpuBufRef `of` $hierarchy attr-dict"
    # "`:` type($hostBuffer) `" # kw # "` qualified(type($hierarchy))";
}

def ScatterBlocksOp: BlockTransferLike<"scatter_blocks", "onto", MemRead, MemWrite<DpuResource>> {
  let summary = "copy several per-DPU blocks of a host buffer into MRAM using the UPMEM SDK scatter transfer API";
  let description = [{
    The general scatter. Unlike `upmem.scatter_on_array`, a DPU's data may
    come from several places in `hostBuffer` that are not contiguous with one
    another.

    ```mlir
    // scatter 8-element blocks into @input_buf on each DPU; DPU (r, d)'s
    // block b is read from hostBuf[d, b, 0]
    upmem.scatter_blocks %hostBuf[8 elts, affine_map<(r, d, b) -> (d, b, 0)>, 4 blocks] onto @input_buf of %dpus
        : memref<1x4x8xi32> onto !upmem.hierarchy<1x4x4>
    ```
  }];
}

def GatherBlocksOp: BlockTransferLike<"gather_blocks", "from", MemWrite, MemRead<DpuResource>> {
  let summary = "collect several per-DPU MRAM blocks into a host buffer using the UPMEM SDK gather transfer API";
  let description = [{
    The general gather, and the inverse of `upmem.scatter_blocks`. Unlike
    `upmem.gather_on_array`, a DPU's results may be written to several places
    in `hostBuffer` that are not contiguous with one another.

    ```mlir
    // gather 8-element blocks from @output_buf on each DPU; DPU (r, d)'s
    // block b is written to hostBuf[d, b, 0]
    upmem.gather_blocks %hostBuf[8 elts, affine_map<(r, d, b) -> (d, b, 0)>, 4 blocks] from @output_buf of %dpus
        : memref<1x4x8xi32> from !upmem.hierarchy<1x4x4>
    ```
  }];
}

def BroadcastOp: Upmem_Op<"broadcast", [
      MemoryEffects<[MemRead, MemWrite<DpuResource>]>,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "copy a host buffer into the MRAM of every DPU, identically";
  let description = [{
    Copies all of `hostBuffer` into the DPU's MRAM buffer named `dpuBufRef`,
    identically for every DPU in `hierarchy`. Unlike `upmem.scatter_on_array`, there is
    no affine map and no per-DPU slicing: every DPU receives the same data, a
    single flat transfer starting at offset zero in `hostBuffer` and covering
    all of its elements. To broadcast a subset of a larger buffer, take a
    `memref.subview` of it first.

    `hostBuffer` must be contiguous in memory (a `memref.subview` is only
    valid here if the sliced region is itself contiguous). Its shape must be
    compatible with the target buffer's shape up to rank reduction/expansion
    by extent-1 dimensions.

    `dpuBufRef` is the symbol name of a `upmem.static_alloc` declared inside the
    body of the `upmem.dpu_program` loaded by the `hierarchy`. The verifier checks
    that the symbol exists and refers to a `upmem.static_alloc`.

    ```mlir
    // broadcast 1024 elements into @input_buf on every DPU
    upmem.broadcast %hostBuf onto @input_buf of %dpus
        : memref<1024xi32> onto !upmem.hierarchy<1x128x16>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$hostBuffer,
                  FlatSymbolRefAttr:$dpuBufRef,
                  DeviceHierarchyType:$hierarchy);

  let results = (outs);

  let extraClassDeclaration = [{
    int64_t getNumElements() {
      return getHostBuffer().getType().getNumElements();
    }

    int64_t getDpuBufferSizeInBits() {
      return getNumElements() * getHostBuffer().getType().getElementTypeBitWidth();
    }

    int64_t getDpuBufferSizeInBytes() {
      return getDpuBufferSizeInBits() / 8;
    }

    /// Return the DpuProgramOp loaded on this hierarchy, or null.
    DpuProgramOp getDpuProgram();
  }];

  let hasVerifier = 1;
  let assemblyFormat = [{
    $hostBuffer `onto` $dpuBufRef `of` $hierarchy attr-dict
    `:` type($hostBuffer) `onto` qualified(type($hierarchy))
  }];
}



def DpuProgramOp : Upmem_Op<"dpu_program",
[
  IsolatedFromAbove,
  SymbolTable,
  Symbol,
  AffineScope,
  AutomaticAllocationScope
]> {
  let summary = "the DPU-side kernel program";
  let description = [{
    Defines the program that runs on each DPU. The body region contains the
    computation and may use `upmem.static_alloc`, `upmem.pwram_alloc`,
    `upmem.local_transfer`, and standard arithmetic/control-flow ops.

    `numTasklets` specifies how many tasklets execute the program concurrently on
    each DPU. Each tasklet can identify itself via `upmem.tasklet_dim()`.

    `upmem.dpu_program` is a `SymbolTable`: named `upmem.static_alloc` ops inside
    its body are registered as symbols and can be referenced by name from
    `upmem.scatter_*` and `upmem.gather_*` ops on the host side.

    The body must be terminated by `upmem.return`.

    ```mlir
    upmem.dpu_program @my_kernel() tasklets(16) {
      %input  = static_alloc @input(mram)  : memref<1024xi32, "mram">
      %output = static_alloc @output(mram) : memref<1xi32, "mram">
      %tmp    = pwram_alloc() : memref<1024xi32, "wram">
      // ... computation ...
      return
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$numTasklets);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name `(` `)` `tasklets` `(` $numTasklets `)` attr-dict-with-keyword $body";

  let extraClassDeclaration = [{

  }];
}


def AllocDPUsOp: Upmem_Op <"alloc_dpus",  [ DeclareOpInterfaceMethods<SymbolUserOpInterface>, MemoryEffects<[MemAlloc<DpuResource>]> ]> {
  let summary = "UPMEM alloc dpus";
  let description = [{
    The upmem.alloc_dpus op calls initialization routine of the UPMEM SDK.
    It loads the specified DPU program onto the DPUs. This is an expensive
    operation that should be hoisted outside of loops. If we are going to
    run the same program several times, the same allocation should be reused.
    The DPU program still needs input buffers to be scattered before it can
    be executed.
  }];
  let arguments = (ins SymbolRefAttr:$dpu_program_ref);
  let results = (outs DeviceHierarchyType:$wg);

  let assemblyFormat = [{
     `with` `program` $dpu_program_ref attr-dict `:` qualified(type($wg))
  }];

  let extraClassDeclaration = [{
    /// Look up and return the DpuProgramOp referenced by this op, or null.
    DpuProgramOp getDpuProgram();
  }];
}



def FreeDPUsOp: Upmem_Op <"free_dpus", [MemoryEffects<[MemFree<DpuResource>]>]>{
  let summary = "release a DPU allocation back to the runtime";
  let description = [{
    Releases the DPU set acquired by the corresponding `upmem.alloc_dpus`.
    After this point the hierarchy value must not be used.

    ```mlir
    upmem.free_dpus %dpus : !upmem.hierarchy<1x64x16>
    ```
  }];
  let arguments = (ins DeviceHierarchyType:$hierarchy);

  let assemblyFormat = [{
     $hierarchy attr-dict `:` qualified(type($hierarchy))
  }];
}

def WaitForOp : Upmem_Op<"wait_for", [MemoryEffects<[MemRead<DpuResource>, MemWrite<DpuResource>]>]> {
  let summary = "launch DPU execution and block until all DPUs complete";
  let description = [{
    Triggers execution of the DPU program previously loaded by `upmem.alloc_dpus`
    on all DPUs in the hierarchy. Blocks the host until every DPU finishes.
    After this op, the results can be retrieved with `upmem.gather_on_array`.

    ```mlir
    upmem.wait_for %dpus : !upmem.hierarchy<1x64x16>
    ```
  }];

  let arguments = (ins DeviceHierarchyType:$dpuSet);

  let assemblyFormat = " $dpuSet attr-dict `:` qualified(type($dpuSet))";

  let extraClassDeclaration = [{
    /// Return the DpuProgramOp running on this hierarchy, or null.
    DpuProgramOp getDpuProgram();
  }];
}

def ReturnOp : Upmem_Op<"return", [ParentOneOf<["DpuProgramOp"]>, Pure, Terminator]>,
    Arguments<(ins)>, Results<(outs)>
      {
  let summary = "terminator for a upmem.dpu_program body";
  let description = [{
    Terminates the body region of a `upmem.dpu_program`. Every control-flow
    path through the program body must end with this op.

    ```mlir
    upmem.dpu_program @kernel() tasklets(1) {
      // ...
      return
    }
    ```
  }];
  let assemblyFormat = "attr-dict";
}

#endif

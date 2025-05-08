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
    Upmem_Op<mnemonic, !listconcat(traits, [Pure])>,
        Results<(outs Index)> {
  let assemblyFormat = "attr-dict";
}

def RankDimOp : IndexOp<"rank_dim"> {
  let description = [{
    Returns the index of the running rank in the code.

    Example:

    ```mlir
    %dId = upmem.rank_id
    ```
  }];
}


def DPUDimOp : IndexOp<"dpu_dim"> {
  let description = [{
    Returns the index of the running dpu in the code.

    Example:

    ```mlir
    %dId = upmem.dpu_dim
    ```
  }];
}

def TaskletDimOp : IndexOp<"tasklet_dim", [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let description = [{
    Returns the index of the running task in the code.
    Example:

    ```mlir
    %tId = upmem.tasklet_dim x
    ```
  }];
}

def TaskletIDOp : Upmem_Op<"tasklet_id"> {
  let description = [{
  }];
  let results = (outs Index:$tasklet_id);
  let assemblyFormat = [{
     attr-dict `:` type($tasklet_id)
  }];
}



def PrivateWRAMAllocOp : Upmem_Op<"pwram_alloc", [
   TypesMatchWith<"workgroup shape matches buffer shape", "buffer", "bytes",
        "upmem::detail::flatMemRefType($_self)">,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {

  let summary = "UPMEM WRAM (fast scratchpad) memory allocation operation for a single tasklet.";
  let description = [{
    The `upmem.pwram_alloc` operation allocates a wram region that is supposed to be only accessible
    to a single tasklet. In reality, no constraints is placed on the memory itself, but this space
    is allocated by an allocator that allocates WRAM for each tasklet seperately and the addresses
    are assigned automatically. In the original C interface, it is possible to access the other
    tasklet space, but the goal here is to make it more abstract, simpler, and to not reduce the power
    as well.
    Interface-wise, it is similar to the `memref.alloc` op.

    Example:

    ```mlir
    %buffer, %bytes = upmem.pwram_alloc() : memref<64xf32>
    ```
  }];

  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$buffer, I8MemRef:$bytes);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "MemRefType":$bufferTy), [{
      $_state.addTypes({
        bufferTy,
        detail::flatMemRefType(bufferTy)
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


def StaticAllocOp : Upmem_Op<"static_alloc", [Symbol,
 TypesMatchWith<"workgroup shape matches buffer shape", "buffer", "bytes",
        "upmem::detail::flatMemRefType($_self)">,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "static buffer allocation in wram or mram";
  let description = [{
    The `upmemx.static_alloc` op declares a buffer in wram or mram.
    The buffer is shared between all threads, and the allocation happens statically.
    Example 1:

    ```mlir
    upmemx.static_alloc(wram) : memref<100x100xf32, "wram">
    upmemx.static_alloc @bufname(mram) : memref<100x100xf32, "mram">
    ```
  }];

  let arguments = (ins UnitAttr:$isWram, DefaultValuedAttr<UnitAttr, "false">:$noinit, OptionalAttr<SymbolNameAttr>:$sym_name);
  let results = (outs Res<AnyMemRef, "structured view of the source buffer",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$buffer,
                          Res<I8MemRef, "byte view on the source buffer">:$bytes);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "MemRefType":$structuredType,
      "bool":$isWram,
      CArg<"StringRef", "{}">:$sym_name,
      CArg<"bool", "false">:$noinit
    )>
  ];
  let assemblyFormat = "($sym_name^)? `(` (`wram` $isWram^):(`mram`)? `)` (`noinit` $noinit^)? attr-dict `:` type($buffer)";

  let extraClassDeclaration = [{

    bool isOptionalSymbol() { return true; }
  
    bool isMram() {
      return !getIsWram();
    }

    bool isWram() {
      return getIsWram();
    }

  }];

}


def MemcpyOp : Upmem_Op<"memcpy", []> {

  let summary = "UPMEM memcpy operation";

  let description = [{
    The `upmem.memcpy` operation copies the content of one memref to another in a blocking manner.

    Example:

    ```mlir
    %token = upmem.memcpy [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
    ```
  }];

  let arguments = (ins MemcpyDirAttr:$direction,
                   Arg<AnyMemRef, "", [MemWrite]>:$buffer,
                   Index:$size, Index:$offset);

  let assemblyFormat = [{
    $direction $buffer`,` $size`,` $offset `:` type($buffer)`,` type($size)`,` type($offset) attr-dict
  }];

  // let hasFolder = 1;
  // let hasVerifier = 1;
  // let hasCanonicalizer = 1;
}



def TransferOp : Upmem_Op<"local_transfer", []> {

  let summary = "UPMEM memcpy operation between mram and wram";

  let description = [{
    The `upmem.on_dpu_transfer` operation copies the content of one memref to another.
    Note: 
    - the memrefs may be views on another memref? ideally no - you need to know 
    at this level that they're contiguous and aligned
    - can this operator transfer more than 2048 bytes? Probably, 
    we could abstract over transfer size.
    - can this operator transfer multi-dimensional memrefs? Probably not

    Translation strategy:
    - tilefirst MRAM buffers are mapped to views on a linear i8 buffer (memref.view).
    - transfers operate directly on the underlying buffer.

    Example:

    ```mlir
    %token = upmem.local_transfer %src[%offset] to %dst[%offset2] count 16 : memref<16xf32, "wram">, memref<16xf32, "mram">
    ```
  }];

  let arguments = (ins 
                   Arg<I8MemRef, "", [MemRead]>:$source,
                   Arg<I8MemRef, "", [MemWrite]>:$dest,
                   Index:$sourceOffset,
                   Index:$destOffset,
                   I64Attr:$count);

  let assemblyFormat = [{
    $source `[` $sourceOffset `]` `to` $dest `[` $destOffset `]` `count` $count attr-dict `:` type($source) `to` type($dest)
  }];

  // let hasFolder = 1;
  // let hasVerifier = 1;
  // let hasCanonicalizer = 1;
}



def BaseMRAMAddrOp: Upmem_Op <"dpu_heap_base_addr", []> {
  let summary = "Base offset of the MRAM in the DPUs for starting data copy";
  let description = [{
  }];
  let results = (outs Index:$base_offset);

  let assemblyFormat = [{
     attr-dict `:` type($base_offset)
  }];
}

def BaseDPUMemOffsetOp: Upmem_Op <"base_dpu_mem_offset", []> {
  let summary = "Base offset of the MRAM in the DPUs for starting data copy";
  let description = [{
  }];
  let results = (outs Index:$base_offset);

  let assemblyFormat = [{
     attr-dict `:` type($base_offset)
  }];
}


class ScatterGatherLike<string mnemonic, string kw, MemoryEffect effect>
  : Upmem_Op<mnemonic, [MemoryEffects<[MemWrite]>]> {

  let arguments = (ins Arg<AnyMemRef, "", [effect]>:$hostBuffer,
                  FlatSymbolRefAttr:$dpuBufRef,
                  I64Attr:$transferCount,
                  AffineMapAttr:$scatterMap,
                  DeviceHierarchyType:$hierarchy);

  let results = (outs);

  let description = [{
    Scatter or gather a host buffer onto a DPU array.
    The buffer ref is the simple name of a buffer declared with upmem.static_alloc in
    the body of the upmem.dpu_program of the workgroup.
    

  }];

  let extraClassDeclaration = [{
    int64_t getDpuBufferSizeInBits() {
      return getTransferCount() * getHostBuffer().getType().getElementTypeBitWidth();
    }

    int64_t getDpuBufferSizeInBytes() {
      return getDpuBufferSizeInBits() / 8;
    }

  }];

  let hasVerifier = 1;
  let assemblyFormat =
     "$hostBuffer `[` $transferCount `,` $scatterMap `]` `" # kw # "` $dpuBufRef `of` $hierarchy attr-dict"
     # "`:` type($hostBuffer) `" # kw # "` qualified(type($hierarchy))";
}

def ScatterOp: ScatterGatherLike<"scatter", "onto", MemRead> {
  let summary = "scatter data to dpus";
  let description = [{
  }];

}


def GatherOp: ScatterGatherLike<"gather", "from", MemWrite> {
  let summary = "Gather data from dpus into a buffer on the host";
  let description = [{
  }];

}



def DpuProgramOp : Upmem_Op<"dpu_program",
[
  IsolatedFromAbove,
  SymbolTable,
  Symbol,
  AffineScope
]> {
  let summary = "Represents the final DPU program";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$numTasklets);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name `(` `)` `tasklets` `(` $numTasklets `)` attr-dict-with-keyword $body";

}


def AllocDPUsOp: Upmem_Op <"alloc_dpus",  [ DeclareOpInterfaceMethods<SymbolUserOpInterface> ]> {
  let summary = "UPMEM alloc dpus";
  let description = [{
  }];
  let arguments = (ins SymbolRefAttr:$dpu_program_ref);
  let results = (outs DeviceHierarchyType:$wg);

  let assemblyFormat = [{
     `with` `program` $dpu_program_ref attr-dict `:` qualified(type($wg))
  }];
}



def FreeDPUsOp: Upmem_Op <"free_dpus", []>{
  let summary = "UPMEM free dpus";
  let description = [{
  }];
  let arguments = (ins DeviceHierarchyType:$hierarchy);

  let assemblyFormat = [{
     $hierarchy attr-dict `:` qualified(type($hierarchy))
  }];
}

def WaitForOp : Upmem_Op<"wait_for", []> {
  let summary = "Run a program on a dpu set synchronously";
  let description = [{

  }];

  let arguments = (ins DeviceHierarchyType:$dpuSet);

  let assemblyFormat = " $dpuSet attr-dict `:` qualified(type($dpuSet))";

}

def ReturnOp : Upmem_Op<"return", [ParentOneOf<["DpuProgramOp"]>, Pure, Terminator]>,
    Arguments<(ins)>, Results<(outs)>
      {
  let summary = "Terminator for UPMEM functions.";
  let description = [{
  }];
  let assemblyFormat = "attr-dict";
}

#endif

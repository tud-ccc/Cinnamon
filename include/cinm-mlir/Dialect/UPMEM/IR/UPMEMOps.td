//===- Ops.td - UPMEM dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_OPS
#define UPMEM_OPS


include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.td"
include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.td"
include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.td"

include "AsyncOpInterface.td"

include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"


class UPMEM_IndexOp<string mnemonic, list<Trait> traits = []> :
    UPMEM_Op<mnemonic, !listconcat(traits, [
        Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>])>, 
        Results<(outs Index)> {
  let assemblyFormat = "attr-dict";
}

def UPMEM_RankDimOp : UPMEM_IndexOp<"rank_dim"> {
  let description = [{
    NOTE: Not used at the moment.

    Returns the index of the running rank in the code.

    Example:

    ```mlir
    %dId = upmem.rank_id 
    ```
  }];
}


def UPMEM_DPUDimOp : UPMEM_IndexOp<"dpu_dim"> {
  let description = [{
    NOTE: Not used at the moment.

    Returns the index of the running dpu in the code.

    Example:

    ```mlir
    %dId = upmem.dpu_dim 
    ```
  }];
}

def UPMEM_TaskletDimOp : UPMEM_IndexOp<"tasklet_dim"> {
  let description = [{
    NOTE: Not used at the moment.

    Returns the index of the running task in the code.
    Example:

    ```mlir
    %tId = upmem.tasklet_dim x
    ```
  }];
}


def UPMEM_PrivateWRAMAllocOp : UPMEM_Op<"pwram_alloc", [
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
    TODO: support async 

    Example:

    ```mlir
    %memref = upmem.pwram_alloc : memref<64xf32>
    ```
  }];

  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref);

  let extraClassDeclaration = [{
    MemRefType getType() { return ::llvm::cast<MemRefType>(getMemref().getType()); }
  }];

  let assemblyFormat = [{
     attr-dict `:` type($memref)
  }];
}



def UPMEM_LaunchOp : UPMEM_Op<"launch", [
      AutomaticAllocationScope, AttrSizedOperandSegments, UPMEM_AsyncOpInterface,
      DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins Variadic<UPMEM_AsyncToken>:$asyncDependencies, DeviceHierarchy:$device_hierarchy,
               Index:$rankSize, 
               Index:$dpuSize, Index:$taskletSize,
               Optional<I32>:$dynamicSharedMemorySize)>,
    Results<(outs Optional<UPMEM_AsyncToken>:$asyncToken)> {
  let summary = "UPMEM kernel launch operation";

  let description = [{
    Syntax:

    ```
    operation ::= `upmem.launch` (`async` (`[` ssa-id-list `]`)? )?
                             ssa-id 
                             `ranks` `(` ssa-id `)` `in` ssa-reassignment
                             `dpus` `(` ssa-id `)` `in` ssa-reassignment
                             `tasklets` `(` ssa-id `)` `in` ssa-reassignment
                             (dynamic_shared_memory_size ssa-use)?
                             memory-attribution
                             region attr-dict?
    ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
    memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                           (`private` `(` ssa-id-and-type-list `)`)?
    ```

    Example:

    ```mlir
    upmem.launch %upmem_token !upmem.hierarchy<2x64x16> ranks(%rd) in (%sr = %0)
                 dpus(%dd) in (%sd = %1)
                 tasklets(%td) in (%st= %3) {
      "some_op"(%bx, %tx) : (index, index) -> ()
      %42 = load %val1[%bx] : memref<?xf32, 1>
    }
    }
    ```
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "Value":$device_hierarchy, "Value":$rankSize, "Value":$dpuSize,
      "Value":$taskletSize, 
      CArg<"Value", "nullptr">:$dynamicSharedMemorySize,
      CArg<"Type", "nullptr">:$asyncTokenType,
      CArg<"ValueRange", "{}">:$asyncDependencies,
      CArg<"TypeRange", "{}">:$workgroupAttributions,
      CArg<"TypeRange", "{}">:$privateAttributions)>
  ];

  let extraClassDeclaration = [{
    KernelDim getRankIdClass();
    KernelDim getDPUIdClass();
    KernelDim getTaskletIdClass();
    
    KernelDim getRankSizeClass();
    KernelDim getDPUSizeClass();
    KernelDim getTaskletSizeClass();

    KernelDim getRankSizeOperandValue();
    KernelDim getDPUSizeOperandValue();
    KernelDim getTaskletSizeOperandValue();

    static StringRef getRanksKeyword() { return "ranks"; }
    static StringRef getDPUsKeyword() { return "dpus"; }
    static StringRef getTaskletsKeyword() { return "tasklets"; }
    static StringRef getDynamicSharedMemorySizeKeyword() {
      return "dynamic_shared_memory_size";
    }

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    static constexpr unsigned kNumConfigOperands = 3;

    /// The number of region attributes containing the launch configuration,
    /// placed in the leading positions of the argument list.
    static constexpr unsigned kNumConfigRegionAttributes = 6;

    /// Returns the keywords used in the custom syntax for this Op.
    static StringRef getWorkgroupKeyword() { return "workgroup"; }
    static StringRef getPrivateKeyword() { return "private"; }

    /// Returns the number of buffers located in the workgroup memory.
    unsigned getNumWorkgroupAttributions() {
      auto attr = (*this)->getAttrOfType<IntegerAttr>(
          getNumWorkgroupAttributionsAttrName());
      return attr ? attr.getInt() : 0;
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the workgroup memory
    ArrayRef<BlockArgument> getWorkgroupAttributions() {
      auto begin =
          std::next(getBody().args_begin(), kNumConfigRegionAttributes);
      auto end = std::next(begin, getNumWorkgroupAttributions());
      return {begin, end};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// workgroup memory.
    BlockArgument addWorkgroupAttribution(Type type, Location loc);

    /// Returns the number of buffers located in the private memory.
    unsigned getNumPrivateAttributions() {
      return getBody().getNumArguments() - kNumConfigRegionAttributes -
          getNumWorkgroupAttributions();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the private memory.
    ArrayRef<BlockArgument> getPrivateAttributions() {
      // Buffers on the private memory always come after buffers on the workgroup
      // memory.
      auto begin =
          std::next(getBody().args_begin(),
                    kNumConfigRegionAttributes + getNumWorkgroupAttributions());
      return {begin, getBody().args_end()};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// private memory.
    BlockArgument addPrivateAttribution(Type type, Location loc);

    /// Returns the name of the attribute containing the number of buffers
    /// located in the workgroup memory.
    static StringRef getNumWorkgroupAttributionsAttrName() {
      return "workgroup_attributions";
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def UPMEM_TerminatorOp : UPMEM_Op<"terminator", [HasParent<"LaunchOp">,
                                             Pure, Terminator]>,
    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for UPMEM launch regions.";
  let description = [{
    A terminator operation for regions that appear in the body of `upmem.launch`
    operation.  These regions are not expected to return any value so the
    terminator takes no operands.
  }];

  let assemblyFormat = "attr-dict";
}




def UPMEM_MemcpyOp : UPMEM_Op<"memcpy", []> {

  let summary = "UPMEM memcpy operation";

  let description = [{
    The `upmem.memcpy` operation copies the content of one memref to another.

    The op does not execute before all async dependencies have finished
    executing.
copy_to_mram
    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !upmem.async.token.

    Example:

    ```mlir
    %token = upmem.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
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


def UPMEM_AllocDPUsOp: UPMEM_Op <"alloc_dpus", []> {
  let summary = "UPMEM alloc dpus";
  let description = [{
  }];
  let results = (outs DeviceHierarchy:$hierarchy_shape);

  let assemblyFormat = [{
     attr-dict `:` type($hierarchy_shape)
  }];
}

def UPMEM_BaseMRAMAddrOp: UPMEM_Op <"dpu_heap_base_addr", []> {
  let summary = "Base offset of the MRAM in the DPUs for starting data copy";
  let description = [{
  }];
  let results = (outs Index:$base_offset);

  let assemblyFormat = [{
     attr-dict `:` type($base_offset)
  }];
}

def UPMEM_BaseDPUMemOffsetOp: UPMEM_Op <"base_dpu_mem_offset", []> {
  let summary = "Base offset of the MRAM in the DPUs for starting data copy";
  let description = [{
  }];
  let results = (outs Index:$base_offset);

  let assemblyFormat = [{
     attr-dict `:` type($base_offset)
  }];
}

def UPMEM_ScatterOp: UPMEM_Op <"scatter", []> {
  let summary = "scatter data to dpus";
  let description = [{
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$host_data,
                  Index:$dpu_mem_offset, DeviceHierarchy:$hierarchy);


  let results = (outs Index:$new_mem_offset);
  let assemblyFormat = [{
     $host_data `into` $hierarchy `at` $dpu_mem_offset attr-dict `:` type($host_data)`,` type($hierarchy) `,` type($dpu_mem_offset)  `->` type($new_mem_offset)
  }];
}


def UPMEM_GatherOp: UPMEM_Op <"gather", []> {
  let summary = "gather data from dpus";
  let description = [{
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$host_memref,
                  Index:$dpu_mem_offset, DeviceHierarchy:$hierarchy);


  let results = (outs Index:$new_mem_offset);
  let assemblyFormat = [{
     $host_memref `from` $hierarchy `at` $dpu_mem_offset attr-dict `:` type($host_memref)`,` type($hierarchy) `,` type($dpu_mem_offset)  `->` type($new_mem_offset)
  }];
}

#endif

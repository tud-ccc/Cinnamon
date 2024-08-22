//===- Ops.td - Hbmpim dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Hbmpim dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef HBMPIM_OPS
#define HBMPIM_OPS


include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimBase.td"
include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimAttributes.td"
include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimTypes.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/BuiltinAttributes.td"


def PreloadGemvOp: Hbmpim_Op<"preload_gemv", []>{
    let summary = "Load data for GEMV-based kernels";
    let description = [{}];
    let arguments = (ins DataDimTypeAttr:$dim,
                  Index:$startRow, Index:$startCol);
    let assemblyFormat = [{$dim `,` $startRow `,` $startCol attr-dict `:` 
        type($startRow) `,` type($startCol)}];
}

def HostPreloadGemvOp: Hbmpim_Op<"host_preload_gemv", []>{
    let summary = "Load data for GEMV-based kernels";
    let description = [{}];
    let arguments = (ins DataDimTypeAttr:$dim, AnyMemRef:$inBuffer);
    let assemblyFormat = [{$dim `,` $inBuffer attr-dict `:` type($inBuffer) }];
}

def ExecuteGemvOp: Hbmpim_Op<"execute_gemv", []>{
  let summary = "Execute gemv operations, supporting tree based and normal gemv";
  let description = [{}];
  let arguments = (ins Index:$batch, Index:$M, Index:$N, Index:$numOutputTile, Index:$numInputTile, BoolAttr:$is_tree, 
    I64Attr:$numBank, I64Attr:$numChannel, I64Attr:$numRank, I64Attr:$numGrf);
  let assemblyFormat = [{$batch `,` $M `,` $N `,` $numOutputTile `,` $numInputTile
      attr-dict `:` type($batch) `,` type($M) `,` type($N) `,` type($numOutputTile) `,` type($numInputTile)}];
}

def ReadResultOp: Hbmpim_Op<"read_result", []>{
    let summary = "Read the result of a **gemv** operation";
    let description = [{Although it sounds weird, but this is how the simulator implements it.
                        This only works for gemv. }];
    let arguments = (ins DeviceBurstTypeAttr:$burst, PimBankTypeAttr:$bank_type, Index:$out_dim,
                  Index:$base_addr, Index:$startRow, Index:$startCol);
    let assemblyFormat = [{$burst `,` $bank_type `,` $out_dim `,` $base_addr `,`
        $startRow `,` $startCol attr-dict `:` 
        type($out_dim) `,` type($base_addr) `,` type($startRow) `,` type($startCol)}];
}

def HostReadResultOp: Hbmpim_Op<"host_read_result", []>{
    let summary = "read results data for GEMV-based kernels";
    let description = [{}];
    let arguments = (ins DeviceBurstTypeAttr:$dim, AnyMemRef:$outBuffer);
    let assemblyFormat = [{$dim `,` $outBuffer attr-dict `:` type($outBuffer) }];
}


def PreloadNoReplacementOp: Hbmpim_Op<"preload_no_replacement", []>{
    let summary = "Load data that are not being replaced";
    let description = [{}];
    let arguments = (ins AnyMemRef:$inBuffer,
                  Index:$startRow, Index:$startCol);
    let assemblyFormat = [{$inBuffer `,` $startRow `,` $startCol attr-dict `:` 
         type($inBuffer) `,` type($startRow) `,` type($startCol)}];
}


def ReadDataOp: Hbmpim_Op<"read_data", []>{
    let summary = "Read the results of a elementwise operation";
    let description = [{Although it sounds weird, but this is how the simulator implements it.
                        This only works for elementwise operations. }];
    let arguments = (ins AnyMemRef:$outBuffer,
                  Index:$startRow, Index:$startCol);
    let results = (outs Burst:$out_burst);
    let assemblyFormat = [{$outBuffer `,` $startRow `,` $startCol attr-dict `:` 
         type($outBuffer) `,` type($startRow) `,` type($startCol) `->` qualified(type($out_burst))}];
}



// def SimulatorPreloadNoReplacementOp: Hbmpim_Op<"sim_preload_no_replacement", []>{
//     let summary = "Load data that are not being replaced";
//     let description = [{}];
//     let arguments = (ins Index:$startRow, Index:$startCol);
//     let assemblyFormat = [{$startRow `,` $startCol attr-dict `:`  type($startRow) `,` type($startCol)}];
// }



// def SimulatorReadDataOp: Hbmpim_Op<"sim_read_data", []>{
//     let summary = "Simulator function for reading the results of a elementwise operation";
//     let description = [{Although it sounds weird, but this is how the simulator implements it.
//                         This only works for elementwise operations. }];
//     let arguments = (ins 
//                   Index:$startRow, Index:$startCol, DenseI64ArrayAttr:$dimensions);
//     let assemblyFormat = [{ $startRow `,` $startCol attr-dict `:` 
//          type($startRow) `,` type($startCol)}];
// }


// def SimulatorPreloadGemvOp: Hbmpim_Op<"sim_preload_gemv", []>{
//     let summary = "Simulator function for loading data for GEMV-based kernels using the";
//     let description = [{}];
//     let arguments = (ins Index:$startRow, Index:$startCol, DenseI64ArrayAttr:$dimensions);
//     let assemblyFormat = [{$startRow `,` $startCol attr-dict `:` 
//         type($startRow) `,` type($startCol)}];
// }


// def SimulatorReadResultOp: Hbmpim_Op<"sim_read_result", []>{
//     let summary = "Read the result of a **gemv** operation";
//     let description = [{Although it sounds weird, but this is how the simulator implements it.
//                         This only works for gemv. }];
//     let arguments = (ins PimBankTypeAttr:$bank_type, Index:$out_dim,
//                   Index:$base_addr, Index:$startRow, Index:$startCol, DenseI64ArrayAttr:$dimensions);
//     let assemblyFormat = [{$bank_type `,` $out_dim `,` $base_addr `,`
//         $startRow `,` $startCol attr-dict `:` 
//         type($out_dim) `,` type($base_addr) `,` type($startRow) `,` type($startCol)}];
// }

def SetDeviceConfigOp : Hbmpim_Op<"set_dev_config", [Pure]> {
    let summary = "Set the channels, ranks, ..., of the device.";
    let description = [{}];
    let results = (outs DeviceConfiguration:$config);
    let assemblyFormat = " attr-dict `:` qualified(type($config))";
}

def SimulatorSetDeviceConfigOp : Hbmpim_Op<"sim_set_dev_config", []> {
    let summary = "Set the channels, ranks, ..., of the device.";
    let description = [{}];
    let arguments = (ins I64Attr:$numBank, I64Attr:$numChannel, I64Attr:$numRank, I64Attr:$numGrf);
    let assemblyFormat = "attr-dict";
}

def GetPimCmdsOp : Hbmpim_Op<"get_pim_cmds", []> {
    let summary = "Create a vector of operations that are executed by pim units.";
    let description = [{}];
    let arguments = (ins PimKernelTypeAttr:$kerType, Index:$jump_take, Index:$odd_jump, Index:$even_jump);
    let results = (outs PIMCommandsVectorType:$result);
    let assemblyFormat = "$kerType `,` $jump_take `,` $odd_jump `,` $even_jump attr-dict `:` type($jump_take) `,` type($odd_jump) `,` type($even_jump) `->` qualified(type($result))";
}

def GetToggleCondOp : Hbmpim_Op<"get_toggle_cond", []> {
    let summary = "get the toggle condition, based on the type of bank";
    let description = [{}];
    let arguments = (ins PimBankTypeAttr:$pb_type);
    let results = (outs Index:$toggle_cond);
    let assemblyFormat = "$pb_type attr-dict `:` type($toggle_cond)";
}

def PimAddrGenOp: Hbmpim_Op<"pim_addr_gen", []> {
    let summary = "generates address based on channel, rank, bankgroup, bank, row, col";
    let description = [{}];
    let arguments = (ins Index:$channel, Index:$rank, Index:$bankgroup, Index:$bank, 
        Index:$row, Index:$col);
    let results = (outs Index:$address);
    let assemblyFormat = [{$channel `,` $rank `,` $bankgroup `,` $bank `,` $row  `,` $col
        attr-dict `:` type($channel) `,` type($rank) `,` type($bankgroup) `,` type($bank)
         `,` type($row) `,` type($col) `->` type($address)
    
    }];
}

def PimAddrGenSafeOp: Hbmpim_Op<"pim_addr_gen_safe", []> {
    let summary = "generates safe address based on channel, rank, bankgroup, bank, row, col";
    let description = [{}];
    let arguments = (ins Index:$channel, Index:$rank, Index:$bankgroup, Index:$bank, 
        Index:$row, Index:$col);
    let results = (outs Index:$address);
    let assemblyFormat = [{$channel `,` $rank `,` $bankgroup `,` $bank `,` $row  `,` $col
        attr-dict `:` type($channel) `,` type($rank) `,` type($bankgroup) `,` type($bank)
         `,` type($row) `,` type($col) `->` type($address)
    
    }];
}

def SetControlOp: Hbmpim_Op<"set_control", []> {
    let summary = "Set the control register file";
    let description = [{}];
    let arguments = (ins Burst:$burstType, BoolAttr:$pim_op, 
            Index:$toggle_cond, BoolAttr:$grfA_zero, BoolAttr:$grfB_zero);
    let extraClassDeclaration = [{}];
    let assemblyFormat = [{$burstType `,` $pim_op `,` $toggle_cond `,` 
            $grfA_zero `,` $grfB_zero attr-dict `:` 
            qualified(type($burstType)) `,` type($toggle_cond) }];
}

def ParkInOp: Hbmpim_Op<"park_in", []> {
    let summary = "Park in operation";
    let description = [{}];
    let extraClassDeclaration = [{}];
    let assemblyFormat = "attr-dict";
}

def ParkOutOp: Hbmpim_Op<"park_out", []> {
    let summary = "Park out operation";
    let description = [{}];
    let extraClassDeclaration = [{}];
    let assemblyFormat = "attr-dict";
}

def ChangePIMModeOp: Hbmpim_Op<"change_pim_mode", []> {
    let summary = "Change the pim mode in the device operation";
    let description = [{}];
    let extraClassDeclaration = [{}];
    let arguments = (ins DRAMModeAttr:$curMode, DRAMModeAttr:$nextMode);
    let assemblyFormat = "$curMode `,` $nextMode attr-dict";
}

def ProgramCrfOp: Hbmpim_Op<"program_crf", []>{
    let summary = "Program the register file with list of commands.";
    let description = [{}];
    let arguments = (ins PIMCommandsVectorType:$pim_cmds);
    let assemblyFormat = "$pim_cmds attr-dict `:` qualified(type($pim_cmds))";
}

def GetDeviceBurst: Hbmpim_Op<"get_device_burst", []>{
    let summary = "";
    let description = [{}];
    let results = (outs Burst:$burstType);
    let assemblyFormat = "attr-dict `:` qualified(type($burstType))";
}

// def AddTransactionAllOp: Hbmpim_Op<"add_transaction_all", []>{
//     let summary = "Send transaction to all of the banks.";
//     let description = [{}];
//     let arguments = (ins BoolAttr:$is_write, Index:$bg_idx, Index:$bank_idx,
//                     Index:$row, Index:$col, BurstTypeAttr:$bst, BoolAttr:$use_barrier, Index:$num_loop);
//     let assemblyFormat = [{ $is_write `,` $bg_idx `,` $bank_idx `,` $row `,` $col `,` $bst `,` 
//                           $use_barrier `,` $num_loop attr-dict `:` type($bg_idx) `,` type($bank_idx) `,` 
//                           type($row) `,` type($col)  `,` type($num_loop) }];
// }

def AddTransactionOp: Hbmpim_Op<"add_transaction", []>{
    let summary = "Send transaction to all of the banks.";
    let description = [{}];
    let arguments = (ins BoolAttr:$is_write, Index:$addr, Index:$data_idx);
    let assemblyFormat = [{ $is_write `,` $addr `,` $data_idx attr-dict `:` type($addr) `,` type($data_idx) }];
}

def GetNumElementsPerBlocksOp: Hbmpim_Op<"get_num_elements_per_block", []>{
    let summary = "Based on the device configuration (JEDEC_DATA_BUS_BITS and BL and the datatype it returns the number of elements per block)";
    let description = [{}];
    let arguments = (ins DataTypeAttr:$datatype);
    let results = (outs Index:$num);

    let assemblyFormat = "$datatype attr-dict `->` type($num)";
}

def ExecuteElementwiseOp: Hbmpim_Op<"execute_el_wise", []>{
  let summary = "Execute element-wise operations such as Relu, add, mul";
  let description = [{}];
  let arguments = (ins Index: $dim, PimBankTypeAttr:$bankType, PimKernelTypeAttr:$kernelType, 
                   Index:$input0row, Index:$resultRow, Index:$input1row);
  let assemblyFormat = [{$dim `,` $bankType `,` $kernelType `,` $input0row `,` $resultRow `,` $input1row
      attr-dict `:` type($dim) `,` type($input0row) `,` type($resultRow) `,` type($input1row)}];
}



def HbmpimModuleOp : Hbmpim_Op<"module", [
   IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    ] # GraphRegionNoTerminator.traits > {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
}

def HbmpimFuncOp : Hbmpim_Op<"func", [
    HasParent<"HbmpimModuleOp">, AutomaticAllocationScope,
    IsolatedFromAbove, FunctionOpInterface]
  > {
  let summary = "Function executable on samsung Hbmpim";

  let description = [{
  }];

  let arguments = (ins
         StrAttr:$sym_name
  );
  let regions = (region MinSizedRegion<1>:$body);

  let extraClassDeclaration = [{
    StringRef getName() {
      return getSymName();
    }
    FunctionType getFunctionType() {
      return FunctionType::get(getContext(), {}, {});
    }
    void setFunctionTypeAttr(TypeAttr ty) {
      if (ty.getValue() != getFunctionType())
        emitOpError("cannot set type ") << ty;
    }
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return {}; }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return {}; }

    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;

}

def ReturnOp : Hbmpim_Op<"return", [HasParent<"HbmpimFuncOp">, Pure,
                                     Terminator]>, 
    Arguments<(ins)>, Results<(outs)>
      {
  let summary = "Terminator for Hbmpim functions.";
  let description = [{
  }];
  let assemblyFormat = "attr-dict";
}

def LaunchOp: Hbmpim_Op<"launch", [IsolatedFromAbove, SingleBlockImplicitTerminator<"TerminatorOp">]> {
    let summary = "Launch a code kernel on a given workgroup. Input buffers should have been scattered beforehand.";
    let description = [{}];

    let arguments = (ins DeviceConfiguration:$config);
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);

    // let hasVerifier = 1;
    let assemblyFormat = "$config attr-dict `on` qualified(type($config)) $body";

    let extraClassDeclaration = [{
    }];
}

def HbmpimLaunchFuncOp :Hbmpim_Op<"launch_func", []>,
    Arguments<(ins SymbolRefAttr:$kernel, Variadic<AnyType>:$kernelOperands)> {
  let summary = "Launches a Hbmpim kernel instance";
  let description = [{
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "HbmpimFuncOp":$kernelFunc,  
      "ValueRange":$kernelOperands)>
  ];

  let extraClassDeclaration = [{
    /// The name of the kernel's containing module.
    StringAttr getKernelModuleName();

    /// The name of the kernel.
    StringAttr getKernelName();

    /// The number of operands passed to the kernel function.
    unsigned getNumKernelOperands();

    /// The i-th operand passed to the kernel function.
    Value getKernelOperand(unsigned i);
  }];

  let assemblyFormat = [{
        $kernel custom<LaunchFuncOperands>($kernelOperands, type($kernelOperands)) attr-dict
  }];
}

def TerminatorOp: Hbmpim_Op<"terminator", [Terminator]> {
    let summary = "Terminates an `hbmpim.launch` operator region.";
    let description = [{}];

    let arguments = (ins);
    let results = (outs);
    let assemblyFormat = "attr-dict";
}


#endif

//===- Ops.td - Cnm dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_OPS
#define CNM_OPS


include "cinm-mlir/Dialect/Cnm/IR/CnmBase.td"
include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/IR/CommonAttrConstraints.td"


def WorkgroupOp : Cnm_Op<"workgroup", []> {
    let summary = "Allocate a workgroup.";
    let description = [{}];

    let arguments = (ins);
    let results = (outs WorkgroupType:$result);

    let builders = [
        OpBuilder<(ins "ArrayRef<int64_t>":$shape),
          "build($_builder, $_state, WorkgroupType::get($_builder.getContext(), shape));">
    ];

    let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def FreeWorkgroupOp : Cnm_Op<"free_workgroup", []> {
    let summary = "Deallocate a workgroup.";
    let description = [{}];

    let arguments = (ins WorkgroupType:$workgroup);
    let results = (outs);

    let assemblyFormat = "$workgroup attr-dict `:` qualified(type($workgroup))";
}


class InferWorkgroupTypeFromBuffer<string buffer, string wg>
: TypesMatchWith<"workgroup shape matches buffer shape", buffer, wg,
                               "cnm::WorkgroupType::get($_self.getContext(), ::llvm::cast<cnm::BufferType>($_self).getWorkgroupShape())">;

def AllocOp : Cnm_Op<"alloc", [InferWorkgroupTypeFromBuffer<"result", "wg">]> {
    let summary = "Allocate a buffer for a given workgroup.";
    let description = [{}];

    let arguments = (ins WorkgroupType:$wg);
    let results = (outs BufferType:$result);

    let builders = [
        OpBuilder<(ins "ArrayRef<int64_t>":$bufShape, "Type":$elementTy, "Value":$wg, CArg<"int64_t", "0">:$level),
          "build($_builder, $_state, BufferType::get(bufShape, elementTy, wg.getType().cast<cnm::WorkgroupType>().getShape(), level), wg);">
    ];


    let extraClassDeclaration = [{

    }];

    let assemblyFormat = "`(` `)` `for` $wg attr-dict `:` qualified(type($result))";
}

def SetZeroOp: Cnm_Op<"set_zero", [SameOperandsAndResultType]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins BufferType:$buffer);
    let results = (outs BufferType:$result);

    let assemblyFormat = "$buffer attr-dict `:` qualified(type($buffer))";
}

def ScatterOp: Cnm_Op<"scatter", [InferWorkgroupTypeFromBuffer<"buffer", "wg">]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins AnyShaped:$input, BufferType:$buffer, WorkgroupType:$wg, AffineMapAttr:$scatterMap);
    let results = (outs);

    let extraClassDeclaration = [{
        int64_t getTransferCountInItems() {
            auto bufShape = getBuffer().getType().getShape();
            return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
        }
    }];

    let hasVerifier = 1;

    let assemblyFormat = "$input `into` $buffer `[` $scatterMap `]` `of` $wg attr-dict `:` type($input) `into` qualified(type($buffer))";
}

def GatherOp: Cnm_Op<"gather", [InferWorkgroupTypeFromBuffer<"buffer", "wg">, InferTensorTypeAdaptor, DestinationStyleOpInterface]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins BufferType:$buffer, WorkgroupType:$wg, AffineMapAttr:$gatherMap, AnyShaped:$outputBuf);
    let results = (outs Optional<AnyRankedTensor>:$output);

    let extraClassDeclaration = [{
        MutableOperandRange getDpsInitsMutable() {
            return getOutputBufMutable();
        }

        int64_t getTransferCountInItems() {
            auto bufShape = getBuffer().getType().getShape();
            return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
        }
    }];

    let hasVerifier = 1;
    let assemblyFormat = "$buffer `[` $gatherMap `]` `of` $wg attr-dict `into` $outputBuf `:` qualified(type($buffer)) `into` type($outputBuf)";
}

def LaunchOp: Cnm_Op<"launch", [AttrSizedOperandSegments, IsolatedFromAbove, SingleBlockImplicitTerminator<"TerminatorOp">]> {
    let summary = "Launch a code kernel on a given workgroup. Input buffers should have been scattered beforehand.";
    let description = [{}];

    let arguments = (ins WorkgroupType:$wg, Variadic<BufferType>:$inBuffers, Variadic<BufferType>:$outBuffers);
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;
    let assemblyFormat = "$wg `ins` `(` $inBuffers `:` type($inBuffers) `)` `outs` `(` $outBuffers `:` type($outBuffers) `)` attr-dict `on` qualified(type($wg)) $body";

    let extraClassDeclaration = [{
        SmallVector<Value> getParams() {
            SmallVector<Value> result{getInBuffers()};
            result.append(getOutBuffers().begin(), getOutBuffers().end());
            return result;
        }
    }];
}


def ComputeOp: Cnm_Op<"compute", [IsolatedFromAbove, 
    SingleBlockImplicitTerminator<"TerminatorOp">,
    AttrSizedOperandSegments]> {
    let summary = "Transformable operation";
    let description = [{}];

    let arguments = (ins DenseI64ArrayAttr:$workgroupShape,
                         I64Attr:$numInputs,
                         Variadic<Index>:$symbol_bindings,
                         Variadic<AnyShaped>:$buffers,
                         AffineMapArrayAttr:$affineMaps
                         );

    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region SizedRegion<1>:$body);

    let builders = [
        OpBuilder<(ins 
            CArg<"ArrayRef<int64_t>">:$workgroupShape,
            CArg<"ValueRange">:$inputs,
            CArg<"ValueRange">:$inits,
            CArg<"ArrayRef<AffineMap>">:$affineMaps,
            CArg<"ValueRange", "{}">:$symbol_bindings
        )>
    ];

    let skipDefaultBuilders = 1;
    let hasVerifier = 1;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        SmallVector<Value> getInBuffers() {
            if (getNumInputs() == 0)
                return {};
            SmallVector<Value> result;
            result.reserve(getNumInputs());
            auto begin = getBuffers().begin();
            result.append(begin, begin + getNumInputs());
            return result;
        }

        SmallVector<Value> getOutBuffers() {
            if (getNumOutputs() == 0)
                return {};
            SmallVector<Value> result;
            result.reserve(getNumOutputs());
            auto begin = getBuffers().begin() + getNumInputs();
            result.append(getBuffers().begin() + getNumInputs(), getBuffers().end());
            return result;
        }

        uint64_t getNumOutputs() {
            return getBuffers().size() - getNumInputs();
        }

        SmallVector<AffineMap> getInMaps() {
            if (getNumInputs() == 0)
                return {};
            SmallVector<AffineMap> result;
            result.reserve(getNumInputs());
            auto begin = getAffineMaps().getAsValueRange<AffineMapAttr>().begin();
            result.append(begin, begin + getNumInputs());
            return result;
        }

        SmallVector<AffineMap> getOutMaps() {
            if (getNumOutputs() == 0)
                return {};
            SmallVector<AffineMap> result;
            result.reserve(getNumOutputs());
            auto it = getAffineMaps().getAsValueRange<AffineMapAttr>();
            result.append(it.begin() + getNumInputs(), it.end());
            return result;
        }

        template<unsigned N = llvm::CalculateSmallVectorDefaultInlinedElements<AffineMap>::value>
        SmallVector<AffineMap, N> getAffineMapsVec() {
            SmallVector<AffineMap, N> result(getAffineMaps().getAsValueRange<AffineMapAttr>());
            return result;
        }

        MutableArrayRef<BlockArgument> getKernelArgs() {
            return getBody().getArguments();
        }
        
    }];
}


def TerminatorOp: Cnm_Op<"terminator", [Terminator]> {
    let summary = "Terminates an `upmem.launch` operator region.";
    let description = [{}];

    let arguments = (ins);
    let results = (outs);
    let assemblyFormat = "attr-dict";
}


#endif

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
include "mlir/IR/CommonAttrConstraints.td"

def WorkgroupOp : Cnm_Op<"workgroup"> {
    let summary = "";
    let description = [{}];

    let arguments = (ins);
    let results = (outs WorkgroupType:$result);

    let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

class InferWorkgroupTypeFromBuffer<string buffer, string wg>
: TypesMatchWith<"workgroup shape matches buffer shape", buffer, wg,
                               "cnm::WorkgroupType::get($_self.getContext(), ::llvm::cast<cnm::BufferType>($_self).getWorkgroupShape())">;

def AllocOp : Cnm_Op<"alloc", [InferWorkgroupTypeFromBuffer<"result", "wg">]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins WorkgroupType:$wg);
    let results = (outs BufferType:$result);

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

    let arguments = (ins Builtin_RankedTensor:$input, BufferType:$buffer, WorkgroupType:$wg, AffineMapAttr:$scatterMap);
    let results = (outs ScatterTokenType:$token);

    let extraClassDeclaration = [{
        int64_t getTransferCountInItems() {
            auto bufShape = getBuffer().getType().getShape();
            return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
        }
    }];

    let hasVerifier = 1;

    let assemblyFormat = "$input `into` $buffer `[` $scatterMap `]` `of` $wg attr-dict `:` type($input) `into` qualified(type($buffer))";
}

def GatherOp: Cnm_Op<"gather", [InferWorkgroupTypeFromBuffer<"buffer", "wg">]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins BufferType:$buffer, WorkgroupType:$wg, AffineMapAttr:$gatherMap);
    let results = (outs Builtin_RankedTensor:$output, GatherTokenType:$token);

    let extraClassDeclaration = [{
        int64_t getTransferCountInItems() {
            auto bufShape = getBuffer().getType().getShape();
            return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
        }
    }];

    let hasVerifier = 1;
    let assemblyFormat = "$buffer `[` $gatherMap `]` `of` $wg attr-dict `:` qualified(type($buffer)) `into` type($output)";
}

def LaunchOp: Cnm_Op<"launch", [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"TerminatorOp">]> {
    let summary = "Launch a code kernel on a given workgroup. Input buffers should have been scattered beforehand.";
    let description = [{}];

    let arguments = (ins WorkgroupType:$wg, Variadic<BufferType>:$inBuffers, Variadic<BufferType>:$outBuffers);
    let results = (outs LaunchTokenType:$token);
    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;
    let assemblyFormat = "$wg `in` `(` $inBuffers `:` type($inBuffers) `)` `out` `(` $outBuffers `:` type($outBuffers) `)` attr-dict `on` qualified(type($wg)) $body";

    let extraClassDeclaration = [{
        SmallVector<Value> getParams() {
            SmallVector<Value> result{getInBuffers()};
            result.append(getOutBuffers().begin(), getOutBuffers().end());
            return result;
        }
    }];
}

def TerminatorOp: Cnm_Op<"terminator", [Terminator]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins);
    let results = (outs);
    let assemblyFormat = "attr-dict";
}

#endif

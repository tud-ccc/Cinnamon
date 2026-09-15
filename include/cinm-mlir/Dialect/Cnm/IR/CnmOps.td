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
    let summary = "Acquire a workgroup of the accelerator.";
    let description = [{
        Acquires the part of the accelerator described by the result type's
        accelerator attribute, and produces the handle that `cnm.declare_buffer`
        and `cnm.launch` take. Every op that names a workgroup names this
        value, so the handle is what ties a program's buffers and launches to
        one another.

        Acquiring is not free -- on UPMEM it allocates DPUs and loads a
        program onto them -- which is why it is an op with a result rather
        than a property of the launch, and why `--cnm-hoist-workgroups` moves
        it out of loops.

        ```mlir
        %wg = cnm.workgroup : !cnm.workgroup<#upmem.array<1x16x1, #platform>>
        ```
    }];

    let arguments = (ins);
    let results = (outs WorkgroupType:$result);

    let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def FreeWorkgroupOp : Cnm_Op<"free_workgroup", []> {
    let summary = "Release a workgroup.";
    let description = [{
        Releases what `cnm.workgroup` acquired. Nothing may use the workgroup
        afterwards, and neither may the buffers declared for it.
    }];

    let arguments = (ins WorkgroupType:$workgroup);
    let results = (outs);

    let assemblyFormat = "$workgroup attr-dict `:` qualified(type($workgroup))";
}


class InferWorkgroupTypeFromBuffer<string buffer, string wg>
: TypesMatchWith<"workgroup shape matches buffer shape", buffer, wg,
                               "cnm::WorkgroupType::get($_self.getContext(), ::llvm::cast<cnm::BufferType>($_self).getAccelerator())">;

def DeclareBufferOp : Cnm_Op<"declare_buffer", [InferWorkgroupTypeFromBuffer<"result", "wg">, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let summary = "Name a buffer distributed over a workgroup.";
    let description = [{
        Introduces a `!cnm.buffer`: one block of the given shape and element
        type *per leaf* of the workgroup. A buffer is therefore
        `workgroupShape x bufferShape` elements in total, of which a leaf sees
        only its own `bufferShape`, as the memref it receives in a
        `cnm.launch` body.

        It declares rather than allocates: the storage is the accelerator's,
        it exists for as long as the workgroup does, and where in the memory
        hierarchy it lands is the type's `level` and the backend's business.
        So there is no matching free, and no allocation the op could be
        eliminated as dead against -- a buffer that nothing scatters into,
        gathers from or launches over simply disappears.

        ```mlir
        %buf = cnm.declare_buffer() for %wg : !cnm.buffer<1024xi32 on #acc>
        ```
    }];

    let arguments = (ins WorkgroupType:$wg);
    let results = (outs BufferType:$result);

    let assemblyFormat = "`(` `)` `for` $wg attr-dict `:` qualified(type($result))";
}


def ScatterOp: Cnm_Op<"scatter", [InferWorkgroupTypeFromBuffer<"buffer", "wg">]> {
    let summary = "Distribute a host value into a workgroup's buffer.";
    let description = [{
        Fills every leaf's block of `buffer` from `input`, which lives on the
        host. Which host elements a leaf receives is `scatterMap`'s to say:

        ```
        buf[w][i_0..i_{p-1}, rest] = input[map(w, i_0..i_{p-1}) ++ rest]
        ```

        The map is *pointwise*: its domain is the workgroup shape
        `w_0..w_{n-1}` followed by all of the buffer's own dimensions, and it
        has one result per host dimension. So it names a host element for
        every buffer element and leaves nothing implicit, which is what makes
        two transfers comparable without first agreeing on a spelling.

        Nothing here says what moves as one block. That is a question about
        the host value's layout as much as about the map, so a consumer that
        moves whole blocks derives the answer for itself -- see
        cnm::deflateScatterMap in CnmScatterMap.h.

        A scatter need not be injective: two leaves reading the same host
        element is a broadcast, which is how an operand shared across a
        workgroup axis is expressed. It must, however, stay inside the host
        value; the verifier checks this on the map's whole index space.

        ```mlir
        // Leaf (_, d1, d2) receives row d1 of a 16x1024 host matrix.
        cnm.scatter %a into %buf[affine_map<(d0, d1, d2, d3) -> (d1, d3)>] of %wg
            : tensor<16x1024xi32> into !cnm.buffer<1024xi32 on #acc>
        ```
    }];

    let arguments = (ins AnyShaped:$input, BufferType:$buffer, WorkgroupType:$wg, AffineMapAttr:$scatterMap);
    let results = (outs);

    let extraClassDeclaration = [{
        int64_t getTransferCountInItems() {
            auto bufShape = getBuffer().getType().getShape();
            return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
        }
        ShapedType getHostType() {
          return getInput().getType();
        }
        TypedValue<ShapedType> getHostValue() {
          return getInput();
        }
        ::mlir::MutableOperandRange getHostValueMutable() {
          return getInputMutable();
        }
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let assemblyFormat = "$input `into` $buffer `[` $scatterMap `]` `of` $wg attr-dict `:` type($input) `into` qualified(type($buffer))";
}

def GatherOp: Cnm_Op<"gather", [InferWorkgroupTypeFromBuffer<"buffer", "wg">, InferTensorTypeAdaptor, DestinationStyleOpInterface]> {
    let summary = "Collect a workgroup's buffer back into a host value.";
    let description = [{
        The inverse of `cnm.scatter`: `gatherMap` is pointwise and read the
        same way, but the elements travel from the leaves to `outputBuf`.

        The one asymmetry is that a gather map must be injective. Two leaves
        reading the same host element is a broadcast; two leaves *writing* it
        is a race, so a gather has to partition the host value rather than
        cover it. The verifier rejects a map that does not.

        This op is in destination-passing style, so `outputBuf` is both where
        the result goes and what its type is taken from. On tensors it also
        produces a result; on memrefs it writes in place and produces nothing.

        ```mlir
        %y = cnm.gather %buf[affine_map<(d0, d1, d2, d3) -> (d1, d3)>] of %wg into %init
            : !cnm.buffer<64xi32 on #acc> into tensor<16x64xi32>
        ```
    }];

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
        AffineMap getScatterMap() {
          return getGatherMap();
        }
        void setScatterMap(AffineMap map) {
          setGatherMap(map);
        }
        ShapedType getHostType() {
          if (getOutput())
            return getOutput().getType();
          return getOutputBuf().getType();
        }
        TypedValue<ShapedType> getHostValue() {
          return ::llvm::cast<TypedValue<ShapedType>>(getOutputBuf());
        }
        ::mlir::MutableOperandRange getHostValueMutable() {
          return getOutputBufMutable();
        }
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
    let assemblyFormat = "$buffer `[` $gatherMap `]` `of` $wg attr-dict `into` $outputBuf `:` qualified(type($buffer)) `into` type($outputBuf)";
}

def LaunchOp: Cnm_Op<"launch", [AffineScope, AttrSizedOperandSegments, AutomaticAllocationScope, IsolatedFromAbove, SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "Run a kernel on every leaf of a workgroup.";
    let description = [{
        Runs the body once per leaf of `wg`, on that leaf's own blocks of the
        buffers passed in `ins` and `outs`. The body sees each `!cnm.buffer`
        as a plain `memref<bufferShape x elt, level>` -- one leaf's block, not
        the distributed whole -- so it is written as if it were a single
        sequential kernel, and the workgroup axes appear nowhere in it. Scalar
        operands are passed through unchanged.

        The leaves run concurrently and do not communicate: the body may not
        refer to another leaf's block, and the region is IsolatedFromAbove so
        that it cannot reach anything but its arguments. Inputs must have been
        filled by `cnm.scatter` beforehand, and `outs` buffers are read back
        with `cnm.gather` -- there is no other way in or out.

        A launch over a buffer whose level is not the innermost one (MRAM on
        UPMEM) tiles into faster memory inside the body, with
        `cnm.local_transfer`.

        ```mlir
        cnm.launch %wg ins(%a = %bufA : <1024xi32>, %b = %bufB : <1024xi32>)
                       outs(%c = %bufC : <i32>) on !cnm.workgroup<#acc> {
          linalg.contract indexing_maps = [#id, #id, #scalar]
            ins(%a, %b : memref<1024xi32>, memref<1024xi32>)
            outs(%c : memref<i32>)
        }
        ```
    }];

    let arguments = (ins WorkgroupType:$wg, Variadic<AnyType>:$inputs, Variadic<BufferType>:$outBuffers);
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        SmallVector<Value> getParams() {
            SmallVector<Value> result{getInputs()};
            result.append(getOutBuffers().begin(), getOutBuffers().end());
            return result;
        }
    }];
}

def LocalTransferOp: Cnm_Op<"local_transfer"> {
    let summary = "Copy a buffer between memory levels of the device.";
    let description = [{
        Copies `source` into `target`, which must have the same shape and
        element type but generally live in different memory levels of the
        device (their memref memory spaces). The direction is therefore implied
        by the operands and is not spelled out separately.

        This op operates on memrefs, not on `!cnm.buffer`: a `cnm.buffer`
        describes a buffer distributed over a whole workgroup, whereas a
        transfer moves an ordinary, locally-shaped block. It is consequently a
        *device-side* op, appearing inside a `cnm.launch` body, in contrast
        with `cnm.scatter`/`cnm.gather`, which move data between host and
        device. Transferring part of a buffer is expressed with an ordinary
        `memref.subview` on either operand.

        How the transfer is realized (DMA engine, blocked copy, leader tasklet)
        is the business of the conversion to the backend dialect.

        ```mlir
        cnm.local_transfer %mram_tile into %wram_buf
          : memref<64xi32, #upmem.mram> to memref<64xi32, #upmem.wram>
        ```
    }];

    let arguments = (ins Arg<AnyMemRef, "the memref to copy from",
                             [MemReadAt<0, FullEffect>]>:$source,
                         Arg<AnyMemRef, "the memref to copy to",
                             [MemWriteAt<0, FullEffect>]>:$target);
    let results = (outs);

    let hasVerifier = 1;
    // Absorbs memref.cast on either operand, so a transfer keeps the most
    // static type available -- the backend needs static shapes to size the
    // transfer. Same rule memref.copy uses.
    let hasFolder = 1;
    let assemblyFormat = [{
        $source `into` $target attr-dict `:` type($source) `to` type($target)
    }];
}

def CompactBufferOp: Cnm_Op<"compact_buffer"> {
    let summary = "Repack a host value into a contiguous buffer.";
    let description = [{
        Writes every element of `target` from the element of `source` that
        `map` names:

        ```
        target[i0, .., iN] = source[map(i0, .., iN)]
        ```

        so the map's domain is `target`'s index space and its results are
        `source` indices. `target` must be contiguous -- producing something a
        single flat transfer can move is the whole point of the op.

        It exists because a repack is a real cost that has to be attributable.
        Expressed as `memref.alloc` + `memref.copy`, a repack is
        indistinguishable from any other copy, and whether it even appears in a
        measurement depends on a lowering heuristic: MemRefToLLVM turns
        `memref.copy` into `llvm.intr.memcpy` when both sides are contiguous
        and only calls the instrumented `memrefCopy` otherwise, so exactly the
        repacks that have been made tidy are the ones that disappear. Naming
        the operation lets the backend route it somewhere it can be timed, and
        gives the cost model something to predict.

        A repack whose source is a `cinm.static` value costs nothing per
        inference -- it can be hoisted out of the serving loop and amortized
        over the workload's lifetime -- while one on a per-inference operand
        cannot. That difference decides whether the time is discounted, so it
        is carried on the op as the `cinm.static` unit attribute rather than
        being re-derived downstream, where the defining op may no longer be
        reachable.

        ```mlir
        // Pack a K-major tile of a row-major matrix so each staged slice is
        // one contiguous run.
        cnm.compact_buffer %a into %packed
            [affine_map<(c, m, j) -> (m, c * 64 + j)>]
            : memref<4096x4096xi32> into memref<8x8x64xi32>
        ```
    }];

    let arguments = (ins Arg<AnyMemRef, "the memref to repack",
                             [MemReadAt<0, FullEffect>]>:$source,
                         Arg<AnyMemRef, "the contiguous memref to fill",
                             [MemWriteAt<0, FullEffect>]>:$target,
                         AffineMapAttr:$map);
    let results = (outs);

    let hasVerifier = 1;
    // Folds a repack of a repack into one. Chains arise when an operand needs
    // packing for more than one reason -- a strided host layout and a leaf
    // ordering, say -- each decided by a different pass with no view of the
    // other.
    let hasCanonicalizer = 1;
    // Absorbs memref.cast on either operand, so a repack keeps the most static
    // type available -- the backend sizes the copy from it, and the verifier
    // needs a provably contiguous target. Same rule memref.copy and
    // cnm.local_transfer use.
    let hasFolder = 1;
    let assemblyFormat = [{
        $source `into` $target `[` $map `]` attr-dict
            `:` type($source) `into` type($target)
    }];

    let extraClassDeclaration = [{
        /// Whether this repack is amortizable over the serving lifetime, i.e.
        /// whether its source holds the same data on every inference. Set
        /// when the op is created; see cinm::isStaticValue.
        bool isStatic() {
            return (*this)->hasAttr(
                ::mlir::cinm::CinmDialect::STATIC_ATTR_NAME);
        }
    }];
}

def ExpandBufferOp: Cnm_Op<"expand_buffer"> {
    let summary = "Write a contiguous buffer back out to a scattered one.";
    let description = [{
        The inverse of `cnm.compact_buffer`: it walks the *source*'s index
        space and writes each element where `map` says it belongs,

        ```
        target[map(i0, .., iN)] = source[i0, .., iN]
        ```

        so the map's domain is `source`'s index space and its results are
        `target` indices. `source` must be contiguous.

        Both directions are needed because a repack serves transfers that go
        both ways. A scatter reads the host value, so packing it is a read
        through the map -- `compact_buffer`. A gather writes the host value,
        so the buffer the transfer filled has to be written back out through
        the map, which is not the same op with its operands exchanged: that
        would need the map inverted, and these maps carry the floordivs and
        mods of a delinearized workgroup index.

        Same reason as `compact_buffer` for being an op rather than a
        `memref.copy`: a repack is a cost that has to be attributable, and a
        copy between two contiguous memrefs lowers to `llvm.intr.memcpy` and
        disappears from the accounting.

        ```mlir
        // Write per-leaf results, gathered contiguously, back into the
        // partial-sum buffer the host reduction reads.
        cnm.expand_buffer %packed into %partials
            [affine_map<(d, t, j) -> (t, d * 64 + j)>]
            : memref<4x8x64xi32> into memref<8x256xi32>
        ```
    }];

    let arguments = (ins Arg<AnyMemRef, "the contiguous memref to write out",
                             [MemReadAt<0, FullEffect>]>:$source,
                         Arg<AnyMemRef, "the memref to scatter into",
                             [MemWriteAt<0, FullEffect>]>:$target,
                         AffineMapAttr:$map);
    let results = (outs);

    let hasVerifier = 1;
    let assemblyFormat = [{
        $source `into` $target `[` $map `]` attr-dict
            `:` type($source) `into` type($target)
    }];

    let extraClassDeclaration = [{
        /// Whether this repack is amortizable over the serving lifetime. A
        /// gather's destination is written every inference, so this is
        /// normally false; it is carried for symmetry with
        /// `cnm.compact_buffer` and read by the same accounting.
        bool isStatic() {
            return (*this)->hasAttr(
                ::mlir::cinm::CinmDialect::STATIC_ATTR_NAME);
        }
    }];
}

def ReturnOp: Cnm_Op<"return", [Terminator]> {
    let summary = "Terminates a `cnm.launch` region.";
    let description = [{
        Carries no operands: a launch body communicates through its `outs`
        buffers, not through returned values. It is inserted implicitly, so
        it rarely appears in written IR.
    }];

    let arguments = (ins);
    let results = (outs);
    let assemblyFormat = "attr-dict";
}

#endif

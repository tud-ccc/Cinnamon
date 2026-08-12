//===- Types.td - Cnm dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_TYPES
#define CNM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "CnmBase.td"
include "CnmInterfaces.td"

def WorkgroupType : Cnm_Type<"Workgroup", [CnmWorkgroupTypeInterface]> {
  let mnemonic = "workgroup";
  let summary = "A piece of an accelerator, seen as a grid of leaves.";
  let description = [{
    Handle to the part of the accelerator a program runs on, produced by
    `cnm.workgroup`. Its whole content is the accelerator attribute, which is
    also where the shape comes from: `#upmem.array<1x16x1, #platform>` is a
    3-dimensional grid of rank x DPU x tasklet. A single point of that grid is
    a *leaf* -- one block of every buffer, one run of a `cnm.launch` body.

    The shape is a hierarchy rather than a flat count because that is what
    decides cost: two leaves that share a DPU share its memory, and two that
    do not must be reached by separate transfers. Which iteration dimension
    occupies which axis is therefore a real choice, made by
    `--convert-linalg-to-cnm` (see `cnm.workgroup_dim_order`).

    ```mlir
    !cnm.workgroup<#upmem.array<1x16x1, #platform>>
    ```
  }];

  // let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let parameters = (ins "::mlir::cnm::CnmAcceleratorAttrInterface":$accelerator);

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    ArrayRef<int64_t> getShape() const {
      return getAccelerator().getWorkgroupShape();
    }
    int64_t getNumElements() const {
      auto wgShape = getShape();
      return std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
    }
    /// CnmWorkgroupTypeInterface.
    ::llvm::SmallVector<int64_t> getWorkgroupShape() const {
      auto shape = getShape();
      return {shape.begin(), shape.end()};
    }
  }];
}

def BufferType : Cnm_Type<"Buffer", [ShapedTypeInterface]> {
  let mnemonic = "buffer";
  let summary = "One block of storage per leaf of a workgroup.";
  let description = [{
    A buffer gives every leaf of `accelerator`'s workgroup a block of `shape`
    elements of `elementType`, so it holds `workgroupShape x shape` elements
    in total. The printed shape is the *leaf's*, not the whole: a
    `!cnm.buffer<1024xi32 on #upmem.array<1x16x1, #pf>>` is 16 blocks of 1024.

    `shape` is what a `cnm.launch` body sees, as a plain
    `memref<shape x elementType, level>`; the workgroup dimensions appear only
    in the maps of `cnm.scatter`/`cnm.gather`, which is the only place a whole
    buffer is addressed at once. Note that the type implements
    `ShapedTypeInterface` over the leaf's shape for that reason -- `getShape()`
    is one block, and `getWorkgroupShape()` is the rest.

    `level` is the memory level the blocks live at, as a memref memory space
    (`#upmem.mram`, `#upmem.wram`). Null means the backend decides, which is
    what the pipeline produces unless a level was asked for explicitly.

    ```mlir
    !cnm.buffer<64x1024xi32 on #upmem.array<1x16x1, #pf>, #upmem.mram>
    ```
  }];

  let parameters = (
    ins ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "::mlir::cnm::CnmAcceleratorAttrInterface":$accelerator,
    "Attribute":$level
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
    "ArrayRef<int64_t>":$shape,
    "Type": $elementType,
    "::mlir::cnm::CnmAcceleratorAttrInterface":$accelerator,
    CArg<"Attribute", "{}">:$level
    ), [{
      return $_get(elementType.getContext(), shape, elementType, accelerator, level);
    }]>
  ];

  let extraClassDeclaration = [{
    bool hasRank() const { return true; }
    ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> newShape, ::mlir::Type elementType) const {
      auto shape = newShape.value_or(getShape());
      return BufferType::get(getContext(), shape, elementType, getAccelerator(), getLevel());
    }
    ArrayRef<int64_t> getWorkgroupShape() const {
      return getAccelerator().getWorkgroupShape();
    }
    int64_t getItemCount() const {
      auto bufShape = getShape();
      return bufShape.empty() ? 1 : mlir::computeProduct(bufShape);
    }

    int64_t getSizeInBytes() const {
      return getItemCount() * getElementTypeBitWidth() / 8;
    }
  }];
}


#endif

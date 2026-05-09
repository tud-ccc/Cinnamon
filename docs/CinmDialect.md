
# Concepts

- Platform (static description) vs accelerator (instance of the platform)
  - Several accelerators for the same platform can coexist
  - Accelerators have a lifecycle (initialization, destruction)

Platform and accelerator are attribute interfaces.
The implementations are provided by the backend dialects. 
Eg upmem can have the following platform description:
```mlir
#upmem = #upmem.platform<type=v1A, dimensions=8x64>
```
and accelerator:
```mlir
#upmem_4_64_16 = #upmem.array<4x32x16, #upmem>
```
The syntax is entirely up to the given dialect, but here the platform definition means that we have 8 ranks with 64 DPUs each in the system. The v1A type is used to inform hardware parameters (like max number of threads, max DPU memory).

The accelerator declaration specifies that it uses 4 ranks of 32 DPUs, each running 16 concurrent tasklets. The accelerator must have a reference to the platform it derives from.

The accelerator attribute must implement CinmAcceleratorAttrInterface.

If it's a CNM system, it can also implement CnmAcceleratorAttrInterface.

### CINM dialect

#### Operations

#### Tiling


CINM ops can be tiled down, which is necessary to support accelerators with limited memory.
The `--cinm-tiling` pass is used to apply tiling factors and tile down all CINM ops annotated with a `cinm.tile_sizes` attribute. The tile sizes must match the structure expected by the specific tile op. For instance GEMM expects 3 tile sizes (it has 3 tileable dimensions), while element-wise operations expect one tile size. For instance:
```mlir
%C = cinm.op.gemm %A, %B {cinm.tile_sizes = array<i64: 4, 12, 16>} : tensor<8x32xi32>, tensor<32x24xi32> -> tensor<8x24xi32>
```

Tiling factors can be manually annotated, or inferred by the `--cinm-infer-tile-sizes` pass. This pass calls back to the CinmAcceleratorAttrInterface for the accelerator to infer tile sizes most suitable to its hardware. This is an optional method of the interface.

#### Platform assignment

The `--cinm-assign-platforms` pass automates the wrapping of `cinm.op.*` operations into `cinm.compute` regions based on which platforms want to handle them. It operates on `func.func` ops that carry the `cinm.available_platforms` attribute (a list of platform attrs).

For each `cinm.op.*` op in the function, the pass queries every listed platform via the `CinmPlatformAttrInterface::isOffloadingTarget` method. If at least one platform is interested, the op is wrapped in a new `cinm.compute` op whose own `cinm.available_platforms` attribute is set to the interested subset. Ops for which no platform returns true are left as-is.

Example: given a UPMEM platform (which handles `cinm.op.gemm` and `cinm.op.gemv`):
```mlir
// Input
func.func @f(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
  return %r : tensor<8x128xi32>
}

// Output
func.func @f(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<8x128xi32> attributes {cinm.available_platforms = [#upmem]} {
    %0 = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
    cinm.yield %0 : tensor<8x128xi32>
  }
  return %r : tensor<8x128xi32>
}
```

To add support for new ops in a backend platform, implement `isOffloadingTarget` on its `CinmPlatformAttrInterface` attribute. The default returns `false`.

#### Other CINM passes

- `--cinm-unwrap-compute-blocks` removes the `cinm.compute` and `cinm.compute_block` operations by inlining their content region.
- `--cinm-isolate-compute-blocks` and `--cinm-deisolate-compute-blocks` turn `cinm.compute` into `cinm.compute_block` and back


### CNM dialect

CNM is a 


### UPMEM dialect

















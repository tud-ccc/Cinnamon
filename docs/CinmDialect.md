
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

#### Other CINM passes

- `--cinm-unwrap-compute-blocks` removes the `cinm.compute` and `cinm.compute_block` operations by inlining their content region.
- `--cinm-isolate-compute-blocks` and `--cinm-deisolate-compute-blocks` turn `cinm.compute` into `cinm.compute_block` and back


### CNM dialect

CNM is a 


### UPMEM dialect

















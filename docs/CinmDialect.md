
## CINM dialect

The CINM dialect is the IR dialect used as the entry point into the CINM compiler. It contains high-level operations commonly offloaded to CIM or CNM accelerators. Those operations are currently:
- `cinm.op.gemm` — matrix-matrix multiplication (also batched: `cinm.op.batch_gemm`)
- `cinm.op.gemv` — matrix-vector multiplication (also batched: `cinm.op.batch_gemv`)
- `cinm.op.elementwise` — unary or binary element-wise operation; also covers complex activation functions (ReLU, sigmoid, tanh)
- `cinm.op.reduce` — reduction along one dimension using a given binary operator (add, min, max, …)
- `cinm.op.scan` — prefix scan over a tensor, returning a tensor of the same shape
- `cinm.op.transpose` — permutes tensor dimensions according to a given permutation
- `cinm.op.topK` — returns the top-K values (and their indices) of a tensor
- `cinm.op.simSearch` — similarity search between two tensors (cosine or dot-product metric)

Additionally, the CINM dialect provides container ops (`cinm.compute` and `compute_block`) to delimit sections of code that are to be offloaded, along with an abstract interface for platform and accelerator descriptions. Those interfaces allow the backend dialects to surface low-level architectural information to higher level transformations in a cross-cutting, device agnostic way. They are described in the next subsection.


#### Platform and accelerator

A **platform** is a static description of the hardware (topology, memory hierarchy, hardware parameters). An **accelerator** is a concrete instance of a platform — a specific subset of resources to use. Several accelerators for the same platform can coexist, and accelerators have a lifecycle (initialization, destruction).

Both are MLIR attribute interfaces (`CinmPlatformAttrInterface`, `CinmAcceleratorAttrInterface`) implemented by one or more attributes in each backend dialect. For example, the UPMEM dialect provides:
```mlir
#upmem = #upmem.platform<type=v1A, dimensions=8x64>       // platform: 8 ranks × 64 DPUs, v1A hardware params
#upmem_4_64_16 = #upmem.array<4x32x16, #upmem>           // accelerator: 4 ranks × 32 DPUs × 16 tasklets
```
An accelerator attribute must carry a reference to its parent platform. For CNM-based systems, the accelerator can additionally implement `CnmAcceleratorAttrInterface`.


#### Tiling


CINM ops can be tiled down, which is necessary to support accelerators with limited memory.
The `--cinm-tiling` pass is used to apply tiling factors and tile down all CINM ops annotated with a `cinm.tile_sizes` attribute. The tile sizes must match the structure expected by the specific tile op. For instance GEMM expects 3 tile sizes (it has 3 tileable dimensions), while element-wise operations expect one tile size. For instance:
```mlir
%C = cinm.op.gemm %A, %B {cinm.tile_sizes = array<i64: 4, 12, 16>} : tensor<8x32xi32>, tensor<32x24xi32> -> tensor<8x24xi32>
```

Tiling factors can be manually annotated, or inferred by the `--cinm-infer-tile-sizes` pass. This pass calls back to the CinmAcceleratorAttrInterface for the accelerator to infer tile sizes most suitable to its hardware. This is an optional method of the interface.

#### Platform assignment

The `--cinm-assign-platforms` pass automates the wrapping of `cinm.op.*` operations into `cinm.compute` regions based on which platforms want to handle them. It operates on `func.func` ops that carry the `cinm.available_platforms` attribute. Which ops each platform claims is determined via the `CinmPlatformAttrInterface::isOffloadingTarget` hook (default: `false`).


#### Accelerator inference

Once you have a platform assigned, you want to turn that into an accelerator.
This entails inferring accelerator parameters and other program parameters, such as tiling factors for a tiled program. CINM provides a framework to 
perform this inference using Bayesian optimization.

The implementation is split into two:
- A framework component in CINM sources (Dialect/Cinm/AcceleratorInference),
which handles the core exploration logic
- A number of plugins implemented by individual backend dialects. Plugins must implement hooks to describe the configuration space and evaluate individual configurations. See interface InferencePlugin. 

For now, inference is only implemented for UPMEM. You can call it with --upmem-infer-accelerator. 

#### Other CINM passes

- `--cinm-unwrap-compute-blocks` removes the `cinm.compute` and `cinm.compute_block` operations by inlining their content region.
- `--cinm-isolate-compute-blocks` and `--cinm-deisolate-compute-blocks` turn `cinm.compute` into `cinm.compute_block` and back

TODO Georg: document bufferization cleanups and such.

## CNM dialect

CNM is a 


## UPMEM dialect

















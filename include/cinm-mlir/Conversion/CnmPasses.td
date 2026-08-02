//===- Passes.td - Cnm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_CONVERSION_PASSES
#define CNM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

#ifdef CINM_GPU_SUPPORT
def ConvertCnmToGPUPass : Pass<"convert-cnm-to-gpu"> {
    let summary = "";
    let constructor = "mlir::cnm::createConvertCnmToGPUPass()";
    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::bufferization::BufferizationDialect",
        "mlir::gpu::GPUDialect",
        "mlir::memref::MemRefDialect",
        "mlir::tensor::TensorDialect"
    ];
}
#endif

def ConvertLinalgToCnmPass : Pass<"convert-linalg-to-cnm"> {
    let summary = "Distribute a linalg op onto a cnm workgroup";
    let description = [{
      Rewrites a `linalg` op with pure tensor semantics into
      `cnm.workgroup` / `cnm.alloc` / `cnm.scatter` / `cnm.launch` /
      `cnm.gather`, driven entirely by a `cnm.tile_sizes` attribute giving
      one block size per iteration dimension.

      Unlike `--convert-cinm-to-cnm`, this pass takes no decisions: the
      distribution is a mechanical consequence of the block sizes and the
      op's own indexing maps. The op must be inside a region that carries an
      accelerator (`cinm.compute_block`), and the product of the tile counts
      `dimSize / blockSize` must equal the number of workgroup leaves.

      A block size smaller than the extent of a *reduction* dimension spreads
      that reduction across the workgroup: each leaf computes a partial
      result from the reduction identity, and a merge is left on the host.
      See docs/CnmMemoryLevelsDesign.md §G4.
    }];
    let constructor = "mlir::cnm::createConvertLinalgToCnmPass()";
    let dependentDialects = [
        "mlir::cnm::CnmDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
    ];
    let options = [
      Option<"bufferLevel", "cnm-buffer-level", "std::string", [{""}],
             "Name of the memory level to allocate cnm.buffers in, resolved "
             "against the accelerator's platform (e.g. `mram` or `wram` for "
             "UPMEM). The level becomes the buffer type's level and the memory "
             "space of the launch body's memrefs. Empty means unlevelled "
             "buffers, which is what the backend conversion assumes when it "
             "picks the staging itself.">,
      Option<"allowFloatReassociation", "allow-float-reassociation", "bool",
             "false",
             "Permit splitting a floating-point reduction across the "
             "workgroup. Doing so reassociates the sum and changes the "
             "result, so it is off by default rather than something the "
             "search does silently.">,
      ListOption<"perDimAttrs", "per-dim-attrs", "std::string",
             "Names of discardable attributes holding one i64 per iteration "
             "dimension of the op (e.g. `upmem.leaf_tile_sizes`). Splitting a "
             "reduction rewrites the op and adds a dimension, so these are "
             "carried onto the new op with a 1 inserted for it. Without this "
             "a later pass would silently read a stale or missing list.">
    ];
}

def ConvertCnmToUPMEMPass : Pass<"convert-cnm-to-upmem"> {
    let summary = "";
    let constructor = "mlir::cnm::createConvertCnmToUPMEMPass()";
    let options = [
      Option<"kernelModuleName", "kernel-module-name", "std::string",
             [{"dpu_kernels"}],
             "Name of the submodule to create DPU kernels in">,
      Option<"cinm1Codegen", "cinm1-codegen", "bool", "false",
             "Whether to match CINM 1 code generation (no barriers, tasklets don't share buffers)">
    ];
    let dependentDialects = [
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::memref::MemRefDialect",
        "mlir::tensor::TensorDialect",
        "mlir::upmem::UPMEMDialect"
    ];
}

#endif // CNM_CONVERSION_PASSES

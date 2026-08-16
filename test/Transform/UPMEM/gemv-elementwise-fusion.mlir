// RUN: cinm-opt %s --cinm-assign-platforms --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=16,tasklets=1,gemv.M.mram=64,gemv.K.mram=1024,gemv.M.wram=64,gemv.K.wram=128,gemv.order[0]=1,gemv.order[1]=2,elementwise.D0.mram=64,elementwise.D0.wram=64,fuse.gemv->elementwise=2" \
// RUN: | FileCheck %s --check-prefix=FUSED
// RUN: cinm-opt %s --cinm-assign-platforms --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=16,tasklets=1,gemv.M.mram=256,gemv.K.mram=256,gemv.M.wram=64,gemv.K.wram=64,gemv.order[0]=2,gemv.order[1]=1,elementwise.D0.mram=64,elementwise.D0.wram=64,fuse.gemv->elementwise=1" \
// RUN: | FileCheck %s --check-prefix=SPLIT

// `gemv` then elementwise, end to end, on the two configurations that decide
// whether the pair fuses. See docs/LaunchFusionDesign.md.
//
// Linalg fusion cannot help here: it fuses an elementwise *producer* into its
// consumer, and the elementwise is the consumer. So the pair reaches
// --convert-linalg-to-cnm as two ops and becomes two launches, with the
// intermediate travelling leaf -> host -> leaf in between.
//
// Whether that round trip is real is a property of the configuration, not of
// the program, which is why --cnm-fuse-launches only recognises it and never
// forces it. `fuse.gemv->elementwise` is the search parameter that says which
// of the two a configuration is: 1 is "not fused", and the space accepts a
// higher value for the first configuration only.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// K is not split: 1024/64 = 16 M-tiles fill the 16-leaf workgroup on their own,
// so every leaf holds a *whole* block of the result, and it is the same block
// the elementwise wants. The gather and the scatter cancel.
//
// One dispatch, one kernel, and the two computations in it back to back on an
// MRAM buffer that never leaves the DPU.
// FUSED-LABEL: func.func @gemv_4MB
// FUSED:       upmem.alloc_dpus
// FUSED-NOT:   upmem.alloc_dpus
// FUSED:       upmem.dpu_program
// FUSED-NOT:   upmem.dpu_program

// K is split 4 ways: each leaf holds a partial sum over a quarter of K, the
// gather brings back 4096 partials and the host reduces them. No leaf ever
// holds a finished block, so there is nothing to fuse and both dispatches
// stay. The partials come back shaped 4x16x64 rather than 4x4x256 because M
// is staged in 64-row chunks, and the buffers carrying a staged dimension are
// cut along it -- the same elements, grouped by what a transfer moves.
// A leaf's partials do not sit together in that 4x16x64 buffer -- K is what
// the leaves differ in and it is the outermost dimension -- so the gather
// fills a buffer laid out leaf by leaf and cnm.expand_buffer writes it back
// out. The alternative is a transfer of one block per leaf, which the SDK's
// scatter/gather API mishandles once a DPU's blocks stop ascending.
// SPLIT-LABEL: func.func @gemv_4MB
// SPLIT:       upmem.alloc_dpus
// SPLIT:       upmem.gather_from_array {{.*}} : memref<4x4x1x4x1x64xi32>
// SPLIT:       cnm.expand_buffer {{.*}} : memref<4x4x1x4x1x64xi32> into memref<4x16x64xi32>
// SPLIT:       upmem.alloc_dpus
// SPLIT:       upmem.dpu_program
// SPLIT:       upmem.dpu_program

func.func @gemv_4MB(%A: tensor<1024x1024xi32>, %x: tensor<1024xi32>, %c: i32) -> tensor<1024xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %cs = tensor.splat %c : tensor<1024xi32>
  %r = cinm.compute -> tensor<1024xi32> {
    %g = cinm.op.gemv %A, %x : tensor<1024x1024xi32>, tensor<1024xi32> -> tensor<1024xi32>
    %e = cinm.op.elementwise mul %g, %cs : tensor<1024xi32>
    cinm.yield %e : tensor<1024xi32>
  }
  return %r : tensor<1024xi32>
}

// Unrolling schedule for roberta_base.mlir, applied by the front end's first
// stage through --transform-preload-library.
//
// It lives beside the model rather than inside it because the payload has to
// stay free of transform ops all the way to code generation: cinm-translate
// does not register the transform dialect, so a module that still carries
// its schedule cannot be turned into host code. Preloading keeps the schedule
// out of the module it rewrites.
//
// The per-head loop is fully unrolled: its blocks read slices of Q, K and V
// at constant offsets, which the graph needs as distinct members. The layer
// loop stays rolled: each layer's weights are slices of the static stacks
// at the loop index, which the compute graph treats as one block per kernel
// that runs once per layer with every layer's weight resident
// (resolveStaticSlice), so unrolling it would only multiply the host code
// by the layer count. The two embedding gathers stay rolled too: they hold
// only slice ops. Loops are matched by attribute rather than position, so
// adding a loop elsewhere in the function does not silently retarget the
// unroll.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %head_loop = transform.structured.match ops{["scf.for"]}
                attributes{unroll_heads} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %head_loop { factor = 12 } : !transform.any_op
    transform.yield
  }
}

// Unrolling schedule for roberta_base.mlir, applied by the front end's first
// stage through --transform-preload-library.
//
// It lives beside the model rather than inside it because the payload has to
// stay free of transform ops all the way to code generation: cinm-translate
// does not register the transform dialect, so a module that still carries
// its schedule cannot be turned into host code. Preloading keeps the schedule
// out of the module it rewrites.
//
// --cinm-complete-compute-graph can only connect the graph if no compute op
// is left under a loop, so the layer loop and the per-head loop are fully
// unrolled. The two embedding gathers stay rolled: they hold only slice ops.
// Loops are matched by attribute rather than position, so adding a loop
// elsewhere in the function does not silently retarget the unrolls.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %layer_loop = transform.structured.match ops{["scf.for"]}
                attributes{unroll_layers} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %layer_loop { factor = 12 } : !transform.any_op

    %head_loop = transform.structured.match ops{["scf.for"]}
                attributes{unroll_heads} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %head_loop { factor = 12 } : !transform.any_op
    transform.yield
  }
}

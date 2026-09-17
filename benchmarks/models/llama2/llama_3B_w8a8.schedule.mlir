// Unrolling schedule for llama_3B_w8a8.mlir, applied by the front end's
// first stage through --transform-preload-library (see
// roberta_base.schedule.mlir for why it lives beside the model).
//
// --cinm-complete-compute-graph can only connect the graph if no compute op
// is left under a loop, so the layer loop and the per-head loop of @imha are
// fully unrolled. Loops are matched by attribute rather than position.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %layer_loop = transform.structured.match ops{["scf.for"]}
                attributes{unroll_layers} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %layer_loop { factor = 26 } : !transform.any_op

    %head_loop = transform.structured.match ops{["scf.for"]}
                attributes{unroll_heads} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %head_loop { factor = 32 } : !transform.any_op
    transform.yield
  }
}

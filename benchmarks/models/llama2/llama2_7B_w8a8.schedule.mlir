// Unrolling schedule for llama2_7B_w8a8.mlir, applied by the front end's
// first stage through --transform-preload-library (see
// roberta_base.schedule.mlir for why it lives beside the model).
//
// The per-head loop of @imha is fully unrolled: its blocks read slices of
// the key and value caches at constant offsets, which the graph needs as
// distinct members. The layer loop stays rolled: each layer's weight is the
// slice of the static stack at the loop index, which the compute graph
// treats as one block that runs once per layer with every layer's weight
// resident (resolveStaticSlice), so unrolling it would only multiply the
// host code by the layer count. Loops are matched by attribute rather than
// position.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %head_loop = transform.structured.match ops{["scf.for"]}
                attributes{unroll_heads} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %head_loop { factor = 32 } : !transform.any_op
    transform.yield
  }
}

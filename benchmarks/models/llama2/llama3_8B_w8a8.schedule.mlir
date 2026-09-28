// Unrolling schedule for the llama3_8B_w8a8_n<N>.mlir variants (the
// Makefile points every one of them here), applied by the front end's first
// stage through --transform-preload-library, as llama2_7B_w8a8.schedule.mlir
// is for its model.
//
// The per-query-head loop of @igqa is fully unrolled: its blocks read the
// key/value slices of their head's group at constant offsets, which the
// graph needs as distinct members. The layer loop stays rolled, for the
// reason llama2_7B_w8a8.schedule.mlir gives.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %head_loop = transform.structured.match ops{["scf.for"]}
                attributes{unroll_heads} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %head_loop { factor = 32 } : !transform.any_op
    transform.yield
  }
}

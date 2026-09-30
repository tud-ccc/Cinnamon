// Unrolling schedule for the f32 models llama2_110M.mlir and
// llama2_110M_fusedqkv.mlir (the Makefile points both at this file), applied
// by the front end's first stage through --transform-preload-library: the
// payload has to stay free of transform ops all the way to code generation,
// because cinm-translate does not register the transform dialect.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]}
                attributes{sym_name = "forward"} in %root
        : (!transform.any_op) -> !transform.any_op

    // Returns all 6 scf.for ops in pre-order; the layer loop is first
    %loops = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op

    %a_loop, %layer_loop  =
        transform.split_handle %loops {overflow_result = 1}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Full unroll: trip count is 6 (0 to 6 step 1)
    transform.loop.unroll %layer_loop { factor = 6 } : !transform.any_op

    // Also unroll the head loop of @mha (0 to 768 step 48), so that after
    // inlining no compute op is left under a loop and
    // --cinm-complete-compute-graph can connect the whole graph.
    %mha = transform.structured.match ops{["func.func"]}
                attributes{sym_name = "mha"} in %root
        : (!transform.any_op) -> !transform.any_op
    %mha_loops = transform.structured.match ops{["scf.for"]} in %mha
        : (!transform.any_op) -> !transform.any_op
    // The loops come in post-order: the nested mask loop first, the head
    // loop in the overflow handle.
    %mask_loop, %head_loop =
        transform.split_handle %mha_loops {overflow_result = 1}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.loop.unroll %head_loop { factor = 16 } : !transform.any_op
    transform.yield
  }
}

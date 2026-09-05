# TIR those decisions produced, for ('geva', 16777216, 1, 1) (reproduced).
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((16777216,), "int32"), B: T.Buffer((16777216,), "int32"), C: T.Buffer((16777216,), "int32"), ALPHA: T.Buffer((1,), "int32"), BETA: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A", "B"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_local = T.alloc_buffer((16777216,), "int32", scope="local")
            A_local = T.alloc_buffer((16777216,), "int32", scope="local")
            B_local = T.alloc_buffer((16777216,), "int32", scope="local")
            alpha_val: T.int32 = ALPHA[0]
            beta_val: T.int32 = BETA[0]
            for i_0 in T.thread_binding(2048, thread="blockIdx.x", annotations={"bank": T.int64(1), "pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(0)}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(64):
                        for ax0 in range(8):
                            with T.block("A_local"):
                                v0 = T.axis.spatial(16777216, i_0 * 8192 + i_1 * 512 + i_2 * 8 + ax0)
                                T.reads(A[v0])
                                T.writes(A_local[v0])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                A_local[v0] = A[v0]
                        for ax0 in range(8):
                            with T.block("B_local"):
                                v0 = T.axis.spatial(16777216, i_0 * 8192 + i_1 * 512 + i_2 * 8 + ax0)
                                T.reads(B[v0])
                                T.writes(B_local[v0])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                B_local[v0] = B[v0]
                        for i_3, i_4 in T.grid(4, 2):
                            with T.block("C"):
                                v_i = T.axis.spatial(16777216, i_0 * 8192 + i_1 * 512 + i_2 * 8 + i_3 * 2 + i_4)
                                T.reads(A_local[v_i], B_local[v_i])
                                T.writes(C_local[v_i])
                                T.block_attr({"meta_schedule.tiling_structure": "SSSSS"})
                                C_local[v_i] = alpha_val * A_local[v_i] + beta_val * B_local[v_i]
                        for ax0 in range(8):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(16777216, i_0 * 8192 + i_1 * 512 + i_2 * 8 + ax0)
                                T.reads(C_local[v0])
                                T.writes(C[v0])
                                C[v0] = C_local[v0]

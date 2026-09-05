# TIR those decisions produced, for ('mmtv', 256, 512, 512) (published).
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((256, 512, 512), "int32"), B: T.Buffer((256, 512), "int32"), C: T.Buffer((256, 512), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_local = T.alloc_buffer((256, 512), "int32", scope="local")
            A_local = T.alloc_buffer((256, 512, 512), "int32", scope="local")
            B_local = T.alloc_buffer((256, 512), "int32", scope="local")
            for i_0 in T.thread_binding(256, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
                for j_0 in T.thread_binding(8, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                    for j_1 in T.thread_binding(16, thread="threadIdx.x"):
                        for i_1, i_2, j_2 in T.grid(1, 1, 4):
                            for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(1, 1, 1, 1):
                                with T.block("C_init"):
                                    v_i = T.axis.spatial(256, i_0 + i_1 + i_2 + i_3_init + i_4_init)
                                    v_j = T.axis.spatial(512, j_0 * 64 + j_1 * 4 + j_2 + j_3_init + j_4_init)
                                    T.reads()
                                    T.writes(C_local[v_i, v_j])
                                    T.block_attr({"meta_schedule.tiling_structure": "SSSRSRSR"})
                                    C_local[v_i, v_j] = 0
                            for k_0 in range(8):
                                for ax0_ax1_ax2_fused in range(64):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(256, i_0)
                                        v1 = T.axis.spatial(512, j_0 * 64 + j_1 * 4 + j_2)
                                        v2 = T.axis.spatial(512, k_0 * 64 + ax0_ax1_ax2_fused)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0_ax1_fused in range(64):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(256, i_0)
                                        v1 = T.axis.spatial(512, k_0 * 64 + ax0_ax1_fused)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for i_3, j_3, k_1, i_4, j_4, k_2 in T.grid(1, 1, 8, 1, 1, 8):
                                    with T.block("C_update"):
                                        v_i = T.axis.spatial(256, i_0 + i_1 + i_2 + i_3 + i_4)
                                        v_j = T.axis.spatial(512, j_0 * 64 + j_1 * 4 + j_2 + j_3 + j_4)
                                        v_k = T.axis.reduce(512, k_0 * 64 + k_1 * 8 + k_2)
                                        T.reads(C_local[v_i, v_j], A_local[v_i, v_j, v_k], B_local[v_i, v_k])
                                        T.writes(C_local[v_i, v_j])
                                        T.block_attr({"meta_schedule.tiling_structure": "SSSRSRSR"})
                                        C_local[v_i, v_j] = C_local[v_i, v_j] + A_local[v_i, v_j, v_k] * B_local[v_i, v_k]
                            for ax0, ax1 in T.grid(1, 1):
                                with T.block("C_local"):
                                    v0 = T.axis.spatial(256, i_0 + ax0)
                                    v1 = T.axis.spatial(512, j_0 * 64 + j_1 * 4 + j_2 + ax1)
                                    T.reads(C_local[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_local[v0, v1]

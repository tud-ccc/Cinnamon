# TIR those decisions produced, for ('ttv', 32, 64, 512) (reproduced).
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((32, 64, 512), "int32"), B: T.Buffer((512,), "int32"), C: T.Buffer((32, 64), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_rf_global = T.alloc_buffer((32, 8, 64), "int32")
            C_rf_global_local = T.alloc_buffer((32, 8, 64), "int32", scope="local")
            A_local = T.alloc_buffer((32, 64, 512), "int32", scope="local")
            B_local = T.alloc_buffer((512,), "int32", scope="local")
            for k_0 in T.thread_binding(8, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
                for i_0 in T.thread_binding(32, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                    for j_0 in T.thread_binding(2, thread="blockIdx.z", annotations={"bank": T.int64(1)}):
                        for j_1 in T.thread_binding(16, thread="threadIdx.x"):
                            for i_1, i_2, j_2 in T.grid(1, 1, 1):
                                for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(1, 1, 1, 2):
                                    with T.block("C_rf_init"):
                                        v_i = T.axis.spatial(32, i_0 + i_1 + i_2 + i_3_init + i_4_init)
                                        v_j = T.axis.spatial(64, j_0 * 32 + j_1 * 2 + j_2 * 2 + j_3_init * 2 + j_4_init)
                                        vk_0 = T.axis.spatial(8, k_0)
                                        T.reads()
                                        T.writes(C_rf_global_local[v_i, vk_0, v_j])
                                        T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                        C_rf_global_local[v_i, vk_0, v_j] = 0
                                for k_1_0, i_3, j_3, k_1_1 in T.grid(4, 1, 1, 1):
                                    for ax0, ax1, ax2 in T.grid(1, 2, 16):
                                        with T.block("A_local"):
                                            v0 = T.axis.spatial(32, i_0 + ax0)
                                            v1 = T.axis.spatial(64, j_0 * 32 + j_1 * 2 + ax1)
                                            v2 = T.axis.spatial(512, k_0 * 64 + k_1_0 * 16 + ax2)
                                            T.reads(A[v0, v1, v2])
                                            T.writes(A_local[v0, v1, v2])
                                            T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                            A_local[v0, v1, v2] = A[v0, v1, v2]
                                    for ax0 in range(16):
                                        with T.block("B_local"):
                                            v0 = T.axis.spatial(512, k_0 * 64 + k_1_0 * 16 + ax0)
                                            T.reads(B[v0])
                                            T.writes(B_local[v0])
                                            T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                            B_local[v0] = B[v0]
                                    for i_4, j_4, k_1_2 in T.grid(1, 2, 16):
                                        with T.block("C_rf_update"):
                                            v_i = T.axis.spatial(32, i_0 + i_1 + i_2 + i_3 + i_4)
                                            v_j = T.axis.spatial(64, j_0 * 32 + j_1 * 2 + j_2 * 2 + j_3 * 2 + j_4)
                                            vk_0 = T.axis.spatial(8, k_0)
                                            vk_1 = T.axis.reduce(64, k_1_0 * 16 + k_1_1 * 16 + k_1_2)
                                            T.reads(C_rf_global_local[v_i, vk_0, v_j], A_local[v_i, v_j, vk_0 * 64 + vk_1], B_local[vk_0 * 64 + vk_1])
                                            T.writes(C_rf_global_local[v_i, vk_0, v_j])
                                            T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                            C_rf_global_local[v_i, vk_0, v_j] = C_rf_global_local[v_i, vk_0, v_j] + A_local[v_i, v_j, vk_0 * 64 + vk_1] * B_local[vk_0 * 64 + vk_1]
                                for ax0, ax1, ax2 in T.grid(1, 1, 2):
                                    with T.block("C_rf_global_local"):
                                        v0 = T.axis.spatial(32, i_0 + ax0)
                                        v1 = T.axis.spatial(8, k_0 + ax1)
                                        v2 = T.axis.spatial(64, j_0 * 32 + j_1 * 2 + ax2)
                                        T.reads(C_rf_global_local[v0, v1, v2])
                                        T.writes(C_rf_global[v0, v1, v2])
                                        C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
            for i, j in T.grid(32, 64):
                with T.block("C_init"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads()
                    T.writes(C[v_i, v_j])
                    T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                    C[v_i, v_j] = 0
                for k_0 in range(8):
                    with T.block("C_update"):
                        vk_0, v_i, v_j = T.axis.remap("RSS", [k_0, i, j])
                        T.reads(C[v_i, v_j], C_rf_global[v_i, vk_0, v_j])
                        T.writes(C[v_i, v_j])
                        T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                        C[v_i, v_j] = C[v_i, v_j] + C_rf_global[v_i, vk_0, v_j]

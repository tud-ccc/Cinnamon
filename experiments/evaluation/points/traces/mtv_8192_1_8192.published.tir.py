# TIR those decisions produced, for ('mtv', 8192, 1, 8192) (published).
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((8192, 8192), "int32"), B: T.Buffer((8192,), "int32"), C: T.Buffer((8192,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_rf_global = T.alloc_buffer((32, 8192), "int32")
            C_rf_global_local = T.alloc_buffer((32, 8192), "int32", scope="local")
            A_local = T.alloc_buffer((8192, 8192), "int32", scope="local")
            B_local = T.alloc_buffer((8192,), "int32", scope="local")
            for k_0 in T.thread_binding(32, thread="blockIdx.x", annotations={"bank": T.int64(1), "pragma_auto_unroll_max_step": T.int64(16), "pragma_unroll_explicit": T.int64(0)}):
                for i_0 in T.thread_binding(64, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                    for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                        for i_2 in range(2):
                            for i_3_init, i_4_init in T.grid(4, 1):
                                with T.block("C_rf_init"):
                                    v_i = T.axis.spatial(8192, i_0 * 128 + i_1 * 8 + i_2 * 4 + i_3_init + i_4_init)
                                    vk_0 = T.axis.spatial(32, k_0)
                                    T.reads()
                                    T.writes(C_rf_global_local[vk_0, v_i])
                                    T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                    C_rf_global_local[vk_0, v_i] = 0
                            for k_1_0 in range(8):
                                for ax0, ax1 in T.grid(4, 32):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(8192, i_0 * 128 + i_1 * 8 + i_2 * 4 + ax0)
                                        v1 = T.axis.spatial(8192, k_0 * 256 + k_1_0 * 32 + ax1)
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0 in range(32):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(8192, k_0 * 256 + k_1_0 * 32 + ax0)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for i_3 in range(4):
                                    for k_1_1, i_4, k_1_2 in T.grid(2, 1, 16):
                                        with T.block("C_rf_update"):
                                            v_i = T.axis.spatial(8192, i_0 * 128 + i_1 * 8 + i_2 * 4 + i_3 + i_4)
                                            vk_0 = T.axis.spatial(32, k_0)
                                            vk_1 = T.axis.reduce(256, k_1_0 * 32 + k_1_1 * 16 + k_1_2)
                                            T.reads(C_rf_global_local[vk_0, v_i], A_local[v_i, vk_0 * 256 + vk_1], B_local[vk_0 * 256 + vk_1])
                                            T.writes(C_rf_global_local[vk_0, v_i])
                                            T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                            C_rf_global_local[vk_0, v_i] = C_rf_global_local[vk_0, v_i] + A_local[v_i, vk_0 * 256 + vk_1] * B_local[vk_0 * 256 + vk_1]
                                    for ax0, ax1 in T.grid(1, 1):
                                        with T.block("C_rf_global_local"):
                                            v0 = T.axis.spatial(32, k_0 + ax0)
                                            v1 = T.axis.spatial(8192, i_0 * 128 + i_1 * 8 + i_2 * 4 + i_3 + ax1)
                                            T.reads(C_rf_global_local[v0, v1])
                                            T.writes(C_rf_global[v0, v1])
                                            C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
            for i in T.serial(8192, annotations={"pragma_auto_unroll_max_step": T.int64(16), "pragma_unroll_explicit": T.int64(0)}):
                with T.block("C_init"):
                    v_i = T.axis.spatial(8192, i)
                    T.reads()
                    T.writes(C[v_i])
                    T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                    C[v_i] = 0
                for k_0 in range(32):
                    with T.block("C_update"):
                        vk_0, v_i = T.axis.remap("RS", [k_0, i])
                        T.reads(C[v_i], C_rf_global[vk_0, v_i])
                        T.writes(C[v_i])
                        T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                        C[v_i] = C[v_i] + C_rf_global[vk_0, v_i]

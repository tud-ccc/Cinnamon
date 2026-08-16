# TIR those decisions produced, for ('red', 524288, 1, 1) (published).
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((524288,), "int64"), B: T.Buffer((1,), "int64")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            B_rf_global = T.alloc_buffer((128, 1), "int64")
            B_rf_global_rf_shared = T.alloc_buffer((8, 128, 1), "int64", scope="shared")
            B_rf_global_rf_shared_local = T.alloc_buffer((8, 128, 1), "int64", scope="local")
            A_local = T.alloc_buffer((524288,), "int64", scope="local")
            for i_0 in T.thread_binding(128, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
                for i_1_0 in T.thread_binding(8, thread="threadIdx.x"):
                    with T.block("C_rf_rf_init"):
                        vi_0, vi_1_0 = T.axis.remap("SS", [i_0, i_1_0])
                        T.reads()
                        T.writes(B_rf_global_rf_shared_local[vi_1_0, vi_0, 0])
                        T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SRRR"})
                        B_rf_global_rf_shared_local[vi_1_0, vi_0, 0] = T.int64(0)
                    for i_1_1_0 in range(8):
                        for i_1_1_1 in range(2):
                            for ax0 in range(32):
                                with T.block("A_local"):
                                    v0 = T.axis.spatial(524288, i_0 * 4096 + i_1_0 * 512 + i_1_1_0 * 64 + i_1_1_1 * 32 + ax0)
                                    T.reads(A[v0])
                                    T.writes(A_local[v0])
                                    A_local[v0] = A[v0]
                            for i_1_1_2 in range(32):
                                with T.block("C_rf_rf_update"):
                                    vi_0, vi_1_0 = T.axis.remap("SS", [i_0, i_1_0])
                                    vi_1_1 = T.axis.reduce(512, i_1_1_0 * 64 + i_1_1_1 * 32 + i_1_1_2)
                                    T.reads(B_rf_global_rf_shared_local[vi_1_0, vi_0, 0], A_local[vi_0 * 4096 + (vi_1_0 * 512 + vi_1_1)])
                                    T.writes(B_rf_global_rf_shared_local[vi_1_0, vi_0, 0])
                                    T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SRRR"})
                                    B_rf_global_rf_shared_local[vi_1_0, vi_0, 0] = B_rf_global_rf_shared_local[vi_1_0, vi_0, 0] + A_local[vi_0 * 4096 + (vi_1_0 * 512 + vi_1_1)]
                        for ax0, ax1, ax2 in T.grid(1, 1, 1):
                            with T.block("B_rf_global_rf_shared_local"):
                                v0 = T.axis.spatial(8, i_1_0 + ax0)
                                v1 = T.axis.spatial(128, i_0 + ax1)
                                v2 = T.axis.spatial(1, ax2)
                                T.reads(B_rf_global_rf_shared_local[v0, v1, v2])
                                T.writes(B_rf_global_rf_shared[v0, v1, v2])
                                B_rf_global_rf_shared[v0, v1, v2] = B_rf_global_rf_shared_local[v0, v1, v2]
                    with T.block("C_rf"):
                        vi_1_0, vi_0 = T.axis.remap("RS", [i_1_0, i_0])
                        T.reads(B_rf_global_rf_shared[vi_1_0, vi_0, 0])
                        T.writes(B_rf_global[vi_0, 0])
                        T.block_attr({"meta_schedule.meta_schedule_cross_thread_reduction_block": T.int64(1)})
                        with T.init():
                            B_rf_global[vi_0, 0] = T.int64(0)
                        B_rf_global[vi_0, 0] = B_rf_global[vi_0, 0] + B_rf_global_rf_shared[vi_1_0, vi_0, 0]
            with T.block("C_init"):
                T.reads()
                T.writes(B[0])
                T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                B[0] = T.int64(0)
            for i_0 in range(128):
                with T.block("C_update"):
                    vi_0 = T.axis.reduce(128, i_0)
                    T.reads(B[0], B_rf_global[vi_0, 0])
                    T.writes(B[0])
                    T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                    B[0] = B[0] + B_rf_global[vi_0, 0]

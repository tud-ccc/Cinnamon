# Copied verbatim from ./results/tuned_modules/atim_red_67108864_1_1.py,
# the schedule measured for ('red', 67108864, 1, 1) (published).
from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_atim_red_67108864_1_1:
    @T.prim_func
    def main(A: T.Buffer((67108864,), "int64"), B: T.Buffer((1,), "int64")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            B_rf_global = T.alloc_buffer((2048, 1), "int64")
            B_rf_global_rf_shared = T.alloc_buffer((16, 2048, 1), "int64", scope="shared")
            B_rf_global_rf_shared_local = T.alloc_buffer((16, 2048, 1), "int64", scope="local")
            A_local = T.alloc_buffer((67108864,), "int64", scope="local")
            for i_0 in T.thread_binding(2048, thread="blockIdx.x", annotations={"bank": T.int64(1), "pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(0)}):
                for i_1_0 in T.thread_binding(16, thread="threadIdx.x"):
                    with T.block("C_rf_rf_init"):
                        vi_0, vi_1_0 = T.axis.remap("SS", [i_0, i_1_0])
                        T.reads()
                        T.writes(B_rf_global_rf_shared_local[vi_1_0, vi_0, 0])
                        T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SRRR"})
                        B_rf_global_rf_shared_local[vi_1_0, vi_0, 0] = T.int64(0)
                    for i_1_1_0, i_1_1_1 in T.grid(16, 1):
                        for ax0 in range(128):
                            with T.block("A_local"):
                                v0 = T.axis.spatial(67108864, i_0 * 32768 + i_1_0 * 2048 + i_1_1_0 * 128 + ax0)
                                T.reads(A[v0])
                                T.writes(A_local[v0])
                                A_local[v0] = A[v0]
                        for i_1_1_2 in range(128):
                            with T.block("C_rf_rf_update"):
                                vi_0, vi_1_0 = T.axis.remap("SS", [i_0, i_1_0])
                                vi_1_1 = T.axis.reduce(2048, i_1_1_0 * 128 + i_1_1_1 * 128 + i_1_1_2)
                                T.reads(B_rf_global_rf_shared_local[vi_1_0, vi_0, 0], A_local[vi_0 * 32768 + (vi_1_0 * 2048 + vi_1_1)])
                                T.writes(B_rf_global_rf_shared_local[vi_1_0, vi_0, 0])
                                T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SRRR"})
                                B_rf_global_rf_shared_local[vi_1_0, vi_0, 0] = B_rf_global_rf_shared_local[vi_1_0, vi_0, 0] + A_local[vi_0 * 32768 + (vi_1_0 * 2048 + vi_1_1)]
                        for ax0, ax1, ax2 in T.grid(1, 1, 1):
                            with T.block("B_rf_global_rf_shared_local"):
                                v0 = T.axis.spatial(16, i_1_0 + ax0)
                                v1 = T.axis.spatial(2048, i_0 + ax1)
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
            for i_0 in T.serial(2048, annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(0)}):
                with T.block("C_update"):
                    vi_0 = T.axis.reduce(2048, i_0)
                    T.reads(B[0], B_rf_global[vi_0, 0])
                    T.writes(B[0])
                    T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                    B[0] = B[0] + B_rf_global[vi_0, 0]
# from tvm import tir
def apply_trace_atim_red_67108864_1_1(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  l2, = sch.get_loops(block=b0)
  v3, v4 = sch.sample_perfect_tile2(loop=l2, n=2, min_n_splits=2, max_n_splits=2048, decision=[2048, 32768])
  l5, l6 = sch.split(loop=l2, factors=[v3, v4], preserve_unit_iters=True)
  b7 = sch.rfactor(loop=l5, factor_axis=0, mem_scope="global")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
  sch.annotate(block_or_loop=b7, ann_key="meta_schedule.meta_schedule_rfactor_producer_block", ann_val=1)
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.meta_schedule_rfactor_consumer_block", ann_val=1)
  b8 = sch.get_block(name="C_rf", func_name="main")
  l9, l10 = sch.get_loops(block=b8)
  v11, v12 = sch.sample_perfect_tile2(loop=l10, n=2, min_n_splits=2, max_n_splits=24, decision=[16, 2048])
  l13, l14 = sch.split(loop=l10, factors=[v11, v12], preserve_unit_iters=True)
  b15 = sch.rfactor(loop=l13, factor_axis=0, mem_scope="shared")
  l16, l17, l18 = sch.get_loops(block=b15)
  sch.reverse_compute_at(block=b8, loop=l17, preserve_unit_loops=False, index=-1)
  sch.unannotate(block_or_loop=b8, ann_key="meta_schedule.meta_schedule_rfactor_producer_block")
  sch.annotate(block_or_loop=b8, ann_key="meta_schedule.meta_schedule_cross_thread_reduction_block", ann_val=1)
  sch.annotate(block_or_loop=b15, ann_key="meta_schedule.meta_schedule_rfactor_producer_block", ann_val=1)
  b19 = sch.get_block(name="C_rf_rf", func_name="main")
  sch.reorder_block_iter_var(block=b19, new_order=[1, 0, 2])
  sch.annotate(block_or_loop=b19, ann_key="meta_schedule.tiling_structure", ann_val="SRRR")
  l20, l21, l22 = sch.get_loops(block=b19)
  v23, v24, v25 = sch.sample_perfect_tile(loop=l22, n=3, max_innermost_factor=256, min_innermost_factor=1, decision=[16, 1, 128])
  l26, l27, l28 = sch.split(loop=l22, factors=[v23, v24, v25], preserve_unit_iters=True)
  sch.bind(loop=l20, thread_axis="blockIdx.x")
  sch.bind(loop=l21, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
  b29 = sch.cache_write(block=b19, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b29, loop=l27, preserve_unit_loops=True, index=-1)
  b30 = sch.cache_read(block=b19, read_buffer_index=0, storage_scope="local", consumer_blocks=[b19])
  sch.compute_at(block=b30, loop=l27, preserve_unit_loops=True, index=-1)
  l31, l32, l33, l34, l35 = sch.get_loops(block=b19)
  b36 = sch.decompose_reduction(block=b19, loop=l33)
  v37 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v37)
  b38 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b38, ann_key="meta_schedule.optimization_level", ann_val=4)
  sch.enter_postproc()
  b39 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b39, ann_key="meta_schedule.unroll_implicit")
  b40, b41, b42, b43, b44, b45 = sch.get_child_blocks(b39)
  l46, l47 = sch.get_loops(block=b40)
  sch.annotate(block_or_loop=l46, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l46, ann_key="pragma_unroll_explicit", ann_val=0)
  l48, l49, l50, l51, l52 = sch.get_loops(block=b41)
  sch.annotate(block_or_loop=l48, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l48, ann_key="pragma_unroll_explicit", ann_val=0)
  l53, l54, l55, l56, l57 = sch.get_loops(block=b42)
  sch.annotate(block_or_loop=l53, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l53, ann_key="pragma_unroll_explicit", ann_val=0)
  l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b43)
  sch.annotate(block_or_loop=l58, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l58, ann_key="pragma_unroll_explicit", ann_val=0)
  l65, l66 = sch.get_loops(block=b44)
  sch.annotate(block_or_loop=l65, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l65, ann_key="pragma_unroll_explicit", ann_val=0)
  l67, = sch.get_loops(block=b45)
  sch.annotate(block_or_loop=l67, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l67, ann_key="pragma_unroll_explicit", ann_val=0)
  b68 = sch.get_block(name="C", func_name="main")
  l69, = sch.get_loops(block=b68)
  b70 = sch.decompose_reduction(block=b68, loop=l69)

# Copied verbatim from ./results/tuned_modules/atim_gemv_4096_1_4096.py,
# the schedule measured for ('gemv', 4096, 1, 4096) (published).
from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_atim_gemv_4096_1_4096:
    @T.prim_func
    def main(A: T.Buffer((4096, 4096), "int32"), B: T.Buffer((4096,), "int32"), C: T.Buffer((4096,), "int32"), ALPHA: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_rf_global = T.alloc_buffer((8, 4096), "int32")
            C_rf_global_local = T.alloc_buffer((8, 4096), "int32", scope="local")
            A_local = T.alloc_buffer((4096, 4096), "int32", scope="local")
            B_local = T.alloc_buffer((4096,), "int32", scope="local")
            alpha_val: T.int32 = ALPHA[0]
            for k_0 in T.thread_binding(8, thread="blockIdx.x", annotations={"bank": T.int64(1), "pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(0)}):
                for i_0 in T.thread_binding(256, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                    for i_1 in T.thread_binding(8, thread="threadIdx.x"):
                        for i_2 in range(1):
                            for i_3_init, i_4_init in T.grid(1, 2):
                                with T.block("C_rf_init"):
                                    v_i = T.axis.spatial(4096, i_0 * 16 + i_1 * 2 + i_2 * 2 + i_3_init * 2 + i_4_init)
                                    vk_0 = T.axis.spatial(8, k_0)
                                    T.reads()
                                    T.writes(C_rf_global_local[vk_0, v_i])
                                    T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                    C_rf_global_local[vk_0, v_i] = 0
                            for k_1_0 in range(8):
                                for ax0, ax1 in T.grid(2, 64):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(4096, i_0 * 16 + i_1 * 2 + ax0)
                                        v1 = T.axis.spatial(4096, k_0 * 512 + k_1_0 * 64 + ax1)
                                        T.reads(A[v0, v1])
                                        T.writes(A_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1] = A[v0, v1]
                                for ax0 in range(64):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(4096, k_0 * 512 + k_1_0 * 64 + ax0)
                                        T.reads(B[v0])
                                        T.writes(B_local[v0])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0] = B[v0]
                                for i_3, k_1_1, i_4, k_1_2 in T.grid(1, 32, 2, 2):
                                    with T.block("C_rf_update"):
                                        v_i = T.axis.spatial(4096, i_0 * 16 + i_1 * 2 + i_2 * 2 + i_3 * 2 + i_4)
                                        vk_0 = T.axis.spatial(8, k_0)
                                        vk_1 = T.axis.reduce(512, k_1_0 * 64 + k_1_1 * 2 + k_1_2)
                                        T.reads(C_rf_global_local[vk_0, v_i], A_local[v_i, vk_0 * 512 + vk_1], B_local[vk_0 * 512 + vk_1])
                                        T.writes(C_rf_global_local[vk_0, v_i])
                                        T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                        C_rf_global_local[vk_0, v_i] = C_rf_global_local[vk_0, v_i] + alpha_val * A_local[v_i, vk_0 * 512 + vk_1] * B_local[vk_0 * 512 + vk_1]
                            for ax0, ax1 in T.grid(1, 2):
                                with T.block("C_rf_global_local"):
                                    v0 = T.axis.spatial(8, k_0 + ax0)
                                    v1 = T.axis.spatial(4096, i_0 * 16 + i_1 * 2 + ax1)
                                    T.reads(C_rf_global_local[v0, v1])
                                    T.writes(C_rf_global[v0, v1])
                                    C_rf_global[v0, v1] = C_rf_global_local[v0, v1]
            for i in T.serial(4096, annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(0)}):
                with T.block("C_init"):
                    v_i = T.axis.spatial(4096, i)
                    T.reads()
                    T.writes(C[v_i])
                    T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                    C[v_i] = 0
                for k_0 in range(8):
                    with T.block("C_update"):
                        vk_0, v_i = T.axis.remap("RS", [k_0, i])
                        T.reads(C[v_i], C_rf_global[vk_0, v_i])
                        T.writes(C[v_i])
                        T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                        C[v_i] = C[v_i] + C_rf_global[vk_0, v_i]
# from tvm import tir
def apply_trace_atim_gemv_4096_1_4096(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  l2, l3 = sch.get_loops(block=b0)
  v4, v5 = sch.sample_perfect_tile2(loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[8, 512])
  l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
  b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
  sch.annotate(block_or_loop=b8, ann_key="meta_schedule.meta_schedule_rfactor_producer_block", ann_val=1)
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.meta_schedule_rfactor_consumer_block", ann_val=1)
  b9 = sch.get_block(name="C_rf", func_name="main")
  sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
  sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
  l10, l11, l12 = sch.get_loops(block=b9)
  v13, v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l10, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[256, 8, 1, 1, 2])
  l18, l19, l20, l21, l22 = sch.split(loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True)
  v23, v24, v25 = sch.sample_perfect_tile(loop=l12, n=3, max_innermost_factor=256, min_innermost_factor=1, decision=[8, 32, 2])
  l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
  sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
  sch.bind(loop=l11, thread_axis="blockIdx.x")
  sch.bind(loop=l18, thread_axis="blockIdx.y")
  sch.reorder(l19)
  sch.bind(loop=l19, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
  sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
  b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b29, loop=l20, preserve_unit_loops=True, index=-1)
  b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
  sch.compute_at(block=b30, loop=l26, preserve_unit_loops=True, index=-1)
  v31 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
  b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
  sch.compute_at(block=b32, loop=l26, preserve_unit_loops=True, index=-1)
  v33 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
  v34 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
  b35 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
  sch.enter_postproc()
  b36 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
  b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
  l42, l43, l44, l45, l46, l47, l48 = sch.get_loops(block=b37)
  sch.annotate(block_or_loop=l42, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l42, ann_key="pragma_unroll_explicit", ann_val=0)
  l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b38)
  sch.annotate(block_or_loop=l49, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l49, ann_key="pragma_unroll_explicit", ann_val=0)
  l55, l56, l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b39)
  sch.annotate(block_or_loop=l55, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l55, ann_key="pragma_unroll_explicit", ann_val=0)
  l64, l65, l66, l67, l68, l69 = sch.get_loops(block=b40)
  sch.annotate(block_or_loop=l64, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l64, ann_key="pragma_unroll_explicit", ann_val=0)
  l70, l71 = sch.get_loops(block=b41)
  sch.annotate(block_or_loop=l70, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l70, ann_key="pragma_unroll_explicit", ann_val=0)
  b72 = sch.get_block(name="C_rf", func_name="main")
  l73, l74, l75, l76, l77, l78, l79, l80, l81 = sch.get_loops(block=b72)
  b82 = sch.decompose_reduction(block=b72, loop=l77)
  b83 = sch.get_block(name="C", func_name="main")
  l84, l85 = sch.get_loops(block=b83)
  b86 = sch.decompose_reduction(block=b83, loop=l85)

# Copied verbatim from ./results/tuned_modules/atim_mmtv_128_256_512.py,
# the schedule measured for ('mmtv', 128, 256, 512) (published).
from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_atim_mmtv_128_256_512:
    @T.prim_func
    def main(A: T.Buffer((128, 256, 512), "int32"), B: T.Buffer((128, 512), "int32"), C: T.Buffer((128, 256), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_local = T.alloc_buffer((128, 256), "int32", scope="local")
            A_local = T.alloc_buffer((128, 256, 512), "int32", scope="local")
            B_local = T.alloc_buffer((128, 512), "int32", scope="local")
            for i_0 in T.thread_binding(128, thread="blockIdx.x", annotations={"bank": T.int64(1), "pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(0)}):
                for j_0 in T.thread_binding(16, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                    for j_1 in T.thread_binding(8, thread="threadIdx.x"):
                        for i_1, i_2, j_2 in T.grid(1, 1, 1):
                            for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(1, 2, 1, 1):
                                with T.block("C_init"):
                                    v_i = T.axis.spatial(128, i_0 + i_1 + i_2 + i_3_init + i_4_init)
                                    v_j = T.axis.spatial(256, j_0 * 16 + j_1 * 2 + j_2 * 2 + j_3_init + j_4_init)
                                    T.reads()
                                    T.writes(C_local[v_i, v_j])
                                    T.block_attr({"meta_schedule.tiling_structure": "SSSRSRSR"})
                                    C_local[v_i, v_j] = 0
                            for k_0 in range(8):
                                for ax0, ax1, ax2 in T.grid(1, 2, 64):
                                    with T.block("A_local"):
                                        v0 = T.axis.spatial(128, i_0 + ax0)
                                        v1 = T.axis.spatial(256, j_0 * 16 + j_1 * 2 + ax1)
                                        v2 = T.axis.spatial(512, k_0 * 64 + ax2)
                                        T.reads(A[v0, v1, v2])
                                        T.writes(A_local[v0, v1, v2])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        A_local[v0, v1, v2] = A[v0, v1, v2]
                                for ax0, ax1 in T.grid(1, 64):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(128, i_0 + ax0)
                                        v1 = T.axis.spatial(512, k_0 * 64 + ax1)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                        B_local[v0, v1] = B[v0, v1]
                                for i_3, j_3, k_1, i_4, j_4, k_2 in T.grid(1, 2, 32, 1, 1, 2):
                                    with T.block("C_update"):
                                        v_i = T.axis.spatial(128, i_0 + i_1 + i_2 + i_3 + i_4)
                                        v_j = T.axis.spatial(256, j_0 * 16 + j_1 * 2 + j_2 * 2 + j_3 + j_4)
                                        v_k = T.axis.reduce(512, k_0 * 64 + k_1 * 2 + k_2)
                                        T.reads(C_local[v_i, v_j], A_local[v_i, v_j, v_k], B_local[v_i, v_k])
                                        T.writes(C_local[v_i, v_j])
                                        T.block_attr({"meta_schedule.tiling_structure": "SSSRSRSR"})
                                        C_local[v_i, v_j] = C_local[v_i, v_j] + A_local[v_i, v_j, v_k] * B_local[v_i, v_k]
                            for ax0, ax1 in T.grid(1, 2):
                                with T.block("C_local"):
                                    v0 = T.axis.spatial(128, i_0 + ax0)
                                    v1 = T.axis.spatial(256, j_0 * 16 + j_1 * 2 + ax1)
                                    T.reads(C_local[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_local[v0, v1]
# from tvm import tir
def apply_trace_atim_mmtv_128_256_512(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="root", func_name="main")
  b1 = sch.get_block(name="C", func_name="main")
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
  l2, l3, l4 = sch.get_loops(block=b1)
  v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[128, 1, 1, 1, 1])
  l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True)
  v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[16, 8, 1, 2, 1])
  l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True)
  v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=256, min_innermost_factor=1, decision=[8, 32, 2])
  l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
  sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
  sch.bind(loop=l10, thread_axis="blockIdx.x")
  sch.bind(loop=l20, thread_axis="blockIdx.y")
  sch.reorder(l21, l11)
  sch.bind(loop=l21, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
  sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
  b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
  b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
  sch.compute_at(block=b32, loop=l28, preserve_unit_loops=True, index=-1)
  v33 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
  b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
  sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True, index=-1)
  v35 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
  v36 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
  b37 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
  sch.enter_postproc()
  b38 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
  b39, b40, b41, b42 = sch.get_child_blocks(b38)
  l43, l44, l45, l46, l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b39)
  sch.annotate(block_or_loop=l43, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l43, ann_key="pragma_unroll_explicit", ann_val=0)
  l53, l54, l55, l56, l57, l58, l59, l60, l61 = sch.get_loops(block=b40)
  sch.annotate(block_or_loop=l53, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l53, ann_key="pragma_unroll_explicit", ann_val=0)
  l62, l63, l64, l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b41)
  sch.annotate(block_or_loop=l62, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l62, ann_key="pragma_unroll_explicit", ann_val=0)
  l75, l76, l77, l78, l79, l80, l81, l82 = sch.get_loops(block=b42)
  sch.annotate(block_or_loop=l75, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l75, ann_key="pragma_unroll_explicit", ann_val=0)
  b83 = sch.get_block(name="C", func_name="main")
  l84, l85, l86, l87, l88, l89, l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(block=b83)
  b97 = sch.decompose_reduction(block=b83, loop=l90)

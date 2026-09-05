# Copied verbatim from ./results/tuned_modules/atim_mmtv_256_512_512.py,
# the schedule measured for ('mmtv', 256, 512, 512) (published).
from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_atim_mmtv_256_512_512:
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
# from tvm import tir
def apply_trace_atim_mmtv_256_512_512(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
  l1, l2, l3 = sch.get_loops(block=b0)
  v4, v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l1, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[256, 1, 1, 1, 1])
  l9, l10, l11, l12, l13 = sch.split(loop=l1, factors=[v4, v5, v6, v7, v8], preserve_unit_iters=True)
  v14, v15, v16, v17, v18 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[8, 16, 4, 1, 1])
  l19, l20, l21, l22, l23 = sch.split(loop=l2, factors=[v14, v15, v16, v17, v18], preserve_unit_iters=True)
  v24, v25, v26 = sch.sample_perfect_tile(loop=l3, n=3, max_innermost_factor=256, min_innermost_factor=1, decision=[8, 8, 8])
  l27, l28, l29 = sch.split(loop=l3, factors=[v24, v25, v26], preserve_unit_iters=True)
  sch.reorder(l9, l19, l10, l20, l11, l21, l27, l12, l22, l28, l13, l23, l29)
  sch.bind(loop=l9, thread_axis="blockIdx.x")
  sch.bind(loop=l19, thread_axis="blockIdx.y")
  sch.reorder(l20, l10)
  sch.bind(loop=l20, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=l9, ann_key="bank", ann_val=1)
  sch.annotate(block_or_loop=l19, ann_key="bank", ann_val=1)
  b30 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b30, loop=l21, preserve_unit_loops=True, index=-1)
  b31 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="local", consumer_blocks=[b0])
  sch.compute_at(block=b31, loop=l27, preserve_unit_loops=True, index=-1)
  l32, l33, l34, l35, l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b31)
  l42 = sch.fuse(l39, l40, l41, preserve_unit_iters=True)
  v43 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b31, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
  b44 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="local", consumer_blocks=[b0])
  sch.compute_at(block=b44, loop=l27, preserve_unit_loops=True, index=-1)
  l45, l46, l47, l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b44)
  l54 = sch.fuse(l52, l53, preserve_unit_iters=True)
  v55 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
  b56 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b56, ann_key="meta_schedule.optimization_level", ann_val=4)
  sch.enter_postproc()
  b57 = sch.get_block(name="C", func_name="main")
  l58, l59, l60, l61, l62, l63, l64, l65, l66, l67, l68, l69, l70 = sch.get_loops(block=b57)
  b71 = sch.decompose_reduction(block=b57, loop=l64)

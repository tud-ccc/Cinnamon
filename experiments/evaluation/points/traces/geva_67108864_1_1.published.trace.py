# Copied verbatim from ./results/tuned_modules/atim_geva_67108864_1_1.py,
# the schedule measured for ('geva', 67108864, 1, 1) (published).
from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_atim_geva_67108864_1_1:
    @T.prim_func
    def main(A: T.Buffer((67108864,), "int32"), B: T.Buffer((67108864,), "int32"), C: T.Buffer((67108864,), "int32"), ALPHA: T.Buffer((1,), "int32"), BETA: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A", "B"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_local = T.alloc_buffer((67108864,), "int32", scope="local")
            A_local = T.alloc_buffer((67108864,), "int32", scope="local")
            B_local = T.alloc_buffer((67108864,), "int32", scope="local")
            alpha_val: T.int32 = ALPHA[0]
            beta_val: T.int32 = BETA[0]
            for i_0 in T.thread_binding(2048, thread="blockIdx.x", annotations={"bank": T.int64(1), "pragma_auto_unroll_max_step": T.int64(16), "pragma_unroll_explicit": T.int64(0)}):
                for i_1 in T.thread_binding(8, thread="threadIdx.x"):
                    for i_2 in range(256):
                        for ax0 in range(16):
                            with T.block("A_local"):
                                v0 = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 4096 + i_2 * 16 + ax0)
                                T.reads(A[v0])
                                T.writes(A_local[v0])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                A_local[v0] = A[v0]
                        for ax0 in range(16):
                            with T.block("B_local"):
                                v0 = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 4096 + i_2 * 16 + ax0)
                                T.reads(B[v0])
                                T.writes(B_local[v0])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                B_local[v0] = B[v0]
                        for i_3, i_4 in T.grid(4, 4):
                            with T.block("C"):
                                v_i = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 4096 + i_2 * 16 + i_3 * 4 + i_4)
                                T.reads(A_local[v_i], B_local[v_i])
                                T.writes(C_local[v_i])
                                T.block_attr({"meta_schedule.tiling_structure": "SSSSS"})
                                C_local[v_i] = alpha_val * A_local[v_i] + beta_val * B_local[v_i]
                        for ax0 in range(16):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 4096 + i_2 * 16 + ax0)
                                T.reads(C_local[v0])
                                T.writes(C[v0])
                                C[v0] = C_local[v0]
# from tvm import tir
def apply_trace_atim_geva_67108864_1_1(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSSS")
  l2, = sch.get_loops(block=b0)
  v3, v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[2048, 8, 256, 4, 4])
  l8, l9, l10, l11, l12 = sch.split(loop=l2, factors=[v3, v4, v5, v6, v7], preserve_unit_iters=True)
  sch.reorder(l8, l9, l10, l11, l12)
  sch.bind(loop=l8, thread_axis="blockIdx.x")
  sch.bind(loop=l9, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=l8, ann_key="bank", ann_val=1)
  b13 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b13, loop=l10, preserve_unit_loops=True, index=-1)
  b14 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="local", consumer_blocks=[b0])
  sch.compute_at(block=b14, loop=l10, preserve_unit_loops=True, index=-1)
  v15 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b14, ann_key="meta_schedule.cooperative_fetch", ann_val=v15)
  b16 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="local", consumer_blocks=[b0])
  sch.compute_at(block=b16, loop=l10, preserve_unit_loops=True, index=-1)
  v17 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b16, ann_key="meta_schedule.cooperative_fetch", ann_val=v17)
  v18 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v18)
  b19 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b19, ann_key="meta_schedule.optimization_level", ann_val=4)
  sch.enter_postproc()
  b20 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.unroll_implicit")
  b21, b22, b23, b24 = sch.get_child_blocks(b20)
  l25, l26, l27, l28 = sch.get_loops(block=b21)
  sch.annotate(block_or_loop=l25, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l25, ann_key="pragma_unroll_explicit", ann_val=0)
  l29, l30, l31, l32 = sch.get_loops(block=b22)
  sch.annotate(block_or_loop=l29, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l29, ann_key="pragma_unroll_explicit", ann_val=0)
  l33, l34, l35, l36, l37 = sch.get_loops(block=b23)
  sch.annotate(block_or_loop=l33, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l33, ann_key="pragma_unroll_explicit", ann_val=0)
  l38, l39, l40, l41 = sch.get_loops(block=b24)
  sch.annotate(block_or_loop=l38, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l38, ann_key="pragma_unroll_explicit", ann_val=0)

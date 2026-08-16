# Copied verbatim from ./results/tuned_modules/atim_va_67108864_1_1.py,
# the schedule measured for ('va', 67108864, 1, 1) (published).
from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_atim_va_67108864_1_1:
    @T.prim_func
    def main(A: T.Buffer((67108864,), "int32"), B: T.Buffer((67108864,), "int32"), C: T.Buffer((67108864,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A", "B"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_local = T.alloc_buffer((67108864,), "int32", scope="local")
            A_local = T.alloc_buffer((67108864,), "int32", scope="local")
            B_local = T.alloc_buffer((67108864,), "int32", scope="local")
            for i_0 in T.thread_binding(2048, thread="blockIdx.x", annotations={"bank": T.int64(1)}):
                for i_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for i_2 in range(64):
                        for ax0_fused in range(32):
                            with T.block("A_local"):
                                v0 = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 2048 + i_2 * 32 + ax0_fused)
                                T.reads(A[v0])
                                T.writes(A_local[v0])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                A_local[v0] = A[v0]
                        for ax0_fused in range(32):
                            with T.block("B_local"):
                                v0 = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 2048 + i_2 * 32 + ax0_fused)
                                T.reads(B[v0])
                                T.writes(B_local[v0])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                B_local[v0] = B[v0]
                        for i_3, i_4 in T.grid(2, 16):
                            with T.block("C"):
                                v_i = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 2048 + i_2 * 32 + i_3 * 16 + i_4)
                                T.reads(A_local[v_i], B_local[v_i])
                                T.writes(C_local[v_i])
                                T.block_attr({"meta_schedule.tiling_structure": "SSSSS"})
                                C_local[v_i] = A_local[v_i] + B_local[v_i]
                        for ax0 in range(32):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(67108864, i_0 * 32768 + i_1 * 2048 + i_2 * 32 + ax0)
                                T.reads(C_local[v0])
                                T.writes(C[v0])
                                C[v0] = C_local[v0]
# from tvm import tir
def apply_trace_atim_va_67108864_1_1(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSSS")
  l1, = sch.get_loops(block=b0)
  v2, v3, v4, v5, v6 = sch.sample_perfect_tile(loop=l1, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[2048, 16, 64, 2, 16])
  l7, l8, l9, l10, l11 = sch.split(loop=l1, factors=[v2, v3, v4, v5, v6], preserve_unit_iters=True)
  sch.reorder(l7, l8, l9, l10, l11)
  sch.bind(loop=l7, thread_axis="blockIdx.x")
  sch.bind(loop=l8, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=l7, ann_key="bank", ann_val=1)
  b12 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b12, loop=l9, preserve_unit_loops=True, index=-1)
  b13 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="local", consumer_blocks=[b0])
  sch.compute_at(block=b13, loop=l9, preserve_unit_loops=True, index=-1)
  l14, l15, l16, l17 = sch.get_loops(block=b13)
  l18 = sch.fuse(l17, preserve_unit_iters=True)
  v19 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b13, ann_key="meta_schedule.cooperative_fetch", ann_val=v19)
  b20 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="local", consumer_blocks=[b0])
  sch.compute_at(block=b20, loop=l9, preserve_unit_loops=True, index=-1)
  l21, l22, l23, l24 = sch.get_loops(block=b20)
  l25 = sch.fuse(l24, preserve_unit_iters=True)
  v26 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.cooperative_fetch", ann_val=v26)
  b27 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b27, ann_key="meta_schedule.optimization_level", ann_val=4)
  sch.enter_postproc()

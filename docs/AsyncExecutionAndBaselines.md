# Async execution, CPU baseline, and integer weights

Investigation for RQ4 (`sec:eval:wholeprogram`), 2026-08-25. Three
independent questions, ordered by how much they block the paper.

## 1. The problem this starts from

Execution is synchronous end to end: `upmem.wait_for` blocks the host, so
compute blocks on disjoint device sets still run in program order. But the
latency objective's `makespanOf` (GraphAllocation.cpp) computes

```cpp
for (unsigned pred : node.predecessors)
    start = std::max(start, finish[pred]);   // only dependencies constrain
```

so two independent nodes on different sets both start at 0 and **overlap**.
The allocator therefore optimizes a makespan the runtime cannot realize, and
on RQ4's parallel and mixed instances its predicted latency is optimistic.
The evaluation section already knows not to *claim* concurrency; the
allocator has not been told.

Two ways out, and they are not alternatives so much as now-and-later.

## 2. Option A (now): make the latency model match the machine

Add the program-order edge the runtime actually imposes -- every node waits
for the previous one -- so "latency" means what a synchronous run measures:
the sum of the blocks' costs, each priced at the device size its set got.

- Change: `makespanOf` chains consecutive nodes in addition to the
  dependency and same-set edges. ~10 lines.
- The objective does **not** become degenerate: node cost still depends on
  how many units its set holds, so the allocator is still choosing profile
  points, just under "minimize total work" rather than "minimize critical
  path". For a synchronous runtime that is the correct objective.
- Consequence for the 2x2 (`scoreAllocation`): the round-robin member deal
  stops mattering, since with a total order the makespan no longer depends
  on which set an interchangeable member sits on. The latency score of a
  throughput allocation becomes exact, and the footnote about it being an
  upper bound can be dropped.
- Cost: hours. Do this before the RQ4 runs land, or the latency arm's
  numbers describe a machine we do not have.

## 3. Option B (later): real asynchronous execution

**Do not lower to `async.execute` with the MLIR async runtime.** That path
wants `--async-to-async-runtime` + `--convert-async-to-llvm`, coroutine
lowering, and linking `libmlir_async_runtime`, which brings a thread pool.
Against that: UPMEM `dpu_set_t` handles are not documented thread-safe, so
launching from pool threads is a race we would have to prove absent; and the
runtime's scheduling overhead lands directly in ms-scale measurements we are
trying to attribute to kernels and transfers. It is the heavyweight answer to
a problem that does not need concurrency on the host at all -- only
*non-blocking launches*.

The cheap shape (already recorded in
`experiments/evaluation/README.md` under "Concurrent group execution"): keep
one host thread and use the SDK's own asynchrony. The upstream `async`
dialect is optional there -- worth it only as a *representation* (token
types instead of homegrown futures), never for its runtime.

### What the code actually looks like today

The blocking structure is smaller than it appears, but it is mis-named:

- `upmem.wait_for` **is** the launch. `WaitForOpToFuncCallLowering`
  (UPMEMToLLVM.cpp:1044) lowers it to `upmemrt_dpu_launch`, and
  `upmemrt_dpu_launch` calls `dpu_launch(set, DPU_SYNCHRONOUS)`
  (upmem_rt.c:249). There is no separate launch op.
- `CnmToUPMEM.cpp:743` creates exactly one `WaitForOp` per launch region,
  immediately after it.
- `upmem_rt.c` already has an `ASYNC_TRANSFERS` path calling `dpu_sync`, so
  the concept is half-scaffolded (and the comment there notes it perturbs
  timing -- see the measurement item below).

### Work items

1. **Runtime** (~30 lines): `upmemrt_dpu_launch_async` issuing
   `dpu_launch(set, DPU_ASYNCHRONOUS)`, and `upmemrt_dpu_sync` issuing
   `dpu_sync(set)`. Trivial.
2. **Dialect**: split launch from wait. `upmem.wait_for` must stop being the
   launch -- add `upmem.launch_async` (or an attribute) and give `wait_for`
   its real meaning. Touches UPMEMOps.td, UPMEMToLLVM, and **both**
   simulators, which `Case<WaitForOp>` on the fused op today
   (UpmemOpCountSimulator.cpp:218, UpmemPythonSimulator.cpp:542).
3. **Placement pass**: emit the launch where the region ends and sink the
   wait to the first consumer of the block's results. Straightforward on the
   committed graph, where the consumers are the gathers.
4. **Correctness**: one set cannot have two launches outstanding. Pinned
   groups are disjoint by construction (the allocator guarantees it), so
   dependency edges plus one serialization token per group is the whole
   argument -- no general aliasing analysis needed.
5. **Cost model**: this is the deep part, not the plumbing. With overlap real,
   the simulator stops being a sum over ops and becomes a schedule, and
   `makespanOf` becomes the *right* model rather than the optimistic one. The
   two must agree, or the allocator optimizes something the simulator does not
   price.
6. **Measurement**: `upmemrt_record_launch` times a blocking call today. Async
   moves that time into the sync, so the per-phase stats (and every walltime
   number derived from them) need rework, and the `ASYNC_TRANSFERS` comment
   about perturbed timing is a live warning.

**Estimate**: items 1-4 are roughly a focused week to a working prototype.
Item 5 is the one that can overrun -- it is a modelling change, not an
integration one -- and item 6 quietly invalidates existing timing artifacts.
Not on the paper's critical path; Option A is.

## 4. CPU baseline for the llama model

The trap to avoid: the device arm computes i32 GEMMs, so an fp32 PyTorch
baseline compares two different computations and is unfair in *both*
directions (fp32 does more work per element; PyTorch's int path is not the
same kernel). Whatever we pick has to run the same arithmetic.

- **PyTorch** is the easiest to stand up but the worst fit here. Integer
  matmul is only reachable through the quantization backends (fbgemm /
  qnnpack), which are i8 with i32 accumulate and fused requantization -- not
  our i32-in/i32-out GEMM. Making it match means fighting the framework.
- **TVM** is the better baseline and the autotuning fear is avoidable: it
  only takes days if we *ask* for Ansor/AutoTVM. With **default schedules**
  (`relay.build`, `llvm -mcpu=native`) an i32 model compiles in minutes and
  gives a legitimate "compiler-generated CPU code" number. If a stronger
  number is wanted, run Ansor with a bounded trial budget (a few hundred
  trials, capped wall clock) and report the budget -- a bounded-budget
  autotuner is exactly the comparison this paper is already making
  elsewhere. We also already have TVM scaffolding vendored under
  `experiments/evaluation/I-ViT/TVM_benchmark/`.
- Report thread count explicitly and pin it; a single-thread number and an
  all-cores number answer different questions and reviewers ask for both.

**Recommendation**: TVM, i32, default schedules as the primary baseline;
optionally one bounded-budget Ansor run as a second bar. Skip PyTorch unless
it is only ever a sanity check on numerics.

## 5. Integer weights, I-ViT style

Two requirements are easy to conflate here, and only one of them is blocking:

- **For RQ4 itself**, random i32 data is enough. That experiment measures
  time against shapes and dependency structure; no accuracy claim is made,
  and the section already says "in i32 on random data".
- **For a CPU baseline comparison**, the baseline must compute the same
  thing, which is what forces a coherent integer model rather than i32
  noise through fp32-shaped ops.

I-ViT's contribution is integer-only *nonlinearities* -- ShiftGELU,
ShiftMax, I-LayerNorm -- so that nothing dequantizes mid-network. The llama
analogues are integer RMSNorm, integer SiLU/SwiGLU for the FFN, and integer
softmax in attention. The linear layers are the easy part: symmetric
per-channel W8A8 with i32 accumulation.

Scoping note: the compute blocks that gave the profiler trouble are exactly
the nonlinear ones (the reduction classes -- rmsnorm's sum of squares,
softmax's max and sum), which are also the ones I-ViT has to redesign. So
this work and the transfer-bound/host-fallback question in
`SearchStrategyPlan.md` touch the same operators; do them together.

Suggested order: quantize the linears first (i8 weights, i32 accumulate) and
keep the nonlinearities as they are for RQ4's timing purposes; adopt the
I-ViT-style integer nonlinearities only if the CPU baseline comparison needs
end-to-end integer semantics.

## Open questions for the author

- Is RQ4's latency arm meant to describe the machine we have (Option A) or
  the machine we would like (Option B)? The paper text currently implies the
  first while the allocator implements the second.
- Does the CPU baseline need to be *fair* (same arithmetic, which forces the
  quantized model) or *indicative* (fp32 reference, clearly labelled)? That
  decision sets how much of section 5 is on the critical path.

#include <upmem_cost_model/ProgramBuilder.h>

#include <cstdio>
#include <chrono>

using namespace upmem_cm;

static const int NITER      = 100;
static const int WARM_ITERS = 10;

// Mirrors barrier.c: pre-work loop, main loop (with/without barrier), post-work loop.
static void buildProgram(ProgramBuilder &b, bool withBarrier) {
  auto buf = b.addBuffer("buf", MemSpace::WRAM, DType::I32);

  // Pre-work: x = x*2 + 1, WARM_ITERS times, stored to buf.
  b.beginLoop(0, WARM_ITERS);
  {
    auto v   = b.createLoad(buf);
    auto two = b.createConst(2, DType::I32);
    auto v2  = b.createArith(ArithOp::MUL, DType::I32, v, two);
    auto one = b.createConst(1, DType::I32);
    auto v3  = b.createArith(ArithOp::ADD, DType::I32, v2, one);
    b.createStore(buf, v3);
  }
  b.endLoop();

  // Main loop: buf += 1, optional barrier.
  b.beginLoop(0, NITER);
  {
    auto v   = b.createLoad(buf);
    auto one = b.createConst(1, DType::I32);
    auto v2  = b.createArith(ArithOp::ADD, DType::I32, v, one);
    b.createStore(buf, v2);
    if (withBarrier)
      b.createBarrier();
  }
  b.endLoop();

  // Post-work: same as pre.
  b.beginLoop(0, WARM_ITERS);
  {
    auto v   = b.createLoad(buf);
    auto two = b.createConst(2, DType::I32);
    auto v2  = b.createArith(ArithOp::MUL, DType::I32, v, two);
    auto one = b.createConst(1, DType::I32);
    auto v3  = b.createArith(ArithOp::ADD, DType::I32, v2, one);
    b.createStore(buf, v3);
  }
  b.endLoop();
}

int main() {
  auto timeout = std::chrono::milliseconds(5000);

  printf("tasklets,barrier_ns,control_ns,overhead_ns_per_barrier\n");
  for (int T = 1; T <= 24; T++) {
    ProgramBuilder barrier_prog;
    buildProgram(barrier_prog, /*withBarrier=*/true);
    auto barrier_s = barrier_prog.simulate(T, timeout);

    ProgramBuilder control_prog;
    buildProgram(control_prog, /*withBarrier=*/false);
    auto control_s = control_prog.simulate(T, timeout);

    if (!barrier_s || !control_s) {
      printf("%d,timeout,timeout,timeout\n", T);
      continue;
    }

    double barrier_ns  = *barrier_s  * 1e9;
    double control_ns  = *control_s  * 1e9;
    double overhead_ns = (barrier_ns - control_ns) / NITER;
    printf("%d,%.1f,%.1f,%.2f\n", T, barrier_ns, control_ns, overhead_ns);
  }
  return 0;
}

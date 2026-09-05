#include <upmem_cost_model/ProgramBuilder.h>

#include <chrono>
#include <cstdio>
#include <future>
#include <optional>
#include <vector>

using namespace upmem_cm;

static const int NITER = 100;
static const int WARM_ITERS = 10;

// Mirrors barrier.c: pre-work loop, main loop (with/without barrier), post-work
// loop.
static void buildProgram(ProgramBuilder &b, bool withBarrier) {
  auto buf = b.addBuffer("buf", MemSpace::WRAM, DType::I32);

  // Pre-work: x = x*2 + 1, WARM_ITERS times, stored to buf.
  b.beginLoop(0, WARM_ITERS);
  {
    auto v = b.createLoad(buf);
    auto two = b.createConst(2, DType::I32);
    auto v2 = b.createArith(ArithOp::MUL, DType::I32, v, two);
    auto one = b.createConst(1, DType::I32);
    auto v3 = b.createArith(ArithOp::ADD, DType::I32, v2, one);
    b.createStore(buf, v3);
  }
  b.endLoop();

  // Main loop: buf += 1, optional barrier.
  b.beginLoop(0, NITER);
  {
    auto v = b.createLoad(buf);
    auto one = b.createConst(1, DType::I32);
    auto v2 = b.createArith(ArithOp::ADD, DType::I32, v, one);
    b.createStore(buf, v2);
    if (withBarrier)
      b.createBarrier();
  }
  b.endLoop();

  // Post-work: same as pre.
  b.beginLoop(0, WARM_ITERS);
  {
    auto v = b.createLoad(buf);
    auto two = b.createConst(2, DType::I32);
    auto v2 = b.createArith(ArithOp::MUL, DType::I32, v, two);
    auto one = b.createConst(1, DType::I32);
    auto v3 = b.createArith(ArithOp::ADD, DType::I32, v2, one);
    b.createStore(buf, v3);
  }
  b.endLoop();
}

struct SimResult {
  double barrier_ns;
  double control_ns;
  bool timed_out;
};

static SimResult run_one(int T) {
  auto timeout = std::chrono::milliseconds(20000);

  ProgramBuilder barrier_prog;
  buildProgram(barrier_prog, /*withBarrier=*/true);
  auto barrier_s = barrier_prog.simulate(T, timeout);

  ProgramBuilder control_prog;
  buildProgram(control_prog, /*withBarrier=*/false);
  auto control_s = control_prog.simulate(T, timeout);

  if (!barrier_s || !control_s)
    return {0, 0, true};
  return {*barrier_s * 1e9, *control_s * 1e9, false};
}

int main() {
  std::vector<std::future<SimResult>> futures;
  futures.reserve(24);
  for (int T = 1; T <= 24; T++)
    futures.push_back(std::async(std::launch::async, run_one, T));

  printf("tasklets,barrier_ns,control_ns,overhead_ns_per_barrier\n");
  for (int T = 1; T <= 24; T++) {
    auto r = futures[T - 1].get();
    if (r.timed_out) {
      printf("%d,timeout,timeout,timeout\n", T);
      continue;
    }
    double overhead_ns = (r.barrier_ns - r.control_ns) / NITER;
    printf("%d,%.1f,%.1f,%.2f\n", T, r.barrier_ns, r.control_ns, overhead_ns);
  }
  return 0;
}

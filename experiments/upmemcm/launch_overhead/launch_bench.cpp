// Times dpu_launch over a sweep of DPU-set sizes, running the same trivial
// program (launch_dpu.c) on every one of them.
//
// The timed call is exactly the one the generated host code makes -- see
// upmemrt_dpu_launch in runtime/Upmem/upmem_rt.c, whose launch.csv rows are
// what the cost model gets compared against. Allocation and load are timed
// too, on their own rows: they happen once per set here, and whether they
// belong in a per-inference cost depends on how often the workload switches
// programs, which is not this file's call to make.
//
// Output is one long-format CSV row per timed call:
//   requested_dpus,allocated_dpus,tasklets,phase,iter,ns
//
// Environment:
//   LAUNCH_DPUS     comma-separated set sizes  (default 1,2,...,2048)
//   LAUNCH_ITERS    timed launches per size    (default 50)
//   LAUNCH_WARMUP   untimed launches first     (default 5)
//   LAUNCH_TASKLETS tasklet count to record; must match the binary's
//                   NR_TASKLETS, which the Makefile fixes  (default 8)
//   LAUNCH_BINARY   path to the DPU binary     (default bin/launch_dpu)
//   LAUNCH_CSV_OUT  output path                (default results.csv)
//   UPMEM_PROFILE   passed through to dpu_alloc, as the runtime does

extern "C" {
#include <dpu.h>
}

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::string env_or(const char *name, const char *fallback) {
  const char *v = getenv(name);
  return (v && *v) ? std::string(v) : std::string(fallback);
}

long env_long(const char *name, long fallback) {
  const char *v = getenv(name);
  return (v && *v) ? strtol(v, nullptr, 10) : fallback;
}

std::vector<uint32_t> parse_sizes(const std::string &spec) {
  std::vector<uint32_t> out;
  size_t pos = 0;
  while (pos < spec.size()) {
    size_t comma = spec.find(',', pos);
    std::string tok = spec.substr(pos, comma - pos);
    if (!tok.empty())
      out.push_back(static_cast<uint32_t>(strtoul(tok.c_str(), nullptr, 10)));
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

} // namespace

int main() {
  const std::vector<uint32_t> sizes = parse_sizes(
      env_or("LAUNCH_DPUS", "1,2,4,8,16,32,64,128,256,512,1024,2048"));
  const long iters = env_long("LAUNCH_ITERS", 50);
  const long warmup = env_long("LAUNCH_WARMUP", 5);
  const long tasklets = env_long("LAUNCH_TASKLETS", 8);
  const std::string binary = env_or("LAUNCH_BINARY", "bin/launch_dpu");
  const std::string csv_out = env_or("LAUNCH_CSV_OUT", "results.csv");
  const char *profile = getenv("UPMEM_PROFILE");
  if (profile && !*profile)
    profile = nullptr;

  FILE *csv = fopen(csv_out.c_str(), "w");
  if (!csv) {
    fprintf(stderr, "cannot open %s for writing\n", csv_out.c_str());
    return 1;
  }
  fprintf(csv, "requested_dpus,allocated_dpus,tasklets,phase,iter,ns\n");

  for (uint32_t requested : sizes) {
    struct dpu_set_t set;
    uint64_t t0 = now_ns();
    // Not DPU_ASSERT: a size the machine cannot give us right now (another
    // job holds the ranks) should cost that point and not the whole sweep.
    if (dpu_alloc(requested, profile, &set) != DPU_OK) {
      fprintf(stderr, "skipping %u DPUs: allocation failed\n", requested);
      continue;
    }
    uint64_t alloc_ns = now_ns() - t0;

    uint32_t allocated = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &allocated));

    t0 = now_ns();
    DPU_ASSERT(dpu_load(set, binary.c_str(), nullptr));
    uint64_t load_ns = now_ns() - t0;

    fprintf(csv, "%u,%u,%ld,alloc,0,%lu\n", requested, allocated, tasklets,
            static_cast<unsigned long>(alloc_ns));
    fprintf(csv, "%u,%u,%ld,load,0,%lu\n", requested, allocated, tasklets,
            static_cast<unsigned long>(load_ns));

    for (long i = 0; i < warmup; ++i)
      DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    uint64_t total = 0;
    for (long i = 0; i < iters; ++i) {
      t0 = now_ns();
      DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
      uint64_t ns = now_ns() - t0;
      total += ns;
      fprintf(csv, "%u,%u,%ld,launch,%ld,%lu\n", requested, allocated, tasklets,
              i, static_cast<unsigned long>(ns));
    }

    DPU_ASSERT(dpu_free(set));
    printf("%5u DPUs (%5u allocated): launch mean %8.3f ms over %ld iters\n",
           requested, allocated, total / 1e6 / iters, iters);
    fflush(stdout);
  }

  fclose(csv);
  printf("wrote %s\n", csv_out.c_str());
  return 0;
}

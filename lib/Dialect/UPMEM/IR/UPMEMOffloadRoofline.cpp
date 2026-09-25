//===- UPMEMOffloadRoofline.cpp - is this op worth putting on DPUs? -------===//
//
// A roofline comparison between the host and the DPU array, used to decide
// whether an op should be offloaded at all.
//
// The shape of the answer is forced by two facts about this machine. The
// host out-computes the whole array by orders of magnitude, and its path to
// DRAM, 15 GB/s, is about as fast as the host-to-DPU scatter at the DPU
// counts worth using (12.9 GB/s at 1024 DPUs, 17.2 at 2048, 16.2 at 2560 by
// the calibrated model) and faster than the gather back (6 to 9 GB/s). So
// for an op whose every operand has to be shipped in afresh the host wins
// unless the op moves almost nothing back: it reads the same bytes from
// DRAM as fast as we push them over the wire, does the arithmetic faster,
// and never pays the return trip.
//
// The second fact is the weaker of the two and is worth not overstating: at
// the top of the array the scatter edges DRAM by a few percent, so an op
// that streams a large operand in and a small result out (a gemv with a
// streamed matrix) can pass on traffic alone, by that margin. The rates
// come from the calibrated transfer model for the array actually in use
// rather than from an assumed ordering.
//
// What is left is amortization. An operand that is the same on every
// invocation -- a weight matrix, anything `cinm.static` -- can be scattered
// once and kept resident, after which the device does not pay for it again
// while the host still streams it from DRAM every single time. That is the
// only source of advantage, so the gate reduces to: is there a resident
// operand big enough that not re-reading it beats being slower at
// everything else?
//
// Both sides are priced as a roofline, max(compute, traffic):
//
//   host   = max(W / F_host,   (S + D) / B_dram)
//   device = max(W / F_dpu,     D      / B_scatter)
//
// with W the arithmetic, S the resident bytes and D the per-invocation
// traffic. Offload when device < host. The two conditions worth stating
// separately both fall out of that inequality rather than being bolted on:
//
//   - If the host is compute-bound (W/(S+D) above its ridge), host reduces
//     to W/F_host, and device >= W/F_dpu > W/F_host. Never profitable.
//   - If S is zero, the comparison is D/B_scatter < D/B_dram, which needs
//     the scatter to beat DRAM. At the array sizes this targets it does
//     not, so the traffic term alone cannot carry an op, and the compute
//     term never will.
//
// What this deliberately does not model, and which way each error points:
//
//   - The device side counts arithmetic only. A DPU MAC also costs two WRAM
//     loads and the MRAM staging around them, so the real F_dpu is some way
//     below what deviceOpsPerSecond returns. The gate is therefore generous
//     to the device: an op it rejects would not have been rescued by a
//     finer model, while one it accepts might still lose.
//   - The host side is peak, not achieved. Compiled code does not reach
//     either number, but the two errors are in the same direction and the
//     comparison is a ratio.
//   - Nothing here knows about tiling, so it cannot see a configuration
//     that would have made a rejected op fit. This is a gate on whether to
//     look at all, not a substitute for the search.
//
// It is a screen, in other words, not a predictor: it is meant to be right
// about the order of magnitude and about the sign.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOffloadModel.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"

#include "mlir/IR/BuiltinTypes.h"

#include <upmem_cost_model/ScatterGatherCm.h>

#include <algorithm>

using namespace mlir;
using namespace mlir::cinm;

namespace mlir::upmem {

namespace {

//===----------------------------------------------------------------------===//
// Device constants
//===----------------------------------------------------------------------===//

/// DPU clock. The v1A parts in this machine run at 342-350 MHz, which is
/// also the cost model's default (ProgramBuilder::simulate). Worth knowing
/// when reading other people's numbers: PIM-LLM (CGO'25) reports 450 MHz
/// parts, so their device compute is 29% cheaper than this at equal DPU
/// counts.
constexpr double kDpuClockHz = 350.0e6;

/// Pipeline depth. With at least this many tasklets resident the DPU retires
/// one instruction per cycle, so an instruction quoted at L cycles of
/// latency costs L/kPipelineDepth cycles of throughput -- and the entries
/// below are dominated by instruction counts for the emulated operations,
/// which is exactly what that division recovers.
constexpr double kPipelineDepth = 11.0;

/// Seconds to move `bytes` between host and array, priced by the calibrated
/// transfer model rather than by a bandwidth.
///
/// Both of the model's terms are charged, and per op. An offloaded region
/// scatters its inputs and gathers its results across the host boundary
/// every time it runs -- activations do not survive on the device from one
/// offloaded op to the next -- so the fixed cost of reaching the array is
/// paid once per op and not once per chain.
///
/// That fixed cost is the term that decides most of these verdicts, and it
/// is large: 0.44 ms to reach 2048 DPUs and 0.75 ms to reach 2560 whatever
/// the size, against a RoBERTa projection kernel of 0.17 ms. Pricing only
/// the slope would have quietly assumed the residency the compiler does not
/// provide, and would have accepted a great many ops whose transfers
/// dominate them.
///
/// The slope matters too, and is not guessable: the scatter runs 8.3 GB/s
/// at 512 DPUs, 12.9 at 1024, 17.2 at 2048 and 16.2 at 2560, the gather
/// 4.2, 6.1, 9.2 and 6.8. Since the gate is a comparison against the host's
/// ~15 GB/s, a flat constant would have flipped verdicts at one end or the
/// other.
///
/// The trees behind these numbers are fitted on the scatter_cost sweep of
/// isca-artifact (1 to 2048 DPUs, 8 bytes to 1 MB per DPU), so within that
/// range they interpolate; 2560 DPUs is the one extrapolation the gate
/// asks of them. One guard stays on the way out: a gather is never priced
/// below the scatter of the same geometry. On this hardware DPU-to-host is
/// the slower direction, and the fit says so everywhere it was measured,
/// so the guard only ever binds where an extrapolation would have inverted
/// that.
double transferSeconds(double bytes, int64_t dpus, bool toDevice) {
  if (bytes <= 0.0 || dpus <= 0)
    return 0.0;
  const int block = static_cast<int>(std::max(1.0, bytes / dpus));
  const double scatterMs = upmem_cm::scatterBlockCostMs(dpus, block);
  const double ms =
      toDevice ? scatterMs
               : std::max(upmem_cm::gatherCostMs(dpus, block), scatterMs);
  return std::max(0.0, ms) * 1e-3;
}

/// Cycles of latency for one arithmetic instruction, transcribed from the
/// cost model's kStaticLatency table (upmem_cost_model/Simulation.h), which
/// is itself transcribed from the calibrated LUT. Only the rows the gate
/// prices are kept.
///
/// The 32-bit multiply is the entry to be careful with: the DPU has no
/// 32-bit multiplier, so it is a __mulsi3 call costing 88 + 4*11 = 132
/// cycles (kMulGeneralLatency in the model's ProgramBuilderImpl.h), while
/// the native 8- and 16-bit multiply is 11. A quantized model is therefore
/// twelve times cheaper per multiply than the same program in i32, on top
/// of moving a quarter of the bytes.
struct OpCycles {
  double mul;
  double add;
};

OpCycles cyclesFor(Type elemTy) {
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    switch (intTy.getWidth()) {
    case 1:
    case 8:
    case 16:
      return {11.0, 22.0};
    case 32:
      return {132.0, 10.0};
    default:
      return {1421.0, 11.0}; // 64-bit
    }
  }
  if (auto fltTy = dyn_cast<FloatType>(elemTy)) {
    // No FPU: everything here is a software routine.
    return fltTy.getWidth() <= 32 ? OpCycles{1939.0, 683.0}
                                  : OpCycles{7195.0, 1002.0};
  }
  return {132.0, 10.0};
}

/// Aggregate arithmetic throughput of the array, in ops per second, for a
/// program whose multiplies are in `mulTy` and whose accumulation is in
/// `accTy`. Mixed precision is the reason those are separate: an
/// i8 x i8 -> i32 contraction multiplies at the i8 rate and accumulates at
/// the i32 one.
double deviceOpsPerSecond(int64_t dpuCount, Type mulTy, Type accTy) {
  const double dpus = static_cast<double>(dpuCount);
  const OpCycles mulC = cyclesFor(mulTy);
  const OpCycles addC = cyclesFor(accTy);
  // One multiply-accumulate is two ops; price the pair and halve.
  const double cyclesPerOp =
      std::max(1.0, (mulC.mul + addC.add) / (2.0 * kPipelineDepth));
  return dpus * kDpuClockHz / cyclesPerOp;
}

} // namespace

//===----------------------------------------------------------------------===//
// The gate
//===----------------------------------------------------------------------===//

/// The device half of the roofline at `dpus` devices: the greater of the
/// array's arithmetic and the traffic that crosses the wire, with the
/// transfer priced by the calibrated model at that same count. Both terms
/// move with `dpus`, and in opposite directions -- the arithmetic falls
/// while the transfer's fixed cost grows -- which is why the count belongs
/// in the signature rather than being read off the platform.
cinm::OffloadVerdict evaluateUpmemOffloadAt(const cinm::OffloadFootprint &f,
                                            int64_t dpus,
                                            const cinm::HostModel &host) {
  cinm::OffloadVerdict v;
  v.work = f.work;
  v.staticBytes = f.staticBytes;
  v.dynamicBytes = f.dynamicBytes;
  v.dynamicOutBytes = f.dynamicOutBytes;
  if (!f.known || dpus <= 0) {
    v.unknown = true;
    v.profitable = true;
    v.reason = "cost model could not measure this op";
    return v;
  }

  v.deviceOpsPerSecond = deviceOpsPerSecond(dpus, f.mulType, f.accType);
  // Scatter in, gather out, both paid on every invocation.
  v.transferSeconds =
      transferSeconds(f.dynamicBytes - f.dynamicOutBytes, dpus,
                      /*toDevice=*/true) +
      transferSeconds(f.dynamicOutBytes, dpus, /*toDevice=*/false);
  v.hostSeconds = cinm::hostRooflineSeconds(f, host);
  v.deviceSeconds = std::max(f.work / v.deviceOpsPerSecond, v.transferSeconds);
  v.profitable = v.deviceSeconds < v.hostSeconds;

  if (v.profitable) {
    v.reason = "resident operands amortize enough to beat the host";
  } else if (v.staticBytes == 0.0) {
    // Stated separately because it is the common case and the actionable
    // one: nothing is wrong with the op, it just has nothing to amortize.
    v.reason = "no static operand: every byte is re-sent, and the host "
               "streams them from DRAM faster than we scatter them";
  } else if (v.intensityOpsPerByte() > host.ridgeOpsPerByte()) {
    v.reason = "compute-bound on the host, which out-computes the array";
  } else {
    v.reason = "resident operands do not amortize enough to beat the host";
  }
  return v;
}

cinm::OffloadVerdict evaluateUpmemOffload(Operation *op,
                                          UpmemPlatformAttr platform,
                                          const cinm::HostModel &host) {
  // Priced at the whole array: the op is asked whether the device could ever
  // be worth it, and the count that answers that is a question for the menu
  // screen, which knows which counts the space admits (profileComputeBlock).
  return evaluateUpmemOffloadAt(cinm::measureOffloadFootprint(op),
                                platform.getMaxDpus(), host);
}

} // namespace mlir::upmem

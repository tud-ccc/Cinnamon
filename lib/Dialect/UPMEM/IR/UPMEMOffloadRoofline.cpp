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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
double deviceOpsPerSecond(UpmemPlatformAttr platform, Type mulTy, Type accTy) {
  const double dpus = platform.getMaxDpus();
  const OpCycles mulC = cyclesFor(mulTy);
  const OpCycles addC = cyclesFor(accTy);
  // One multiply-accumulate is two ops; price the pair and halve.
  const double cyclesPerOp =
      std::max(1.0, (mulC.mul + addC.add) / (2.0 * kPipelineDepth));
  return dpus * kDpuClockHz / cyclesPerOp;
}

//===----------------------------------------------------------------------===//
// Reading the op
//===----------------------------------------------------------------------===//

/// Bytes a value occupies: a static shape's elements, or a scalar's own width
/// (a row maximum or sum handed to an elementwise body travels as one
/// number). Nothing when the shape is dynamic or the type has no width.
std::optional<double> bytesOf(Value v) {
  Type type = v.getType();
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped) {
    if (type.isIndex())
      return 8.0;
    if (type.isIntOrFloat())
      return static_cast<double>(
          llvm::divideCeil(type.getIntOrFloatBitWidth(), 8));
    return std::nullopt;
  }
  if (!shaped.hasStaticShape())
    return std::nullopt;
  Type elem = shaped.getElementType();
  if (!elem.isIntOrFloat())
    return std::nullopt;
  return static_cast<double>(shaped.getNumElements()) *
         llvm::divideCeil(elem.getIntOrFloatBitWidth(), 8);
}

/// Arithmetic ops in one iteration of a linalg body. A contraction's body is
/// a multiply and an add, an elementwise body whatever it spells out; casts
/// and yields are not arithmetic and are not counted.
double arithOpsPerIteration(linalg::LinalgOp op) {
  double n = 0.0;
  op.getBlock()->walk([&](Operation *inner) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
            arith::TruncFOp, arith::SIToFPOp, arith::FPToSIOp,
            arith::IndexCastOp, arith::BitcastOp, linalg::YieldOp>(inner))
      return;
    if (isa<arith::ArithDialect, math::MathDialect>(inner->getDialect()))
      n += 1.0;
  });
  return std::max(1.0, n);
}

/// The element type the multiplies happen in, and the one the accumulation
/// happens in. For a linalg contraction the inputs carry the first and the
/// init the second; they differ exactly when the op is mixed precision.
std::pair<Type, Type> arithTypes(linalg::LinalgOp op) {
  Type mulTy, accTy;
  if (op.getNumDpsInputs() > 0)
    if (auto shaped = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType()))
      mulTy = shaped.getElementType();
  if (op.getNumDpsInits() > 0)
    if (auto shaped = dyn_cast<ShapedType>(op.getDpsInits()[0].getType()))
      accTy = shaped.getElementType();
  if (!mulTy)
    mulTy = accTy;
  if (!accTy)
    accTy = mulTy;
  return {mulTy, accTy};
}

/// Fill in work and traffic for a linalg op. Fails when a shape or a loop
/// bound is dynamic, which the caller reports as "unknown" rather than as a
/// rejection.
LogicalResult measureLinalg(linalg::LinalgOp op, OffloadVerdict &v, Type &mulTy,
                            Type &accTy) {
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  double iterations = 1.0;
  for (int64_t r : ranges) {
    if (ShapedType::isDynamic(r))
      return failure();
    iterations *= static_cast<double>(r);
  }
  v.work = iterations * arithOpsPerIteration(op);

  for (OpOperand &operand : op->getOpOperands()) {
    std::optional<double> bytes = bytesOf(operand.get());
    if (!bytes)
      return failure();
    // An init operand is not traffic the device pays: a contraction's
    // accumulator is produced and consumed on the device, and its value
    // comes back as the result, counted below.
    if (op.isDpsInit(&operand))
      continue;
    if (cinm::isStaticValue(operand.get()))
      v.staticBytes += *bytes;
    else
      v.dynamicBytes += *bytes;
  }
  // Results are gathered back on every invocation.
  for (Value result : op->getResults()) {
    std::optional<double> bytes = bytesOf(result);
    if (!bytes)
      return failure();
    v.dynamicBytes += *bytes;
    v.dynamicOutBytes += *bytes;
  }

  std::tie(mulTy, accTy) = arithTypes(op);
  return success(mulTy && accTy);
}

/// Same for the cinm gemm-like ops, which the prim flow still has in hand
/// when platforms are assigned (the whole-program flow has converted to
/// linalg by then). Work is 2*M*N*K, the operands are what they say.
LogicalResult measureGemmlike(cinm::GemmlikeOpInterface op, OffloadVerdict &v,
                              Type &mulTy, Type &accTy) {
  auto lhs = dyn_cast<ShapedType>(op.getLhs().getType());
  auto rhs = dyn_cast<ShapedType>(op.getRhs().getType());
  if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape())
    return failure();

  // Reduction extent is the lhs's last dimension in every variant; the
  // parallel extents are everything else the two operands span.
  double reduction = static_cast<double>(lhs.getDimSize(lhs.getRank() - 1));
  double parallel = static_cast<double>(lhs.getNumElements()) / reduction;
  double rhsParallel = static_cast<double>(rhs.getNumElements()) / reduction;
  // gemv's rhs is the vector, so it contributes no extra parallel extent.
  v.work = 2.0 * parallel *
           (rhs.getRank() > lhs.getRank() - 1 ? rhsParallel : 1.0) * reduction;

  for (Value operand : {op.getLhs(), op.getRhs()}) {
    std::optional<double> bytes = bytesOf(operand);
    if (!bytes)
      return failure();
    if (cinm::isStaticValue(operand))
      v.staticBytes += *bytes;
    else
      v.dynamicBytes += *bytes;
  }
  if (Value result = op.getGemmResult()) {
    std::optional<double> bytes = bytesOf(result);
    if (!bytes)
      return failure();
    v.dynamicBytes += *bytes;
    v.dynamicOutBytes += *bytes;
  }

  mulTy = lhs.getElementType();
  accTy = op.getAccumulatorElementType();
  return success();
}

/// Same for the cinm elementwise and reduce ops: one pass over the operands,
/// one arithmetic op per element. Both are pure traffic with no reuse, so
/// the roofline will reject them unless an operand is resident -- which is
/// the right answer and the reason `va` and `red` belong on the host.
LogicalResult measureCinmPointwise(Operation *op, OffloadVerdict &v,
                                   Type &mulTy, Type &accTy) {
  double elements = 0.0;
  for (Value operand : op->getOperands()) {
    std::optional<double> bytes = bytesOf(operand);
    if (!bytes)
      return failure();
    auto shaped = cast<ShapedType>(operand.getType());
    elements = std::max(elements, static_cast<double>(shaped.getNumElements()));
    if (cinm::isStaticValue(operand))
      v.staticBytes += *bytes;
    else
      v.dynamicBytes += *bytes;
    if (!mulTy)
      mulTy = shaped.getElementType();
  }
  for (Value result : op->getResults()) {
    // A reduce to a scalar has nothing to gather back worth counting, but
    // the shaped case does.
    if (std::optional<double> bytes = bytesOf(result)) {
      v.dynamicBytes += *bytes;
      v.dynamicOutBytes += *bytes;
    } else if (isa<ShapedType>(result.getType()))
      return failure();
  }
  if (!mulTy)
    return failure();
  accTy = mulTy;
  v.work = elements;
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// The gate
//===----------------------------------------------------------------------===//

cinm::OffloadVerdict evaluateUpmemOffload(Operation *op,
                                          UpmemPlatformAttr platform,
                                          const cinm::HostModel &host) {
  cinm::OffloadVerdict v;

  Type mulTy, accTy;
  LogicalResult measured = failure();
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    measured = measureLinalg(linalgOp, v, mulTy, accTy);
  else if (auto gemmlike = dyn_cast<cinm::GemmlikeOpInterface>(op))
    measured = measureGemmlike(gemmlike, v, mulTy, accTy);
  else if (isa<cinm::ElementwiseOp, cinm::ReduceOp>(op))
    measured = measureCinmPointwise(op, v, mulTy, accTy);

  if (failed(measured)) {
    // Dynamic shapes, or an op the model does not know how to read. Not
    // evidence of unprofitability, so do not reject on it.
    v.unknown = true;
    v.profitable = true;
    v.reason = "cost model could not measure this op";
    return v;
  }

  const double deviceOps = deviceOpsPerSecond(platform, mulTy, accTy);
  const int64_t dpus = platform.getMaxDpus();
  // Scatter in, gather out, both paid on every invocation.
  const double transfer =
      transferSeconds(v.dynamicBytes - v.dynamicOutBytes, dpus,
                      /*toDevice=*/true) +
      transferSeconds(v.dynamicOutBytes, dpus, /*toDevice=*/false);
  v.hostSeconds =
      std::max(v.work / host.opsPerSecond,
               (v.staticBytes + v.dynamicBytes) / host.dramBytesPerSecond);
  v.deviceSeconds = std::max(v.work / deviceOps, transfer);
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

} // namespace mlir::upmem

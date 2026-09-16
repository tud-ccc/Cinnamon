//===- CinmOffloadModel.h - roofline inputs for the offload gate -*- C++
//-*-===//
//
// The host side of the offload decision, and the verdict a platform returns
// when asked whether running an op on it would beat leaving it where it is.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_OFFLOAD_MODEL_H
#define CINM_OFFLOAD_MODEL_H

#include "llvm/ADT/StringRef.h"

namespace mlir::cinm {

/// What the host the accelerator hangs off can sustain. Both numbers are
/// properties of the machine, not of the program, so they are supplied by
/// whoever runs the pass rather than derived from the IR.
struct HostModel {
  /// Arithmetic throughput summed over every core, in ops per second.
  double opsPerSecond;
  /// Bandwidth from DRAM to those cores, in bytes per second.
  double dramBytesPerSecond;

  /// The arithmetic intensity at which the two balance. Below it the host is
  /// bandwidth-bound and an accelerator has something to beat; above it the
  /// host is compute-bound and, since it out-computes the DPUs by orders of
  /// magnitude, nothing on the device side can win.
  double ridgeOpsPerByte() const { return opsPerSecond / dramBytesPerSecond; }

  /// The bench machine: an Intel R2312WFTZSR holding 2x Xeon Silver 4216,
  /// 16 Cascade Lake cores each at 2.1 GHz, alongside 20 PIM modules.
  ///
  /// Arithmetic depends on the element type, and by a factor of four. An
  /// int32 multiply-add uses 16 AVX-512 lanes; Cascade Lake also has
  /// AVX512-VNNI, whose vpdpbusd does 64 int8 multiply-accumulates in one
  /// instruction. So the host sustains ~2.1e12 int32 ops/s but ~8.6e12 int8
  /// ops/s, and an experiment has to pass whichever matches its own
  /// numerics. The default below is the int32 figure.
  ///
  /// Bandwidth is the number to be careful with, for a reason particular to
  /// PIM machines: the PIM modules occupy DIMM slots. This one has 4x 64 GB
  /// DDR4 populating 2 slots per socket, so 4 of the 12 memory channels
  /// carry host DRAM, and the 4216's controller caps DDR4 at 2400 MT/s
  /// whatever the DIMMs are rated for -- 76.8 GB/s of peak, against the
  /// ~230 GB/s the same sockets would reach fully populated. That is not a
  /// handicapped baseline, it is what a PIM-heavy build costs, and the
  /// comparison is only honest if it is stated rather than discovered.
  ///
  /// The 15 GB/s kept here is the figure cpu_baseline.py quotes, which is
  /// 20% of that peak and so is almost certainly not a pure streaming
  /// measurement. It is retained as the default because it is the
  /// conservative end: a gate errs toward offloading, and every plausible
  /// measured value is higher, which only makes the host stronger. The
  /// verdicts this decides are insensitive across that whole range -- a
  /// RoBERTa projection needs the host below 6.4 GB/s, 8% of peak, before
  /// it would rather be on the array -- so replacing this with a real
  /// STREAM number is worth doing but will not move the answers.
  static HostModel benchMachine() { return {2.1e12, 15.0e9}; }
};

/// The terms of one offload decision. Kept whole rather than reduced to a
/// bool so the decision can be printed: a gate that silently drops an op
/// from the device is the kind of thing that is discovered late.
struct OffloadVerdict {
  /// Whether the device is predicted to beat the host on this op.
  bool profitable = false;
  /// True when the model could not read the op at all (dynamic shapes, an
  /// op kind it does not know). Callers treat this as "do not reject": a
  /// gate that cannot see is not evidence of unprofitability.
  bool unknown = false;

  /// Arithmetic operations the op performs, per invocation.
  double work = 0.0;
  /// Bytes of operands that are the same on every invocation, so the device
  /// can hold them resident while the host must stream them from DRAM.
  double staticBytes = 0.0;
  /// Bytes that cross the wire on every invocation either way: non-static
  /// operands, plus the results that come back.
  double dynamicBytes = 0.0;
  /// The part of `dynamicBytes` that travels device-to-host. Kept apart
  /// because a gather and a scatter of the same size do not cost the same.
  double dynamicOutBytes = 0.0;

  /// Predicted seconds per invocation on each side.
  double hostSeconds = 0.0;
  double deviceSeconds = 0.0;

  /// Why it came out the way it did, for diagnostics.
  llvm::StringRef reason;

  double intensityOpsPerByte() const {
    double bytes = staticBytes + dynamicBytes;
    return bytes > 0.0 ? work / bytes : 0.0;
  }
};

} // namespace mlir::cinm

#endif // CINM_OFFLOAD_MODEL_H

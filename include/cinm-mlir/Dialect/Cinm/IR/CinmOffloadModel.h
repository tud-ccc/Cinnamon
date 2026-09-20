//===- CinmOffloadModel.h - What the host costs -----------------*- C++ -*-===//
//
// The host side of the offload decision and of the cost model, and the
// verdict a platform returns when asked whether running an op on it would
// beat leaving it where it is.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_OFFLOAD_MODEL_H
#define CINM_OFFLOAD_MODEL_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"

namespace mlir::cinm {

/// What the host the accelerators hang off can sustain, and what running
/// code on it costs. These are properties of the machine, not of the
/// program: they are the parameters of the `#cinm.host_platform` in scope
/// (HostPlatformAttr::getInScope). The defaults are the bench machine; the
/// attribute's description says where each number comes from, under the
/// same name in snake case.
struct HostModel {
  // The offload roofline.

  /// Arithmetic throughput summed over every core, in ops per second.
  double opsPerSecond = 2.1e12;
  /// Bandwidth from DRAM to those cores, in bytes per second.
  double dramBytesPerSecond = 23.0e9;

  // The cost model's host code.

  /// One scalar add in straight-line code, in ns.
  double scalarOpNs = 3.0;
  /// One vector instruction, in ns, and the bytes it operates on.
  double vectorOpNs = 0.5;
  double vectorBytes = 64.0;
  /// What one core streams through a loop nest, in bytes per second.
  double streamBytesPerSecond = 10.8e9;
  /// The rate of a strided repack between two layouts, in bytes per second.
  double copyBytesPerSecond = 0.63e9;

  /// The arithmetic intensity at which the two balance. Below it the host is
  /// bandwidth-bound and an accelerator has something to beat; above it the
  /// host is compute-bound and, since it out-computes the DPUs by orders of
  /// magnitude, nothing on the device side can win.
  double ridgeOpsPerByte() const { return opsPerSecond / dramBytesPerSecond; }

  bool operator==(const HostModel &) const = default;
};

/// Lets a HostModel be an attribute parameter: attribute storage is uniqued
/// by hash, and llvm::hash_value has no overload for double.
inline llvm::hash_code hash_value(const HostModel &m) {
  auto bits = [](double d) { return llvm::bit_cast<uint64_t>(d); };
  return llvm::hash_combine(bits(m.opsPerSecond), bits(m.dramBytesPerSecond),
                            bits(m.scalarOpNs), bits(m.vectorOpNs),
                            bits(m.vectorBytes), bits(m.streamBytesPerSecond),
                            bits(m.copyBytesPerSecond));
}

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

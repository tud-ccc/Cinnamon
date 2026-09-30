//===- UPMEMTransferFootprint.h - Distinct host bytes ---------*- C++ -*-===//
//
// How much of its host buffer a one-block-per-DPU transfer actually touches.
// A scatter whose map sends several DPUs to the same slice (a vector every
// DPU of a row needs) writes D blocks but reads far fewer distinct bytes, and
// its cost follows the latter (upmem_cm::scatterReplicatedCostMs).
//
//===----------------------------------------------------------------------===//

#ifndef CINM_MLIR_DIALECT_UPMEM_IR_UPMEMTRANSFERFOOTPRINT_H
#define CINM_MLIR_DIALECT_UPMEM_IR_UPMEMTRANSFERFOOTPRINT_H

#include "mlir/IR/AffineMap.h"

#include <cstdint>
#include <optional>

namespace mlir::upmem {

class ScatterOnArrayOp;

/// Number of distinct host elements read by `numDpus` DPUs that each take
/// `count` contiguous elements starting at `map(dpu)`, a start index into a
/// buffer with the given element `strides`. The union of the D intervals, by
/// enumeration: one map evaluation per DPU, then a sort. None when the map
/// has symbols or is not one-dimensional.
std::optional<int64_t> uniqueHostElements(AffineMap map, int64_t numDpus,
                                          ArrayRef<int64_t> strides,
                                          int64_t count);

/// The same for a scatter, in bytes. None when the host buffer's strides are
/// not static.
std::optional<int64_t> uniqueHostBytes(ScatterOnArrayOp op);

} // namespace mlir::upmem

#endif // CINM_MLIR_DIALECT_UPMEM_IR_UPMEMTRANSFERFOOTPRINT_H

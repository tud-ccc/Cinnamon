//===- CnmBufferLevel.h - Resolving the cnm-buffer-level option ----------===//
//
// Shared by the conversions that allocate `cnm.buffer`s
// (`--convert-cinm-to-cnm`, `--convert-linalg-to-cnm`): both take a memory
// level by name and have to resolve it against each op's own accelerator,
// since the mapping from level name to memory-space attribute is the
// platform's business.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h>

#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>

namespace mlir::cnm {

/// The memory level selected by `cnm-buffer-level`, resolved against a
/// particular accelerator.
struct BufferLevel {
  /// Goes in the `cnm.buffer` type's level field and in the memory space of
  /// the launch body's memrefs. Null when no level was requested.
  cinm::CinmLevelAttrInterface space;
  /// How many bytes of that level one leaf of the workgroup may use.
  int64_t bytesPerLeaf;
};

/// Bytes of `level` available to a single leaf of the workgroup.
///
/// `getWorkgroupMemoryLevels()` is indexed by workgroup dimension: entry `i`
/// lists the levels owned by a node at dimension `i`. Everything below that
/// dimension shares the level, so one leaf's share is the level's capacity
/// divided by the number of leaves under one such node. For UPMEM's leaf level
/// this reproduces `bufferSizeOfLeaf()` (WRAM per DPU, divided by tasklets).
std::optional<int64_t> capacityPerLeaf(CnmAcceleratorAttrInterface acc,
                                       StringRef levelName);

/// Resolve a `cnm-buffer-level` option value against the accelerator's
/// platform, reporting on `op`.
///
/// An empty name yields a null level and the accelerator's own leaf budget,
/// which is what every buffer got before this option existed: the backend
/// conversion then picks the staging itself.
FailureOr<BufferLevel> resolveBufferLevel(StringRef levelName,
                                          CnmAcceleratorAttrInterface acc,
                                          Operation *op);

} // namespace mlir::cnm

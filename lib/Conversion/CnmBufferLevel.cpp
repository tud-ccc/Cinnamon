//===- CnmBufferLevel.cpp - Resolving the cnm-buffer-level option --------===//

#include "cinm-mlir/Conversion/CnmBufferLevel.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Diagnostics.h>

using namespace mlir;
using namespace mlir::cnm;

std::optional<int64_t>
mlir::cnm::capacityPerLeaf(CnmAcceleratorAttrInterface acc,
                           StringRef levelName) {
  auto wgShape = acc.getWorkgroupShape();
  auto levelsPerDim = acc.getWorkgroupMemoryLevels();
  for (auto [dim, levels] : llvm::enumerate(levelsPerDim)) {
    for (cinm::CinmLevelDefAttr level : levels) {
      if (level.getName() != levelName)
        continue;
      int64_t leaves = 1;
      for (size_t below = dim + 1; below < wgShape.size(); ++below)
        leaves *= wgShape[below];
      return leaves ? level.getSizeInBytes() / leaves : 0;
    }
  }
  return std::nullopt;
}

FailureOr<BufferLevel>
mlir::cnm::resolveBufferLevel(StringRef levelName,
                              CnmAcceleratorAttrInterface acc, Operation *op) {
  if (levelName.empty())
    return BufferLevel{{}, acc.bufferSizeOfLeaf()};

  auto platform = acc.getPlatform();
  if (!platform)
    return op->emitOpError("cannot resolve memory level '")
           << levelName << "': the accelerator declares no platform";

  cinm::CinmLevelDefAttr def = platform.getLevel(levelName);
  if (!def) {
    auto diag = op->emitOpError("unknown memory level '")
                << levelName << "' for platform '" << platform.getName()
                << "'; known levels are ";
    llvm::interleaveComma(platform.getLevels(), diag,
                          [&](cinm::CinmLevelDefAttr l) {
                            diag << "'" << l.getName().getValue() << "'";
                          });
    return diag;
  }

  cinm::CinmLevelAttrInterface space = platform.getMemrefMemspace(def);
  if (!space)
    return op->emitOpError("platform '")
           << platform.getName()
           << "' does not provide a memref memory space for level '"
           << levelName << "'";

  // The buffer budget has to follow the level: sizing an MRAM buffer by the
  // WRAM budget would reject perfectly good tiles.
  std::optional<int64_t> capacity = capacityPerLeaf(acc, levelName);
  if (!capacity)
    return op->emitOpError("level '")
           << levelName
           << "' is not one of the workgroup's memory levels on this "
              "accelerator";
  return BufferLevel{space, *capacity};
}

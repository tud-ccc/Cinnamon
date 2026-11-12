/// Bracket each device node of a solved graph with node timers.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include <cinm-mlir/Dialect/Cinm/IR/CinmBase.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>

#include <mlir/IR/Builders.h>

#include <llvm/ADT/Twine.h>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMTIMECOMPUTEBLOCKSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

} // namespace mlir::upmem

using namespace mlir;

namespace {

/// "node:<graph>:<class>:<member>" from a block's cinm.graph_alloc, or
/// nothing when it lacks one of them.
std::optional<std::string> nodeName(DictionaryAttr alloc) {
  auto graph = alloc.getAs<StringAttr>("graph");
  auto cls = alloc.getAs<IntegerAttr>("class");
  auto member = alloc.getAs<IntegerAttr>("member");
  if (!graph || !cls || !member)
    return std::nullopt;
  return ("node:" + graph.getValue() + ":" + Twine(cls.getInt()) + ":" +
          Twine(member.getInt()))
      .str();
}

struct UpmemTimeComputeBlocksPass
    : upmem::impl::UpmemTimeComputeBlocksPassBase<UpmemTimeComputeBlocksPass> {
  void runOnOperation() override {
    if (failed(cinm::requireIsolatedComputeOps(getOperation(), getArgument())))
      return signalPassFailure();
    getOperation()->walk([&](cinm::ComputeBlockOp block) {
      if (!llvm::isa_and_nonnull<upmem::UpmemAcceleratorAttr>(
              block.getAcceleratorAttr()))
        return;
      auto alloc = block->getAttrOfType<DictionaryAttr>(
          cinm::CinmDialect::GRAPH_ALLOC_NAME);
      if (!alloc)
        return;
      // A host member is outlined before this runs; not a device node either
      // way.
      if (auto placement = alloc.getAs<StringAttr>("placement");
          placement && placement.getValue() == "host")
        return;
      std::optional<std::string> name = nodeName(alloc);
      if (!name) {
        block.emitWarning("cinm.graph_alloc names no graph, class and member: "
                          "the node is not timed");
        return;
      }
      Block &body = block.getBody().front();
      // After the accelerator declarations, which open a block.
      auto first = llvm::find_if(body, [](Operation &op) {
        return !llvm::isa<cinm::AcceleratorOp>(op);
      });
      OpBuilder builder(&body, first);
      upmem::NodeBeginOp::create(builder, block.getLoc(), *name);
      builder.setInsertionPoint(body.getTerminator());
      upmem::NodeEndOp::create(builder, block.getLoc(), *name);
    });
  }
};

} // namespace

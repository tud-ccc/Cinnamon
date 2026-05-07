#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMINFERTILESIZESPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

struct CinmInferTileSizesPass
    : public impl::CinmInferTileSizesPassBase<CinmInferTileSizesPass> {
  using Base::Base;

  void runOnOperation() final {
    Operation *root = getOperation();

    root->walk([&](cinm::CinmTilingInterface op) {
      // Skip ops already annotated or explicitly excluded from tiling.
      if (op->hasAttr(CinmDialect::TILING_FACTORS_NAME))
        return;
      if (op->hasAttr(CinmDialect::NOTILE_NAME))
        return;

      auto accelerator = cinm::getEnclosingAccelerator(op);
      if (!accelerator)
        return;

      SmallVector<int64_t> tilingFactors;
      auto diag =
          accelerator.computeTilingFactors(op.getOperation(), tilingFactors);
      if (!diag.succeeded()) {
        // Silence; the tiling pass will skip ops without cinm.tile_sizes.
        (void)diag.silence();
        return;
      }

      Builder b(op->getContext());
      op->setAttr(CinmDialect::TILING_FACTORS_NAME,
                  b.getDenseI64ArrayAttr(tilingFactors));
    });
  }
};

} // namespace mlir::cinm

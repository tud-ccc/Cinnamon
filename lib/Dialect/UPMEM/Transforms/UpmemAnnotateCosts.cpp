#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>

#include <fstream>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <variant>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMANNOTATECOSTSPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

namespace {

// Wrap `s` in double quotes, doubling any embedded quotes (RFC 4180).
std::string csvQuote(llvm::StringRef s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"')
      out += '"';
    out += c;
  }
  out += '"';
  return out;
}

struct UpmemAnnotateCostsPass
    : impl::UpmemAnnotateCostsPassBase<UpmemAnnotateCostsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *container = getOperation();

    auto timeout = std::chrono::milliseconds(evalTimeoutMs);
    std::unique_ptr<UpmemSimulator> sim =
        createSimulator(simulator, true, timeout, programDumpDir);

    // One row per (block, category, label, excluded). `excluded` rows are
    // costs the program pays that block_total_ms does not include (see
    // SimCost), so a consumer that wants the undiscounted view has to add
    // them back rather than read the block total.
    struct CsvRow {
      unsigned blockId;
      std::string location;
      llvm::StringRef category;
      std::string label;
      double costMs;
      bool excluded;
      double blockTotalMs;
    };
    llvm::SmallVector<CsvRow> rows;
    unsigned blockId = 0;

    container->walk([&](cinm::ComputeBlockOp computeBlock) {
      auto res = sim->simulate(computeBlock.getBody());
      if (std::holds_alternative<DiagnosedSilenceableFailure>(res)) {
        (void)std::get<DiagnosedSilenceableFailure>(res).checkAndReport();
        signalPassFailure();
        return;
      }

      computeBlock->setAttr(kSimCostAttr,
                            FloatAttr::get(Float64Type::get(&getContext()),
                                           std::get<SimCost>(res).total()));

      if (!costsCsv.empty()) {
        const SimCost &cost = std::get<SimCost>(res);
        std::string location;
        llvm::raw_string_ostream(location) << computeBlock.getLoc();
        double blockTotal = cost.total();
        cost.forEachEntry([&](CostCategory category, llvm::StringRef label,
                              double value, bool excluded) {
          rows.push_back(
              {blockId, location, costCategoryName(category),
               label.empty() ? costCategoryName(category).str() : label.str(),
               value, excluded, blockTotal});
        });
      }
      ++blockId;
    });

    if (costsCsv.empty())
      return;

    std::ofstream out(costsCsv);
    if (!out) {
      container->emitWarning() << "upmem-annotate-costs: could not open '"
                               << costsCsv << "' for writing";
      return;
    }
    out << "block_id,location,category,label,cost_ms,excluded,block_total_ms\n";
    for (auto &r : rows)
      out << r.blockId << "," << csvQuote(r.location) << "," << r.category.str()
          << "," << csvQuote(r.label) << "," << r.costMs << ","
          << (r.excluded ? 1 : 0) << "," << r.blockTotalMs << "\n";
  }
};

} // namespace

} // namespace mlir::upmem

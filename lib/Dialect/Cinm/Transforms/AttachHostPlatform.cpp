#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMATTACHHOSTPLATFORMPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

struct CinmAttachHostPlatformPass
    : public impl::CinmAttachHostPlatformPassBase<CinmAttachHostPlatformPass> {
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Every option a run leaves at its default, by name and value: the
    // defaults are one machine's, and a pipeline that means it says so.
    std::string defaulted;
    llvm::raw_string_ostream os(defaulted);
    auto note = [&](const auto &option) {
      if (option.hasValue())
        return;
      os << (defaulted.empty() ? "" : ", ") << option.getArgStr() << "="
         << static_cast<double>(option);
    };
    note(opsPerSecond);
    note(dramBytesPerSecond);
    note(scalarOpNs);
    note(vectorOpNs);
    note(vectorBytes);
    note(streamBytesPerSecond);
    note(copyBytesPerSecond);
    if (!defaulted.empty())
      module.emitWarning()
          << getArgument()
          << ": host parameters left at their bench-machine defaults ("
          << defaulted
          << "); pass the host this program is priced against explicitly";

    HostModel model;
    model.opsPerSecond = opsPerSecond;
    model.dramBytesPerSecond = dramBytesPerSecond;
    model.scalarOpNs = scalarOpNs;
    model.vectorOpNs = vectorOpNs;
    model.vectorBytes = vectorBytes;
    model.streamBytesPerSecond = streamBytesPerSecond;
    model.copyBytesPerSecond = copyBytesPerSecond;
    auto host = HostPlatformAttr::getChecked([&] { return module.emitError(); },
                                             &getContext(), model);
    if (!host)
      return signalPassFailure();

    module.walk([&](FunctionOpInterface func) {
      auto listed =
          func->getAttrOfType<ArrayAttr>(CinmDialect::AVAILABLE_PLATFORMS_NAME);
      if (listed && llvm::any_of(listed, llvm::IsaPred<HostPlatformAttr>))
        return;
      // A function with no list of its own still runs on the host when it
      // offloads: when it holds a compute op, or an op that lists platforms.
      if (!listed &&
          !func->walk([](Operation *op) {
                 return isa<ComputeOpInterface>(op) ||
                                op->hasAttr(
                                    CinmDialect::AVAILABLE_PLATFORMS_NAME)
                            ? WalkResult::interrupt()
                            : WalkResult::advance();
               })
               .wasInterrupted())
        return;
      SmallVector<Attribute> platforms{host};
      if (listed)
        llvm::append_range(platforms, listed);
      func->setAttr(CinmDialect::AVAILABLE_PLATFORMS_NAME,
                    ArrayAttr::get(&getContext(), platforms));
    });
  }
};

} // namespace

} // namespace mlir::cinm

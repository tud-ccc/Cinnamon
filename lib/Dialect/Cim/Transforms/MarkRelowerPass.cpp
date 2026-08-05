#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::cim {

#define GEN_PASS_DEF_CIMMARKRELOWERPASS
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h.inc"

namespace {

static void normalizeOpToken(StringRef in, std::string &out) {
  StringRef s = in.trim();
  if (s.empty()) {
    out.clear();
    return;
  }

  if (s.starts_with("cim.")) {
    out.assign(s.data(), s.size());
    return;
  }
  if (s.starts_with("op.")) {
    out = ("cim." + s).str();
    return;
  }
  out = ("cim.op." + s).str();
}

static LogicalResult parseOpsList(StringRef list, llvm::StringSet<> &out) {
  out.clear();
  if (list.empty())
    return failure();

  SmallVector<StringRef, 16> tokens;
  list.split(tokens, ',', -1, false);
  if (tokens.empty())
    return failure();

  for (StringRef t : tokens) {
    std::string canon;
    normalizeOpToken(t, canon);
    if (!canon.empty())
      out.insert(canon);
  }
  if (out.empty())
    return failure();

  return success();
}

struct CimMarkRelowerPass
    : public impl::CimMarkRelowerPassBase<CimMarkRelowerPass> {
  using Base::Base;

  void runOnOperation() final {
    if (opsOpt.empty()) {
      getOperation()->emitError()
          << "cim-mark-relower: missing --ops "
             "(e.g. --ops=add,gemm or --ops=cim.op.add)";
      signalPassFailure();
      return;
    }

    llvm::StringSet<> targets;
    if (failed(parseOpsList(StringRef(opsOpt), targets))) {
      getOperation()->emitError()
          << "cim-mark-relower: unable to parse --ops='" << opsOpt
          << "' (comma-separated; accept short 'add'/'gemm', 'op.add', or "
             "fully-qualified 'cim.op.add')";
      signalPassFailure();
      return;
    }

    Builder b(&getContext());
    auto trueAttr = b.getBoolAttr(true);

    getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (targets.contains(name))
        op->setAttr("relower", trueAttr);
    });
  }
};

} // namespace
} // namespace mlir::cim

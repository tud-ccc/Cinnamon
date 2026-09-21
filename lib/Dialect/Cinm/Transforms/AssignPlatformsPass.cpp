#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/StringSet.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Pass/Pass.h>

#include <mutex>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMASSIGNPLATFORMSPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// `file:line:col` for a location that has one, else the location printed.
std::string locationString(Location loc) {
  if (auto fileLoc = loc->findInstanceOf<FileLineColLoc>())
    return (fileLoc.getFilename().strref() + ":" + Twine(fileLoc.getLine()) +
            ":" + Twine(fileLoc.getColumn()))
        .str();
  std::string s;
  llvm::raw_string_ostream os(s);
  loc.print(os);
  return s;
}

/// How many times `op` runs per call of its function: the product of the
/// static trip counts of the loops around it. None when one of them is not
/// static. A conditional counts as always taken.
std::optional<int64_t> executionsPerCall(Operation *op) {
  int64_t n = 1;
  for (Operation *parent = op->getParentOp();
       parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp()) {
    auto loop = dyn_cast<LoopLikeOpInterface>(parent);
    if (!loop)
      continue;
    std::optional<APInt> trips = loop.getStaticTripCount();
    if (!trips)
      return std::nullopt;
    n *= trips->getZExtValue();
  }
  return n;
}

std::string printed(Attribute attr) {
  std::string s;
  llvm::raw_string_ostream os(s);
  attr.print(os);
  return s;
}

/// One offload decision as a JSON object: which op, on which platform, what
/// was decided, and every term of the roofline that decided it.
llvm::json::Object decisionRecord(func::FuncOp func, Operation *op,
                                  CinmPlatformAttrInterface platform,
                                  const cinm::HostModel &host,
                                  const cinm::OffloadVerdict &v, bool gated,
                                  bool requested, bool offloaded) {
  llvm::json::Array types;
  for (Value operand : op->getOperands()) {
    std::string s;
    llvm::raw_string_ostream os(s);
    operand.getType().print(os);
    types.push_back(std::move(s));
  }
  llvm::json::Object record{
      {"func", func.getSymName()},
      {"loc", locationString(op->getLoc())},
      {"op", op->getName().getStringRef()},
      {"operand_types", std::move(types)},
      {"platform", printed(platform)},
      {"gated", gated},
      {"requested", requested},
      {"offloaded", offloaded},
      {"profitable", v.profitable},
      {"unknown", v.unknown},
      {"reason", v.reason},
      {"work_ops", v.work},
      {"static_bytes", v.staticBytes},
      {"dynamic_bytes", v.dynamicBytes},
      {"dynamic_out_bytes", v.dynamicOutBytes},
      {"host_seconds", v.hostSeconds},
      {"device_seconds", v.deviceSeconds},
      {"device_ops_per_second", v.deviceOpsPerSecond},
      {"transfer_seconds", v.transferSeconds},
      {"host_ops_per_second", host.opsPerSecond},
      {"host_dram_bytes_per_second", host.dramBytesPerSecond},
  };
  if (auto tag = op->getAttrOfType<StringAttr>(CinmDialect::DEBUG_TAG_NAME))
    record["tag"] = tag.getValue();
  if (std::optional<int64_t> executions = executionsPerCall(op))
    record["executions"] = *executions;
  else
    record["executions"] = nullptr;
  return record;
}

/// Appends `lines` to `path`. Functions are processed in parallel, so the
/// write is serialized, and the first write to a path in this process
/// truncates it: a rerun replaces the previous dump instead of extending it.
LogicalResult appendDecisions(StringRef path, ArrayRef<std::string> lines) {
  static std::mutex mutex;
  static llvm::StringSet<> started;
  std::lock_guard<std::mutex> lock(mutex);
  const bool first = started.insert(path).second;
  std::error_code ec;
  llvm::raw_fd_ostream os(
      path, ec, first ? llvm::sys::fs::OF_Text : llvm::sys::fs::OF_Append);
  if (ec)
    return failure();
  for (const std::string &line : lines)
    os << line << '\n';
  return success();
}

} // namespace

struct CinmAssignPlatformsPass
    : public impl::CinmAssignPlatformsPassBase<CinmAssignPlatformsPass> {
  using Base::Base;

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    auto platformsAttr =
        func->getAttrOfType<ArrayAttr>(CinmDialect::AVAILABLE_PLATFORMS_NAME);
    if (!platformsAttr)
      return;

    SmallVector<CinmPlatformAttrInterface> platforms;
    for (Attribute attr : platformsAttr) {
      if (auto platform = llvm::dyn_cast<CinmPlatformAttrInterface>(attr))
        platforms.push_back(platform);
    }
    if (platforms.empty())
      return;

    IRRewriter rewriter(func->getContext());

    SmallVector<Operation *> opsToWrap;
    func.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("cinm.op.") &&
          op->getName().getDialectNamespace() != "linalg")
        return;
      if (op->getParentOfType<cinm::ComputeBlockOp>())
        return;
      opsToWrap.push_back(op);
    });

    cinm::HostModel host = HostPlatformAttr::getInScope(func).getModel();
    if (hostOpsPerSecond > 0)
      host.opsPerSecond = hostOpsPerSecond;
    if (hostDramBytesPerSecond > 0)
      host.dramBytesPerSecond = hostDramBytesPerSecond;

    // The verdict is computed whenever it is either enforced or recorded, so
    // an ungated run can still be dumped.
    const bool dumping = !dumpDecisions.empty();
    SmallVector<std::string> decisions;

    for (Operation *op : opsToWrap) {
      // An op the program already put inside a cinm.compute was offloaded on
      // purpose. Capability still has to hold, but profitability is not
      // second-guessed: this is the override for the cases the roofline
      // rejects and the author wants anyway.
      const bool explicitlyRequested = op->getParentOfType<cinm::ComputeOp>();

      SmallVector<Attribute> interested;
      for (auto platform : platforms) {
        if (!platform.isOffloadingTarget(op))
          continue;
        const bool gated = requireProfitable && !explicitlyRequested;
        cinm::OffloadVerdict verdict;
        if (gated || dumping)
          verdict = platform.evaluateOffload(op, host);
        if (dumping) {
          const bool offloaded = !gated || verdict.profitable;
          decisions.push_back(llvm::formatv(
              "{0}", llvm::json::Value(decisionRecord(
                         func, op, platform, host, verdict, requireProfitable,
                         explicitlyRequested, offloaded))));
        }
        if (gated) {
          if (!verdict.profitable) {
            std::string terms;
            llvm::raw_string_ostream os(terms);
            os << llvm::format(
                "work %.3g ops, resident %.3g B, per-call %.3g B, host %.3g s "
                "vs device %.3g s",
                verdict.work, verdict.staticBytes, verdict.dynamicBytes,
                verdict.hostSeconds, verdict.deviceSeconds);
            // Anchored to the location, not the op: a diagnostic that
            // carries the op prints it, and printing verifies and numbers
            // the whole enclosing function first -- once per rejected op,
            // that is quadratic in the program.
            emitRemark(op->getLoc())
                << "not offloaded: " << verdict.reason << " (" << terms << ")";
            continue;
          }
        }
        interested.push_back(platform);
      }
      if (interested.empty())
        continue;

      ComputeOp computeOp = wrapOperationInCompute(op, rewriter);
      computeOp->setAttr(CinmDialect::AVAILABLE_PLATFORMS_NAME,
                         ArrayAttr::get(func->getContext(), interested));
    }

    if (dumping && failed(appendDecisions(dumpDecisions, decisions))) {
      func.emitError() << "cannot write offload decisions to '" << dumpDecisions
                       << "'";
      signalPassFailure();
    }
  }
};

} // namespace mlir::cinm

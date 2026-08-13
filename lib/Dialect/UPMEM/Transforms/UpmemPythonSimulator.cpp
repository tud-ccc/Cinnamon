#include "SimulatorBase.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"
#include "upmem_cost_model/Types.h"

#include <cstdint>
#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <upmem_cost_model/ProgramBuilder.h>

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "cinm-inference"

using namespace mlir;
using namespace upmem;
using mlir::cinm::utils::Maybe;
using upmem_cm::ArithOp;
using upmem_cm::DType;
using upmem_cm::MemSpace;
using upmem_cm::ProgramBuilder;

namespace {

struct SubviewInfo {
  ProgramBuilder::BufId buf_id;
  bool iv_indexed;
  int64_t n_elems;
};

struct DpuTranslator {
  ProgramBuilder &builder;
  llvm::DenseMap<Value, ProgramBuilder::BufId> buf_map;
  llvm::DenseMap<Value, ProgramBuilder::ValId> val_map;
  llvm::DenseMap<Value, SubviewInfo> sv_map;
  llvm::SmallVector<Value, 4> iv_stack;
  int buf_ctr = 0;

  explicit DpuTranslator(ProgramBuilder &b) : builder(b) {}

  // ── Type helpers ──────────────────────────────────────────────────────────

  upmem_cm::DType mlirTypeToDtype(Type ty) {
    if (ty.isF32())
      return upmem_cm::DType::F32;
    if (ty.isF64())
      return upmem_cm::DType::F64;
    // F16/BF16 → F32 (UPMEM cost model has no 16-bit float type)
    if (ty.isF16() || ty.isBF16())
      return upmem_cm::DType::F32;
    if (auto it = dyn_cast<IntegerType>(ty)) {
      unsigned w = it.getWidth();
      bool s = !it.isUnsigned();
      if (w == 8)
        return s ? upmem_cm::DType::I8 : upmem_cm::DType::U8;
      if (w == 16)
        return s ? upmem_cm::DType::I16 : upmem_cm::DType::U16;
      if (w == 32)
        return s ? upmem_cm::DType::I32 : upmem_cm::DType::U32;
      if (w == 64)
        return s ? upmem_cm::DType::I64 : upmem_cm::DType::U64;
    }
    return upmem_cm::DType::I64; // index or unknown
  }

  MemSpace memSpaceOf(MemRefType mrt) {
    if (auto attr = dyn_cast_or_null<DpuMemSpaceAttr>(mrt.getMemorySpace()))
      if (attr.getValue() == DpuMemSpace::MRAM)
        return MemSpace::MRAM;
    return MemSpace::WRAM;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  bool ivIndexedIn(ValueRange values) {
    if (iv_stack.empty())
      return false;
    Value iv = iv_stack.back();
    for (Value v : values)
      if (v == iv)
        return true;
    return false;
  }

  int64_t getConstInt(Value v) {
    if (auto *def = v.getDefiningOp())
      if (auto c = dyn_cast<arith::ConstantOp>(def))
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
          return ia.getInt();
    return 0;
  }

  // ── Per-op translation ────────────────────────────────────────────────────

  void translatePwramAlloc(memref::AllocaOp op) {
    auto mrt = op.getType();
    std::string name = "wram_" + std::to_string(buf_ctr++);
    buf_map[op.getResult()] = builder.addBuffer(
        name, MemSpace::WRAM, mlirTypeToDtype(mrt.getElementType()));
  }

  void translateStaticAlloc(StaticAllocOp op) {
    auto mrt = llvm::cast<MemRefType>(op.getBuffer().getType());
    std::string name = op.getSymName() ? op.getSymName()->str()
                                       : ("buf_" + std::to_string(buf_ctr++));
    buf_map[op.getBuffer()] = builder.addBuffer(
        name, memSpaceOf(mrt), mlirTypeToDtype(mrt.getElementType()));
  }

  void translateTaskletDim(TaskletDimOp op) {
    // tid is not a loop IV; store a dummy const so downstream val_map lookups
    // can still find it without crashing (e.g. when tid feeds a BinOp)
    val_map[op.getResult()] = builder.createConst(0, upmem_cm::DType::I64);
  }

  void translateSubView(memref::SubViewOp op) {
    auto srcIt = buf_map.find(op.getSource());
    if (srcIt == buf_map.end())
      return;

    bool iv_indexed = false;
    for (OpFoldResult off : op.getMixedOffsets())
      if (auto v = off.dyn_cast<Value>())
        if (ivIndexedIn({v})) {
          iv_indexed = true;
          break;
        }

    int64_t n = 1;
    for (OpFoldResult sz : op.getMixedSizes())
      if (auto cv = getConstantIntValue(sz))
        n *= *cv;

    sv_map[op.getResult()] = {srcIt->second, iv_indexed, n};
  }

  void translateExpandShape(memref::ExpandShapeOp op) {
    auto srcIt = buf_map.find(op.getSrc());
    if (srcIt != buf_map.end()) {
      bool iv_indexed = false;
      int64_t n = computeProduct(op.getStaticOutputShape());

      sv_map[op.getResult()] = {srcIt->second, iv_indexed, n};
      return;
    }

    auto subv = sv_map.find(op.getSrc());
    if (subv == sv_map.end()) {
      return;
    }

    sv_map[op.getResult()] = subv->second;
    return;
  }

  void translateReinterpretCast(memref::ReinterpretCastOp op) {
    bool iv_indexed = false;
    for (OpFoldResult off : op.getMixedOffsets())
      if (auto v = off.dyn_cast<Value>())
        if (ivIndexedIn({v})) {
          iv_indexed = true;
          break;
        }

    int64_t n = 1;
    for (OpFoldResult sz : op.getMixedSizes())
      if (auto cv = getConstantIntValue(sz))
        n *= *cv;

    if (auto srcIt = buf_map.find(op.getSource()); srcIt != buf_map.end()) {
      sv_map[op.getResult()] = {srcIt->second, iv_indexed, n};
      return;
    }

    if (auto subv = sv_map.find(op.getSource()); subv != sv_map.end()) {
      iv_indexed |= subv->second.iv_indexed;
      sv_map[op.getResult()] = {subv->second.buf_id, iv_indexed, n};
      return;
    }
  }

  void translateLocalTransfer(LocalTransferOp op) {
    Value srcVal = op.getSource();
    Value dstVal = op.getTarget();

    ProgramBuilder::BufId srcId = 0, dstId = 0;
    bool src_iv = false, dst_iv = false;
    int64_t n_elems = 1;

    if (sv_map.count(srcVal)) {
      auto &sv = sv_map.at(srcVal);
      srcId = sv.buf_id;
      src_iv = sv.iv_indexed;
      n_elems = sv.n_elems;
    } else if (buf_map.count(srcVal)) {
      srcId = buf_map.at(srcVal);
      if (auto mrt = dyn_cast<MemRefType>(srcVal.getType()))
        if (mrt.hasStaticShape())
          n_elems = mrt.getNumElements();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[upmem-cpp-sim] transfer: unknown src\n");
      op->dump();
      op->getParentOfType<DpuProgramOp>()->dump();
      return;
    }

    if (sv_map.count(dstVal)) {
      auto &sv = sv_map.at(dstVal);
      dstId = sv.buf_id;
      dst_iv = sv.iv_indexed;
    } else if (buf_map.count(dstVal)) {
      dstId = buf_map.at(dstVal);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[upmem-cpp-sim] transfer: unknown dst\n");
      return;
    }

    builder.createTransfer(srcId, dstId, n_elems, src_iv, dst_iv);
  }

  void translateConstant(arith::ConstantOp op) {
    upmem_cm::DType dtype = mlirTypeToDtype(op.getType());
    int64_t raw = 0;
    if (auto ia = dyn_cast<IntegerAttr>(op.getValue()))
      raw = ia.getInt();
    else if (auto fa = dyn_cast<FloatAttr>(op.getValue()))
      raw = static_cast<int64_t>(fa.getValueAsDouble());
    val_map[op.getResult()] = builder.createConst(raw, dtype);
  }

  void translateLoad(memref::LoadOp op) {
    auto it = buf_map.find(op.getMemRef());
    if (it == buf_map.end())
      return;
    val_map[op.getResult()] =
        builder.createLoad(it->second, ivIndexedIn(op.getIndices()));
  }

  void translateStore(memref::StoreOp op) {
    auto bufIt = buf_map.find(op.getMemRef());
    if (bufIt == buf_map.end())
      return;
    auto valIt = val_map.find(op.getValue());
    if (valIt == val_map.end())
      return;
    builder.createStore(bufIt->second, valIt->second,
                        ivIndexedIn(op.getIndices()));
  }

  void translateBinArith(Value result, Value lhs, Value rhs, ArithOp op) {
    auto lit = val_map.find(lhs);
    auto rit = val_map.find(rhs);
    if (lit == val_map.end() || rit == val_map.end())
      return;
    val_map[result] = builder.createArith(op, mlirTypeToDtype(result.getType()),
                                          lit->second, rit->second);
  }

  // Detect reduction: for iter_args(%acc = %init) { %s = addf %acc, %compute;
  // yield %s } followed immediately by memref.store %forResult, %accBuf[...].
  // If detected, emits beginLoop + body + the accumulating arith op + endLoop,
  // and maps the loop result so the following store is translated normally.
  bool tryTranslateReduction(scf::ForOp forOp) {
    if (forOp.getNumRegionIterArgs() != 1)
      return false;

    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return false;

    Value yielded = yieldOp.getOperands()[0];
    Operation *addOp = yielded.getDefiningOp();
    if (!addOp || !isa<arith::AddFOp, arith::AddIOp>(addOp))
      return false;

    BlockArgument iterArg = forOp.getRegionIterArgs()[0];
    Value addLhs = addOp->getOperand(0), addRhs = addOp->getOperand(1);
    Value computeVal;
    if (addLhs == iterArg)
      computeVal = addRhs;
    else if (addRhs == iterArg)
      computeVal = addLhs;
    else
      return false;

    Value forResult = forOp.getResult(0);
    if (!forResult.hasOneUse())
      return false;
    auto storeOp = dyn_cast<memref::StoreOp>(*forResult.user_begin());
    if (!storeOp)
      return false;

    if (!buf_map.count(storeOp.getMemRef()))
      return false;

    int64_t lb = getConstInt(forOp.getLowerBound());
    int64_t ub = getConstInt(forOp.getUpperBound());
    int64_t step = getConstInt(forOp.getStep());

    iv_stack.push_back(forOp.getInductionVar());
    builder.beginLoop(lb, ub, step);

    for (Operation &op : *forOp.getBody()) {
      if (&op == addOp || isa<scf::YieldOp>(&op))
        continue;
      translateOp(op);
    }

    // The accumulator is an scf.for iter_arg -- a register, not memory -- so
    // the only per-iteration cost is the arithmetic.
    auto cvIt = val_map.find(computeVal);
    if (cvIt != val_map.end()) {
      auto initIt = val_map.find(forOp.getInitArgs()[0]);
      ProgramBuilder::ValId accVal =
          initIt != val_map.end() ? initIt->second : cvIt->second;
      val_map[forResult] = builder.createArith(
          ArithOp::ADD, mlirTypeToDtype(forResult.getType()), accVal,
          cvIt->second);
    }

    builder.endLoop();
    iv_stack.pop_back();
    return true;
  }

  void translateFor(scf::ForOp forOp) {
    if (tryTranslateReduction(forOp))
      return;

    int64_t lb = getConstInt(forOp.getLowerBound());
    int64_t ub = getConstInt(forOp.getUpperBound());
    int64_t step = getConstInt(forOp.getStep());

    iv_stack.push_back(forOp.getInductionVar());
    builder.beginLoop(lb, ub, step);

    for (Operation &op : *forOp.getBody()) {
      if (isa<scf::YieldOp>(&op))
        continue;
      translateOp(op);
    }

    builder.endLoop();
    iv_stack.pop_back();
  }

  void translateIf(scf::IfOp ifOp) {
    if (auto attr =
            ifOp->getAttrOfType<DenseI8ArrayAttr>("upmem_cm.const_tasklets")) {
      std::vector<int> allowedTasklets;
      allowedTasklets.reserve(attr.size());
      for (auto tid : attr.asArrayRef()) {
        allowedTasklets.push_back(tid);
      }
      builder.beginIfThread(std::move(allowedTasklets));
      for (Operation &op : ifOp.getThenRegion().front()) {
        if (isa<scf::YieldOp>(&op))
          continue;
        translateOp(op);
      }
      builder.endIfThread();
      return;
    }

    // Approximation: translate the then-block as unconditional.
    for (Operation &op : ifOp.getThenRegion().front()) {
      if (isa<scf::YieldOp>(&op))
        continue;
      translateOp(op);
    }
  }

  void translateOp(Operation &op) {
    if (auto o = dyn_cast<memref::AllocaOp>(&op))
      translatePwramAlloc(o);
    else if (auto o = dyn_cast<StaticAllocOp>(&op))
      translateStaticAlloc(o);
    else if (auto o = dyn_cast<TaskletDimOp>(&op))
      translateTaskletDim(o);
    else if (auto o = dyn_cast<memref::SubViewOp>(&op))
      translateSubView(o);
    else if (auto o = dyn_cast<memref::ExpandShapeOp>(&op))
      translateExpandShape(o);
    else if (auto o = dyn_cast<memref::ReinterpretCastOp>(&op))
      translateReinterpretCast(o);
    else if (auto o = dyn_cast<LocalTransferOp>(&op))
      translateLocalTransfer(o);
    else if (auto o = dyn_cast<arith::ConstantOp>(&op))
      translateConstant(o);
    else if (auto o = dyn_cast<memref::LoadOp>(&op))
      translateLoad(o);
    else if (auto o = dyn_cast<memref::StoreOp>(&op))
      translateStore(o);
    else if (auto o = dyn_cast<arith::AddFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::ADD);
    else if (auto o = dyn_cast<arith::AddIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::ADD);
    else if (auto o = dyn_cast<arith::MulFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::MUL);
    else if (auto o = dyn_cast<arith::MulIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::MUL);
    else if (auto o = dyn_cast<arith::SubFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::SUB);
    else if (auto o = dyn_cast<arith::SubIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::SUB);
    else if (auto o = dyn_cast<arith::DivFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::DIV);
    else if (auto o = dyn_cast<arith::DivUIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::DIV);
    else if (auto o = dyn_cast<arith::DivSIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), ArithOp::DIV);
    else if (auto o = dyn_cast<scf::ForOp>(&op))
      translateFor(o);
    else if (auto o = dyn_cast<scf::IfOp>(&op))
      translateIf(o);
    else if (auto o = dyn_cast<upmem::BarrierOp>(&op))
      builder.createBarrier();
    // todo add remui, cmpi
    // others → silently skip
  }

  void translateProgram(DpuProgramOp prog) {
    Region &body = prog.getBody();
    if (body.empty() || body.front().empty())
      return;
    for (Operation &op : body.front())
      translateOp(op);
  }
};

// ===----------------------------------------------------------------------===//
// CppSimulator
// ===----------------------------------------------------------------------===//

enum class SimMode { FAST = 0, CYCLEACCURATE = 1, HYBRID = 2 };

/// Writes `json` to <dir>/<stem>.cnmprog.json, creating the directory if
/// needed. Failures warn rather than fail the pass: a dump is a debugging
/// aid, and losing one should never turn a working compile into a broken one.
void writeProgramDump(llvm::StringRef dir, llvm::StringRef stem,
                      llvm::StringRef json, Operation *diagOp) {
  if (auto ec = llvm::sys::fs::create_directories(dir)) {
    diagOp->emitWarning() << "upmem: could not create program dump directory '"
                          << dir << "': " << ec.message();
    return;
  }
  llvm::SmallString<128> outPath(dir);
  llvm::sys::path::append(outPath, stem + ".cnmprog.json");
  std::error_code ec;
  llvm::raw_fd_ostream out(outPath, ec);
  if (ec) {
    diagOp->emitWarning() << "upmem: could not write program dump '" << outPath
                          << "': " << ec.message();
    return;
  }
  out << json;
}

struct CppSimulator : UpmemSimulator {
  bool annotateOpCosts;
  std::chrono::milliseconds timeoutMs;
  SimMode mode;
  /// Non-empty enables the JSON program dump; see createCycleAccurateSimulator.
  std::string programDumpDir;

  explicit CppSimulator(bool annotateOpCosts,
                        std::chrono::milliseconds timeoutMs, SimMode mode,
                        llvm::StringRef programDumpDir = {})
      : annotateOpCosts(annotateOpCosts), timeoutMs(timeoutMs), mode(mode),
        programDumpDir(programDumpDir.str()) {}

  std::unique_ptr<UpmemSimulator> clone() override {
    return std::make_unique<CppSimulator>(annotateOpCosts, timeoutMs, mode,
                                          programDumpDir);
  }

  bool supportsMultithreading() const override { return true; }

  Maybe<SimCost> simulate(Region &region) override {
    std::chrono::milliseconds tms = timeoutMs;
    // Shared across the copies simulateHostRegion may make of this callback,
    // so unnamed programs still get distinct dump filenames.
    auto dumpCtr = std::make_shared<std::atomic<unsigned>>(0);
    auto waitForCb = [tms, mode = this->mode, dumpDir = this->programDumpDir,
                      dumpCtr](Operation *op, bool) -> SimCost {
      auto waitFor = llvm::cast<WaitForOp>(op);
      DpuProgramOp dpuProg = waitFor.getDpuProgram();
      if (!dpuProg)
        return {};
      int T = dpuProg.getNumTasklets();
      ProgramBuilder builder;
      DpuTranslator tr(builder);
      tr.translateProgram(dpuProg);

      if (!dumpDir.empty()) {
        llvm::StringRef name;
        if (auto sym = dpuProg->getAttrOfType<StringAttr>(
                SymbolTable::getSymbolAttrName()))
          name = sym.getValue();
        std::string stem =
            name.empty() ? ("kernel_" + std::to_string(dumpCtr->fetch_add(1)))
                         : name.str();
        writeProgramDump(dumpDir, stem, builder.emitJson(T, stem), op);
      }

      auto kernelNs = builder.simulate(T, tms, mode == SimMode::FAST);
      if (!kernelNs.has_value() && mode == SimMode::HYBRID) {
        // In hybrid mode we first try to simulate with the cycle accurate
        // simulator, and if we time out we reply with the fast simulator.
        kernelNs = builder.simulate(T, tms, true);
      }
      auto kernelMs =
          1e3 * kernelNs.value_or(std::numeric_limits<double>::infinity());

      auto hierarchy =
          llvm::cast<DeviceHierarchyType>(waitFor.getDpuSet().getType());
      int numDpus = hierarchy.getNumDpus();
      auto launchOverhead = 0;
      // auto launchOverhead = 0.0254524 * numDpus / 64;
      // auto launchOverhead = 0.041958 * log2(numDpus);
      // auto launchOverhead =
      //      -2.347115 - 0.001803433284655423 * numDpus +
      //                       0.3805487552732298 * log2(numDpus);
      return SimCost::forKernel(kernelMs) +
             SimCost::forKernel(launchOverhead, "launchOverhead");
    };
    return simulateHostRegion(region, annotateOpCosts, waitForCb);
  }
};

} // anonymous namespace
// ===----------------------------------------------------------------------===//
// Factory
// ===----------------------------------------------------------------------===//

std::unique_ptr<mlir::upmem::UpmemSimulator>
mlir::upmem::createCycleAccurateSimulator(bool annotateOpCosts,
                                          std::chrono::milliseconds timeoutMs,
                                          llvm::StringRef programDumpDir) {
  return std::make_unique<CppSimulator>(annotateOpCosts, timeoutMs,
                                        SimMode::CYCLEACCURATE, programDumpDir);
}
std::unique_ptr<mlir::upmem::UpmemSimulator>
mlir::upmem::createFastSimulator(bool annotateOpCosts,
                                 std::chrono::milliseconds timeoutMs,
                                 llvm::StringRef programDumpDir) {
  return std::make_unique<CppSimulator>(annotateOpCosts, timeoutMs,
                                        SimMode::FAST, programDumpDir);
}
std::unique_ptr<mlir::upmem::UpmemSimulator>
mlir::upmem::createHybridSimulator(bool annotateOpCosts,
                                   std::chrono::milliseconds timeoutMs,
                                   llvm::StringRef programDumpDir) {
  return std::make_unique<CppSimulator>(annotateOpCosts, timeoutMs,
                                        SimMode::HYBRID, programDumpDir);
}

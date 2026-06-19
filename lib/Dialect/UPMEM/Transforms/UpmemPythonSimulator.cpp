#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"
#include "upmem_cost_model/Types.h"

#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <upmem_cost_model/ProgramBuilder.h>

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::upmem {

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
  llvm::DenseSet<Value> skip_vals;
  int buf_ctr = 0;

  explicit DpuTranslator(ProgramBuilder &b) : builder(b) {}

  // ── Type helpers ──────────────────────────────────────────────────────────

  DType mlirTypeToDtype(Type ty) {
    if (ty.isF32())
      return DType::F32;
    if (ty.isF64())
      return DType::F64;
    // F16/BF16 → F32 (UPMEM cost model has no 16-bit float type)
    if (ty.isF16() || ty.isBF16())
      return DType::F32;
    if (auto it = dyn_cast<IntegerType>(ty)) {
      unsigned w = it.getWidth();
      bool s = !it.isUnsigned();
      if (w == 8)
        return s ? DType::I8 : DType::U8;
      if (w == 16)
        return s ? DType::I16 : DType::U16;
      if (w == 32)
        return s ? DType::I32 : DType::U32;
      if (w == 64)
        return s ? DType::I64 : DType::U64;
    }
    return DType::I64; // index or unknown
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

  void translatePwramAlloc(PrivateWRAMAllocOp op) {
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
    val_map[op.getResult()] = builder.createConst(0, DType::I64);
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
    DType dtype = mlirTypeToDtype(op.getType());
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
    if (skip_vals.count(op.getValue()))
      return;
    if (op.getValue().getDefiningOp() &&
        isa<scf::ForOp>(op.getValue().getDefiningOp()))
      return;
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
  // If detected, emits beginLoop + body + createReduceStore + endLoop and marks
  // the forResult so the following store is skipped.
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

    auto accBufIt = buf_map.find(storeOp.getMemRef());
    if (accBufIt == buf_map.end())
      return false;
    ProgramBuilder::BufId accBufId = accBufIt->second;

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

    auto cvIt = val_map.find(computeVal);
    if (cvIt != val_map.end())
      builder.createReduceStore(accBufId, ArithOp::ADD, cvIt->second);

    builder.endLoop();
    iv_stack.pop_back();

    skip_vals.insert(forResult);
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
    // Approximation: translate the then-block as unconditional.
    for (Operation &op : ifOp.getThenRegion().front()) {
      if (isa<scf::YieldOp>(&op))
        continue;
      translateOp(op);
    }
  }

  void translateOp(Operation &op) {
    if (auto o = dyn_cast<PrivateWRAMAllocOp>(&op))
      translatePwramAlloc(o);
    else if (auto o = dyn_cast<StaticAllocOp>(&op))
      translateStaticAlloc(o);
    else if (auto o = dyn_cast<TaskletDimOp>(&op))
      translateTaskletDim(o);
    else if (auto o = dyn_cast<memref::SubViewOp>(&op))
      translateSubView(o);
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
    // BarrierOp, ReturnOp, arith.cmpi, etc. → silently skip
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

// ── simulateGemv cache ───────────────────────────────────────────────────────
// All CppSimulator instances share one process-wide cache (the function is
// pure). Two lookup strategies depending on problem size:
//   lookupBlocking — spin until lock acquired (expensive configs, worth it)
//   lookupTry      — try once, skip on contention (cheap configs)
struct GemvCache {
  using Key = std::array<int64_t, 7>;
  struct KeyHash {
    size_t operator()(const Key &k) const noexcept {
      size_t h = 0;
      for (int64_t v : k)
        h ^= std::hash<int64_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  std::unordered_map<Key, double, KeyHash> map;
  std::atomic_flag lock; // zero-initialized for static-storage instances
  std::atomic<uint64_t> hits{0};
  std::atomic<uint64_t> misses{0};

  // Spin until the lock is available, then look up key.
  // Returns the cached value, or nullopt on a confirmed cache miss.
  std::optional<double> lookupBlocking(const Key &key) {
    while (lock.test_and_set(std::memory_order_acquire))
      ; // short critical section — spin is cheaper than blocking
    auto it = map.find(key);
    if (it != map.end()) {
      double v = it->second;
      lock.clear(std::memory_order_release);
      hits.fetch_add(1, std::memory_order_relaxed);
      return v;
    }
    lock.clear(std::memory_order_release);
    misses.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }

  // Try to acquire the lock once; return cached value if found.
  // Returns nullopt on cache miss OR on lock contention (best-effort).
  std::optional<double> lookupTry(const Key &key) {
    if (!lock.test_and_set(std::memory_order_acquire)) {
      auto it = map.find(key);
      if (it != map.end()) {
        double v = it->second;
        lock.clear(std::memory_order_release);
        hits.fetch_add(1, std::memory_order_relaxed);
        return v;
      }
      lock.clear(std::memory_order_release);
    }
    misses.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }

  void store(const Key &key, double value) {
    if (!lock.test_and_set(std::memory_order_acquire)) {
      map.emplace(key, value);
      lock.clear(std::memory_order_release);
    }
  }

  void printStats(llvm::raw_ostream &os) const {
    auto h = hits.load(std::memory_order_relaxed);
    auto total = h + misses.load(std::memory_order_relaxed);
    os << "[upmem-cpp-sim] simulateGemv cache: " << h << " hits / " << total
       << " lookups";
    if (total > 0)
      os << " (" << (100 * h / total) << "%)";
    os << ", " << map.size() << " unique configs cached\n";
  }
};
static GemvCache gemvCache;

struct CppSimulator : UpmemSimulator {
  bool annotateOpCosts;
  std::chrono::milliseconds timeoutMs;

  explicit CppSimulator(bool annotateOpCosts,
                        std::chrono::milliseconds timeoutMs)
      : annotateOpCosts(annotateOpCosts), timeoutMs(timeoutMs) {}

  std::unique_ptr<UpmemSimulator> clone() override {
    return std::make_unique<CppSimulator>(annotateOpCosts, timeoutMs);
  }

  bool supportsMultithreading() const override { return true; }

  void printStats() const override {
    LLVM_DEBUG(gemvCache.printStats(llvm::dbgs()));
  }

  double simulateGemv(std::chrono::milliseconds timeout, int nTasklets,
                      int64_t mramRows, int64_t mramCols, int64_t rowTile,
                      int64_t colTile, upmem_cm::DType dty) override;

  Maybe<double> simulate(Region &region) override {
    std::chrono::milliseconds tms = timeoutMs;
    auto waitForCb = [tms](Operation *op, bool) -> double {
      auto waitFor = llvm::cast<WaitForOp>(op);
      DpuProgramOp dpuProg = waitFor.getDpuProgram();
      if (!dpuProg)
        return 1.0;
      int T = dpuProg.getNumTasklets();
      ProgramBuilder builder;
      DpuTranslator tr(builder);
      tr.translateProgram(dpuProg);
      return builder.simulate(T, tms).value_or(
          std::numeric_limits<double>::infinity());
    };
    return simulateHostRegion(region, annotateOpCosts, waitForCb);
  }
};

} // anonymous namespace

double mlir::upmem::CppSimulator::simulateGemv(
    std::chrono::milliseconds timeout, int nTasklets, int64_t mramRows,
    int64_t mramCols, int64_t rowTile, int64_t colTile, upmem_cm::DType dty) {
  const GemvCache::Key key{
      timeout.count(),          nTasklets, mramRows, mramCols, rowTile, colTile,
      static_cast<int64_t>(dty)};

  // Large (expensive) configs: spin-wait for a definitive cache check.
  // Small (cheap) configs: try once, fall through on contention.
  const bool expensive = (mramRows * mramCols >= 512);
  if (auto cached =
          expensive ? gemvCache.lookupBlocking(key) : gemvCache.lookupTry(key))
    return *cached;

  using namespace upmem_cm;
  ProgramBuilder b;

  // MRAM buffers
  auto A_mram = b.addBuffer("A_mram", MemSpace::MRAM, dty);
  auto x_mram = b.addBuffer("x_mram", MemSpace::MRAM, dty);
  auto y_mram = b.addBuffer("y_mram", MemSpace::MRAM, dty);

  // WRAM tile buffers
  auto A_wram = b.addBuffer("A_wram", MemSpace::WRAM, dty);
  auto x_wram = b.addBuffer("x_wram", MemSpace::WRAM, dty);
  auto y_wram = b.addBuffer("y_wram", MemSpace::WRAM, dty);

  // Load all y from MRAM to WRAM before loops
  b.createTransfer(y_mram, y_wram, mramRows); // todo mark exclusive

  // note that there is nTasklet threads doing this at the same time.
  int64_t nRowTiles = mramRows / rowTile / nTasklets;
  int64_t nColTiles = mramCols / colTile;

  // int64_t rowTilesMin = std::min(nRowTiles, 4L);
  // int64_t colTilesMin = std::min(nColTiles, 4L);

  b.beginLoop(0, nRowTiles); // row tile loop
  b.beginLoop(0, nColTiles); // col tile loop

  // Transfer A tile [rowTile × colTile] from MRAM; address advances per
  // col-tile iter
  b.createTransfer(A_mram, A_wram, rowTile * colTile, /*src_iv_indexed=*/true);
  // Transfer x tile [colTile] from MRAM; address advances per col-tile iter
  // (in the kernel only tasklet 0 does this via scf.if; modeled
  // unconditionally)
  b.createTransfer(x_mram, x_wram, colTile, /*src_iv_indexed=*/true);

  b.beginLoop(0, rowTile); // row loop within tile
  b.beginLoop(0, colTile); // dot-product loop

  // acc += A_wram[row, col] * x_wram[col]; both stride by 1 per inner iteration
  auto a_val = b.createLoad(A_wram, /*iv_indexed=*/true);
  auto x_val = b.createLoad(x_wram, /*iv_indexed=*/true);
  auto prod = b.createArith(ArithOp::MUL, dty, a_val, x_val);
  // load-add-store into y_wram (models the iter_args reduction pattern)
  b.createReduceStore(y_wram, ArithOp::ADD, prod);

  b.endLoop(); // dot-product loop
  b.endLoop(); // row loop within tile
  b.endLoop(); // col tile loop
  b.endLoop(); // row tile loop

  // Store all y from WRAM back to MRAM
  // todo mark exclusive
  b.createTransfer(y_wram, y_mram, mramRows);

  double result = b.simulate(nTasklets, timeout)
                      .value_or(std::numeric_limits<double>::infinity());

  gemvCache.store(key, result);
  return result;
}

// ===----------------------------------------------------------------------===//
// Factory
// ===----------------------------------------------------------------------===//

std::unique_ptr<UpmemSimulator>
createPythonSimulator(bool annotateOpCosts,
                      std::chrono::milliseconds timeoutMs) {
  return std::make_unique<CppSimulator>(annotateOpCosts, timeoutMs);
}

} // namespace mlir::upmem

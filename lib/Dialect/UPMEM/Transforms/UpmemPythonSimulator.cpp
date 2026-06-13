#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include <llvm/Support/Debug.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/WalkResult.h>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#define DEBUG_TYPE "upmem-python-sim"

#ifndef UPMEM_SIM_PACKAGE_DIR
#error Macro UPMEM_SIM_PACKAGE_DIR should be set to the pythonpath of the upmem python simulator
#endif

namespace py = pybind11;

namespace mlir::upmem {

using mlir::cinm::utils::Maybe;

// ===----------------------------------------------------------------------===//
// Python interpreter singleton
// ===----------------------------------------------------------------------===//

static py::scoped_interpreter *g_interp = nullptr;

static void ensurePythonInitialized() {
  if (!Py_IsInitialized()) {
    g_interp = new py::scoped_interpreter{};
    py::module_::import("sys").attr("path").attr("insert")(
        0, UPMEM_SIM_PACKAGE_DIR);
  }
}

// ===----------------------------------------------------------------------===//
// Python class handles (lazily cached)
// ===----------------------------------------------------------------------===//

struct PyClasses {
  py::object Buffer, Var, VarRef, Const, BinExpr, Load, Store, For, Transfer,
      Program, BinOp;
  py::object MemSpace;
  py::object DataType;
  py::object lower_program;
  py::object Simulator;
};

static PyClasses *g_cls = nullptr;

static PyClasses &getPyClasses() {
  if (!g_cls) {
    auto hl = py::module_::import("upmem_simulator.highlevel_ir");
    auto ll = py::module_::import("upmem_simulator.lowlevel_ir");
    auto ut = py::module_::import("upmem_simulator.ir_utils");
    auto sim = py::module_::import("upmem_simulator");
    g_cls = new PyClasses{
        hl.attr("Buffer"),     hl.attr("Var"),
        hl.attr("VarRef"),     hl.attr("Const"),
        hl.attr("BinExpr"),    hl.attr("Load"),
        hl.attr("Store"),      hl.attr("For"),
        hl.attr("Transfer"),   hl.attr("Program"),
        hl.attr("BinOp"),      ll.attr("MemSpace"),
        ut.attr("DataType"),   sim.attr("lower_program"),
        sim.attr("Simulator"),
    };
  }
  return *g_cls;
}

// ===----------------------------------------------------------------------===//
// DpuTranslator — walks a DpuProgramOp body and builds Python highlevel_ir
// ===----------------------------------------------------------------------===//

namespace {

struct SubviewInfo {
  py::object buffer;
  py::list base;
  int64_t n_elems;
};

struct DpuTranslator {
  const PyClasses &cls;
  llvm::DenseMap<Value, py::object> val_map; // Value → Python expr or Buffer
  llvm::DenseMap<Value, py::object> var_map; // Value → Python Var (for indices)
  llvm::DenseMap<Value, SubviewInfo> sv_map; // subview result → SubviewInfo
  std::vector<py::object> buffers;
  llvm::DenseSet<Value> skip_vals; // for results handled by reduction
  int buf_ctr = 0;
  int var_ctr = 0;

  explicit DpuTranslator(const PyClasses &cls) : cls(cls) {}

  // ── Type helpers ─────────────────────────────────────────────────────────

  py::object mlirTypeToDtype(Type ty) {
    if (ty.isF32())
      return cls.DataType.attr("FP32");
    if (ty.isF64())
      return cls.DataType.attr("FP64");
    if (ty.isF16())
      return cls.DataType.attr("FP16");
    if (ty.isBF16())
      return cls.DataType.attr("BF16");
    if (auto it = dyn_cast<IntegerType>(ty)) {
      unsigned w = it.getWidth();
      bool s = !it.isUnsigned();
      if (w == 8)
        return s ? cls.DataType.attr("S8") : cls.DataType.attr("U8");
      if (w == 16)
        return s ? cls.DataType.attr("S16") : cls.DataType.attr("U16");
      if (w == 32)
        return s ? cls.DataType.attr("S32") : cls.DataType.attr("U32");
      if (w == 64)
        return s ? cls.DataType.attr("S64") : cls.DataType.attr("U64");
    }
    if (ty.isIndex())
      return cls.DataType.attr("S64");
    return cls.DataType.attr("S32");
  }

  py::object memSpaceOf(MemRefType mrt) {
    if (auto dma = dyn_cast_or_null<DpuMemSpaceAttr>(mrt.getMemorySpace()))
      if (dma.getValue() == DpuMemSpace::MRAM)
        return cls.MemSpace.attr("L2");
    return cls.MemSpace.attr("L1");
  }

  // ── Value access helpers ──────────────────────────────────────────────────

  // Return an index (Python int or Var) for a value used as an array subscript.
  py::object getIndex(Value v) {
    if (var_map.count(v))
      return var_map.at(v);
    if (auto *def = v.getDefiningOp())
      if (auto c = dyn_cast<arith::ConstantOp>(def))
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
          return py::int_((long long)ia.getInt());
    LLVM_DEBUG(llvm::dbgs() << "[upmem-python-sim] unknown index → 0\n");
    return py::int_(0);
  }

  // Return a Python expression (Const / VarRef / Load / BinExpr) for a Value.
  py::object getExpr(Value v) {
    if (val_map.count(v))
      return val_map.at(v);
    if (var_map.count(v))
      return cls.VarRef(var_map.at(v));
    LLVM_DEBUG(llvm::dbgs() << "[upmem-python-sim] unknown expr → Const(0)\n");
    return cls.Const(py::int_(0), cls.DataType.attr("S32"));
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
    py::list shape;
    for (int64_t d : mrt.getShape())
      shape.append(py::int_(d > 0 ? d : 1));
    std::string name = "wram_" + std::to_string(buf_ctr++);
    py::object buf = cls.Buffer(name, shape, cls.MemSpace.attr("L1"),
                                mlirTypeToDtype(mrt.getElementType()));
    val_map[op.getResult()] = buf;
    buffers.push_back(buf);
  }

  void translateStaticAlloc(StaticAllocOp op) {
    auto mrt = llvm::cast<MemRefType>(op.getBuffer().getType());
    py::list shape;
    for (int64_t d : mrt.getShape())
      shape.append(py::int_(d > 0 ? d : 1));
    std::string name = op.getSymName() ? op.getSymName()->str()
                                       : ("buf_" + std::to_string(buf_ctr++));
    py::object buf = cls.Buffer(name, shape, memSpaceOf(mrt),
                                mlirTypeToDtype(mrt.getElementType()));
    val_map[op.getBuffer()] = buf;
    buffers.push_back(buf);
  }

  void translateTaskletDim(TaskletDimOp op) {
    py::object tidVar = cls.Var("tid", cls.DataType.attr("S64"));
    var_map[op.getResult()] = tidVar;
    val_map[op.getResult()] = cls.VarRef(tidVar);
  }

  void translateSubView(memref::SubViewOp op) {
    auto srcIt = val_map.find(op.getSource());
    if (srcIt == val_map.end())
      return;

    py::list base;
    for (OpFoldResult off : op.getMixedOffsets()) {
      if (auto cv = getConstantIntValue(off))
        base.append(py::int_(*cv));
      else if (auto v = off.dyn_cast<Value>())
        base.append(getIndex(v));
      else
        base.append(py::int_(0));
    }

    int64_t n = 1;
    for (OpFoldResult sz : op.getMixedSizes())
      if (auto cv = getConstantIntValue(sz))
        n *= *cv;

    sv_map[op.getResult()] = {srcIt->second, base, n};
  }

  void translateLocalTransfer(LocalTransferOp op,
                              std::vector<py::object> &stmts) {
    Value srcVal = op.getSource();
    Value dstVal = op.getTarget();
    py::object srcBuf, dstBuf;
    py::list srcBase, dstBase;
    int64_t n_elems = 1;

    if (sv_map.count(srcVal)) {
      auto &sv = sv_map.at(srcVal);
      srcBuf = sv.buffer;
      srcBase = sv.base;
      n_elems = sv.n_elems;
    } else if (val_map.count(srcVal)) {
      srcBuf = val_map.at(srcVal);
      if (auto mrt = dyn_cast<MemRefType>(srcVal.getType()))
        if (mrt.hasStaticShape())
          n_elems = mrt.getNumElements();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[upmem-python-sim] transfer: unknown src\n");
      return;
    }

    if (sv_map.count(dstVal)) {
      auto &sv = sv_map.at(dstVal);
      dstBuf = sv.buffer;
      dstBase = sv.base;
    } else if (val_map.count(dstVal)) {
      dstBuf = val_map.at(dstVal);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[upmem-python-sim] transfer: unknown dst\n");
      return;
    }

    stmts.push_back(
        cls.Transfer(srcBuf, dstBuf, py::int_(n_elems), srcBase, dstBase));
  }

  void translateConstant(arith::ConstantOp op) {
    py::object dtype = mlirTypeToDtype(op.getType());
    py::object cval;
    if (auto ia = dyn_cast<IntegerAttr>(op.getValue()))
      cval = cls.Const(py::int_((long long)ia.getInt()), dtype);
    else if (auto fa = dyn_cast<FloatAttr>(op.getValue()))
      cval = cls.Const(py::float_(fa.getValueAsDouble()), dtype);
    else
      cval = cls.Const(py::int_(0), dtype);
    val_map[op.getResult()] = cval;
  }

  void translateLoad(memref::LoadOp op) {
    auto it = val_map.find(op.getMemRef());
    if (it == val_map.end())
      return;
    py::list indices;
    for (Value idx : op.getIndices())
      indices.append(getIndex(idx));
    val_map[op.getResult()] = cls.Load(it->second, indices);
  }

  void translateStore(memref::StoreOp op, std::vector<py::object> &stmts) {
    if (skip_vals.count(op.getValue()))
      return;
    if (op.getValue().getDefiningOp() &&
        isa<scf::ForOp>(op.getValue().getDefiningOp()))
      return;
    auto it = val_map.find(op.getMemRef());
    if (it == val_map.end())
      return;
    py::list indices;
    for (Value idx : op.getIndices())
      indices.append(getIndex(idx));
    stmts.push_back(
        cls.Store(it->second, indices, getExpr(op.getValue()), py::none()));
  }

  void translateBinArith(Value result, Value lhs, Value rhs, const char *op) {
    val_map[result] =
        cls.BinExpr(cls.BinOp.attr(op), getExpr(lhs), getExpr(rhs),
                    mlirTypeToDtype(result.getType()));
  }

  // Detect the canonical reduction pattern and emit For+Store(reduce_op=ADD).
  bool tryTranslateReduction(scf::ForOp forOp, std::vector<py::object> &stmts) {
    if (forOp.getNumRegionIterArgs() != 1)
      return false;

    auto *term = forOp.getBody()->getTerminator();
    auto yieldOp = dyn_cast<scf::YieldOp>(term);
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

    auto accBufIt = val_map.find(storeOp.getMemRef());
    if (accBufIt == val_map.end())
      return false;

    // Register the IV.
    std::string ivName = "iv_" + std::to_string(var_ctr++);
    py::object ivVar = cls.Var(ivName, cls.DataType.attr("S64"));
    var_map[forOp.getInductionVar()] = ivVar;
    val_map[forOp.getInductionVar()] = cls.VarRef(ivVar);

    // Translate body ops (except the add and yield) to populate val_map,
    // collecting any side-effect stmts (stores, transfers) into stmts.
    for (Operation &op : *forOp.getBody()) {
      if (&op == addOp || isa<scf::YieldOp>(&op))
        continue;
      translateOp(op, stmts);
    }

    py::list accIdx;
    for (Value idx : storeOp.getIndices())
      accIdx.append(getIndex(idx));

    py::list forBody;
    forBody.append(cls.Store(accBufIt->second, accIdx, getExpr(computeVal),
                             cls.BinOp.attr("ADD")));

    int64_t lb = getConstInt(forOp.getLowerBound());
    int64_t ub = getConstInt(forOp.getUpperBound());
    int64_t step = getConstInt(forOp.getStep());
    stmts.push_back(
        cls.For(ivVar, py::int_(lb), py::int_(ub), py::int_(step), forBody));

    skip_vals.insert(forResult);
    return true;
  }

  void translateFor(scf::ForOp forOp, std::vector<py::object> &stmts) {
    if (tryTranslateReduction(forOp, stmts))
      return;

    std::string ivName = "iv_" + std::to_string(var_ctr++);
    py::object ivVar = cls.Var(ivName, cls.DataType.attr("S64"));
    var_map[forOp.getInductionVar()] = ivVar;
    val_map[forOp.getInductionVar()] = cls.VarRef(ivVar);

    int64_t lb = getConstInt(forOp.getLowerBound());
    int64_t ub = getConstInt(forOp.getUpperBound());
    int64_t step = getConstInt(forOp.getStep());

    std::vector<py::object> body_stmts;
    for (Operation &op : *forOp.getBody()) {
      if (isa<scf::YieldOp>(&op))
        continue;
      translateOp(op, body_stmts);
    }

    py::list body_list;
    for (auto &s : body_stmts)
      body_list.append(s);
    stmts.push_back(
        cls.For(ivVar, py::int_(lb), py::int_(ub), py::int_(step), body_list));
  }

  void translateIf(scf::IfOp ifOp, std::vector<py::object> &stmts) {
    // Approximation: translate the then-block as unconditional.
    // This overestimates cost for tasklet-0-only patterns (e.g., broadcast).
    for (Operation &op : ifOp.getThenRegion().front()) {
      if (isa<scf::YieldOp>(&op))
        continue;
      translateOp(op, stmts);
    }
  }

  void translateOp(Operation &op, std::vector<py::object> &stmts) {
    if (auto o = dyn_cast<PrivateWRAMAllocOp>(&op))
      translatePwramAlloc(o);
    else if (auto o = dyn_cast<StaticAllocOp>(&op))
      translateStaticAlloc(o);
    else if (auto o = dyn_cast<TaskletDimOp>(&op))
      translateTaskletDim(o);
    else if (auto o = dyn_cast<memref::SubViewOp>(&op))
      translateSubView(o);
    else if (auto o = dyn_cast<LocalTransferOp>(&op))
      translateLocalTransfer(o, stmts);
    else if (auto o = dyn_cast<arith::ConstantOp>(&op))
      translateConstant(o);
    else if (auto o = dyn_cast<memref::LoadOp>(&op))
      translateLoad(o);
    else if (auto o = dyn_cast<memref::StoreOp>(&op))
      translateStore(o, stmts);
    else if (auto o = dyn_cast<arith::AddFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "ADD");
    else if (auto o = dyn_cast<arith::AddIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "ADD");
    else if (auto o = dyn_cast<arith::MulFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "MUL");
    else if (auto o = dyn_cast<arith::MulIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "MUL");
    else if (auto o = dyn_cast<arith::SubFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "SUB");
    else if (auto o = dyn_cast<arith::SubIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "SUB");
    else if (auto o = dyn_cast<arith::DivFOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "DIV");
    else if (auto o = dyn_cast<arith::DivUIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "DIV");
    else if (auto o = dyn_cast<arith::DivSIOp>(&op))
      translateBinArith(o.getResult(), o.getLhs(), o.getRhs(), "DIV");
    else if (auto o = dyn_cast<scf::ForOp>(&op))
      translateFor(o, stmts);
    else if (auto o = dyn_cast<scf::IfOp>(&op))
      translateIf(o, stmts);
    // BarrierOp, ReturnOp, arith.cmpi, etc. → silently skip
  }

  py::object translateProgram(DpuProgramOp prog) {
    Region &body = prog.getBody();
    if (body.empty() || body.front().empty())
      return py::none();

    std::vector<py::object> stmts;
    for (Operation &op : body.front())
      translateOp(op, stmts);

    py::object dtype =
        buffers.empty() ? cls.DataType.attr("S32") : buffers[0].attr("dtype");
    py::list buf_list, body_list;
    for (auto &b : buffers)
      buf_list.append(b);
    for (auto &s : stmts)
      body_list.append(s);
    return cls.Program(buf_list, body_list, dtype);
  }
};

// ===----------------------------------------------------------------------===//
// PythonSimulator
// ===----------------------------------------------------------------------===//

struct PythonSimulator : UpmemSimulator {
  Maybe<double> simulate(Region &region) override {
    try {
      ensurePythonInitialized();
      PyClasses &cls = getPyClasses();

      double total = 0.0;
      bool found = false;

      region.walk([&](WaitForOp waitFor) {
        DpuProgramOp dpuProg = waitFor.getDpuProgram();
        if (!dpuProg)
          return WalkResult::skip();

        int T = dpuProg.getNumTasklets();
        DpuTranslator tr(cls);
        py::object program = tr.translateProgram(dpuProg);
        if (program.is_none())
          return WalkResult::skip();

        py::object kernel = cls.lower_program(program);
        py::object sim = cls.Simulator(py::int_(T), kernel);
        py::tuple result = sim.attr("start")().cast<py::tuple>();
        total += result[0].cast<double>();
        found = true;
        return WalkResult::skip();
      });

      if (!found) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[upmem-python-sim] no DPU program found; returning 0\n");
        return 0.0;
      }
      return total;

    } catch (py::error_already_set &e) {
      LLVM_DEBUG(llvm::dbgs() << "[upmem-python-sim] Python error: " << e.what()
                              << "\n  falling back to op-count simulator\n");
      return createOpCountSimulator()->simulate(region);
    }
  }
};

} // anonymous namespace

// ===----------------------------------------------------------------------===//
// Factory
// ===----------------------------------------------------------------------===//

std::unique_ptr<UpmemSimulator> createPythonSimulator() {
  return std::make_unique<PythonSimulator>();
}

} // namespace mlir::upmem

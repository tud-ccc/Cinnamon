//===- TranslateToCpp.cpp - Translating to C++ calls ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOccupancy.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Target/UPMEMCpp/UPMEMCppEmitter.h"
#include "cinm-mlir/Utils/CinmUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstddef>
#include <numeric>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <ranges>
#include <string>
#include <utility>

#define DEBUG_TYPE "translate-to-upmem-cpp"

using namespace mlir;
using namespace mlir::upmem_emitc;

using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
  explicit CppEmitter(raw_ostream &os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(OpResult result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  LogicalResult emitMemVariableDeclaration(OpResult result,
                                           bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  void appendNameOrInt(OpFoldResult value, std::string &expr) {
    if (auto val = llvm::dyn_cast_or_null<Value>(value)) {
      expr.append(getOrCreateName(val));
    } else {
      auto attr = llvm::dyn_cast<IntegerAttr>(llvm::cast<Attribute>(value));
      expr.append(std::to_string(attr.getValue().getSExtValue()));
    }
  }

  LogicalResult recordStaticName(Value val, StringRef name);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(CppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    CppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};
} // namespace

static LogicalResult printConstantOp(CppEmitter &emitter, Operation *operation,
                                     Attribute value) {
  OpResult result = operation->getResult(0);

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Skip the assignment if the emitc.constant has no value.

    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printValueOrConstant(CppEmitter &emitter, Value value) {
  Operation *op = value.getDefiningOp();
  if (arith::ConstantOp constant = dyn_cast_or_null<arith::ConstantOp>(op)) {
    if (emitter.emitAttribute(constant.getLoc(), constant.getValueAttr())
            .failed()) {
      return failure();
    }
  } else {
    emitter.ostream() << emitter.getOrCreateName(value);
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::TaskletDimOp idOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*idOp)))
    return failure();
  os << "me()";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    memref::AllocaOp wramAllocOp) {
  raw_ostream &os = emitter.ostream();
  MemRefType res_type = wramAllocOp.getType();
  Type elementType = res_type.getElementType();

  os << "__dma_aligned ";
  if (emitter.emitType(wramAllocOp.getLoc(), elementType).failed()) {
    return failure();
  }

  size_t size = res_type.getNumElements();
  size = llvm::alignTo(size, 8);
  os << " " << emitter.getOrCreateName(wramAllocOp.getResult()) << "[" << size
     << "]";

  return success();
}

/// Emits the `[...]` subscript for a memref access as a single linearized
/// element index, `i0*s0 + i1*s1 + ...` over the memref's own strides.
///
/// Every DPU buffer is declared as a flat array (see the memref::AllocaOp and
/// StaticAllocOp emitters), so a rank-N access has to be flattened.
///
/// An access whose layout has no static strides is an error rather
/// than a guess -- there is no correct flat subscript to emit for it.
static LogicalResult printLinearSubscript(CppEmitter &emitter, Operation *op,
                                          MemRefType type,
                                          Operation::operand_range indices) {
  raw_ostream &os = emitter.ostream();
  os << "[";

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (!indices.empty()) {
    if (failed(type.getStridesAndOffset(strides, offset)))
      return op->emitError(
          "cannot emit C for this access: memref layout is not strided");
    if (ShapedType::isDynamic(offset) ||
        llvm::any_of(strides, ShapedType::isDynamic))
      return op->emitError("cannot emit C for this access: memref has dynamic "
                           "strides or offset");
  }

  bool empty = true;
  auto separate = [&] {
    if (!empty)
      os << " + ";
    empty = false;
  };

  if (offset != 0) {
    separate();
    os << offset;
  }

  for (auto [index, stride] : llvm::zip_equal(indices, strides)) {
    if (stride == 0)
      continue;
    // A constant-zero index contributes nothing whatever the stride.
    if (auto cst = index.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
        if (intAttr.getInt() == 0)
          continue;
        separate();
        os << intAttr.getInt() * stride;
        continue;
      }
    separate();
    if (stride != 1)
      os << "(";
    if (failed(printValueOrConstant(emitter, index)))
      return failure();
    if (stride != 1)
      os << " * " << stride << ")";
  }

  if (empty)
    os << "0";
  os << "]";
  return success();
}

static bool isInMemspace(MemRefType ty, upmem::DpuMemSpace space);

/// Direct element access only exists in WRAM. An MRAM buffer lives in a
/// separate address space the core cannot dereference -- its bytes move
/// through mram_read/mram_write -- so a load or store on one is a lowering
/// bug, and emitting `buf[i]` for it would compile to a wrong-address read.
static LogicalResult verifyDirectAccess(Operation *op, MemRefType type) {
  if (isInMemspace(type, upmem::DpuMemSpace::MRAM))
    return op->emitError(
        "cannot emit C for a direct element access to MRAM: only "
        "upmem.local_transfer moves MRAM bytes");
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    memref::LoadOp loadOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(verifyDirectAccess(loadOp, loadOp.getMemRefType())))
    return failure();
  if (failed(emitter.emitAssignPrefix(*loadOp)))
    return failure();

  os << emitter.getOrCreateName(loadOp.getMemRef());
  return printLinearSubscript(emitter, loadOp, loadOp.getMemRefType(),
                              loadOp.getIndices());
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    memref::StoreOp storeOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(verifyDirectAccess(storeOp, storeOp.getMemRefType())))
    return failure();
  os << emitter.getOrCreateName(storeOp.getMemRef());
  if (failed(printLinearSubscript(emitter, storeOp, storeOp.getMemRefType(),
                                  storeOp.getIndices())))
    return failure();
  os << " = ";

  return printValueOrConstant(emitter, storeOp.getValueToStore());
}

static LogicalResult
printMRAMCopyBytes(CppEmitter &emitter, upmem::DpuMemSpace fromSpace,
                   Value from, Value to, size_t staticSizeBytes,
                   const std::string &fromOffsetExpr,
                   const std::string &toOffsetExpr, size_t offsetBytes) {
  raw_ostream &os = emitter.ostream();
  if (fromSpace == mlir::upmem::DpuMemSpace::MRAM) {
    os << "mram_read(&" << emitter.getOrCreateName(from);
  } else {
    os << "mram_write(&((const char*) " << emitter.getOrCreateName(from) << ")";
  }

  os << "[" << fromOffsetExpr << " + " << offsetBytes << "], ";

  if (fromSpace == mlir::upmem::DpuMemSpace::MRAM) {
    os << "&((char*) " << emitter.getOrCreateName(to) << ")";
  } else {
    os << "&" << emitter.getOrCreateName(to);
  }

  os << "[" << toOffsetExpr << " + " << offsetBytes << "], ";

  // todo dyn size
  os << staticSizeBytes;

  os << ")";
  return success();
}

static bool isInMemspace(MemRefType ty, upmem::DpuMemSpace space) {
  if (auto attr =
          llvm::dyn_cast_or_null<upmem::DpuMemSpaceAttr>(ty.getMemorySpace())) {
    return attr.getValue() == space;
  }
  return false;
}

// Peel through ignorable reshape-like ops to reach the underlying value.
static Value skipIgnorableOps(Value v) {
  while (Operation *op = v.getDefiningOp())
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp, memref::ReshapeOp,
            memref::ReinterpretCastOp, memref::CastOp>(op))
      v = op->getOperand(0);
    else
      break;
  return v;
}

static LogicalResult getBasePtrOfAlloc(Operation *op, Value &basePtr) {
  if (!op)
    return emitError(UnknownLoc(), "unknown error during translation");
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<upmem::StaticAllocOp>([&](auto op) {
        basePtr = op.getBuffer();
        return success();
      })
      .Case<memref::AllocaOp>([&](auto op) {
        basePtr = op->getResult(0);
        return success();
      })
      .Default([&](auto op) {
        return op->emitOpError("Expected upmem allocation op");
      });
}

/// Computes the base pointer and byte offset of a DMA endpoint.
///
/// `offsetAlign` receives the largest power of two the emitted offset is
/// provably a multiple of. A term whose index is a constant contributes its
/// own value; a term whose index is only known at run time contributes its
/// multiplier times whatever knownMultipleOf() can establish about the index.
/// The DMA engine requires both endpoints to be 8-byte aligned, and
/// checkDmaAlignment() is what turns a weaker guarantee than that into an
/// error.
/// What an index is provably a multiple of, or 1 when nothing can be said.
///
/// A strided loop is why this is worth doing rather than falling back on the
/// multiplier alone. --upmem-coalesce-local-transfers moves several tiles per
/// transfer and addresses them at `strip * k`, which is aligned exactly
/// because of the `* k`; without looking through the multiply the offset reads
/// as an arbitrary index scaled by one tile, and a transfer that is correct by
/// construction gets rejected.
///
/// Deliberately syntactic and shallow: it recognises the arithmetic the index
/// lowering emits and claims nothing otherwise. Under-reporting costs a
/// rejected program, which is visible; over-reporting would emit a misaligned
/// DMA, which is not.
static int64_t knownMultipleOf(OpFoldResult ofr) {
  if (std::optional<int64_t> constant = getConstantIntValue(ofr))
    return *constant ? std::abs(*constant) : 0;
  auto value = dyn_cast<Value>(ofr);
  if (!value)
    return 1;
  Operation *def = value.getDefiningOp();
  if (!def)
    return 1; // a block argument: an induction variable, say
  if (auto mul = dyn_cast<arith::MulIOp>(def))
    return knownMultipleOf(mul.getLhs()) * knownMultipleOf(mul.getRhs());
  // A sum is a multiple of what both sides share.
  if (auto add = dyn_cast<arith::AddIOp>(def))
    return std::gcd(knownMultipleOf(add.getLhs()),
                    knownMultipleOf(add.getRhs()));
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return std::gcd(knownMultipleOf(sub.getLhs()),
                    knownMultipleOf(sub.getRhs()));
  if (auto shl = dyn_cast<arith::ShLIOp>(def))
    if (std::optional<int64_t> by = getConstantIntValue(shl.getRhs()))
      if (*by >= 0 && *by < 62)
        return knownMultipleOf(shl.getLhs()) << *by;
  return 1;
}

static LogicalResult getBasePtrAndOffset(CppEmitter &emitter, Value v,
                                         Value &basePtr,
                                         std::string &offsetExpr,
                                         int64_t &offsetAlign) {
  offsetExpr.resize(0);
  offsetExpr.append("0");
  // Zero is a multiple of everything, so the neutral element of the running
  // gcd is 0 rather than 1.
  offsetAlign = 0;

  // The offset is expressed in bytes: both sides of a DMA are addressed as
  // char arrays (an MRAM buffer is declared as one, and a WRAM array is cast
  // to one), so an offset in elements would land at 1/sizeof(element) of the
  // intended address.
  const int64_t elementBytes =
      llvm::cast<MemRefType>(v.getType()).getElementTypeBitWidth() / 8;

  // Peel ignorable ops, then check for an optional single subview.
  v = skipIgnorableOps(v);
  if (auto view =
          llvm::dyn_cast_or_null<memref::SubViewOp>(v.getDefiningOp())) {
    // Peel ignorable ops between the subview and the allocation.
    Value source = skipIgnorableOps(view.getSource());

    // A buffer is addressed here as base pointer + one linear offset, so only
    // a single subview can be expressed. --fold-memref-alias-ops composes
    // chains into one before translation; if a chain reaches this point the
    // pipeline has changed, and refusing beats emitting a wrong address.
    if (llvm::isa_and_nonnull<memref::SubViewOp>(source.getDefiningOp()))
      return view.emitOpError(
          "nested memref.subview is not supported by the UPMEM C translator; "
          "run --fold-memref-alias-ops to compose the chain into one view");

    // The offset of a subview within its source is the dot product of its
    // offsets with the *source's strides*. Using the subview's sizes instead
    // happens to agree only when every dimension is either fully covered or
    // offset zero, which is why this went unnoticed: the hand-written
    // templates address their MRAM buffers directly and never take a subview.
    SmallVector<int64_t> strides;
    int64_t sourceOffset = 0;
    if (failed(view.getSourceType().getStridesAndOffset(strides, sourceOffset)))
      return view.emitOpError("subview source has no strided layout");

    for (auto [off, stride] :
         llvm::zip_equal(view.getMixedOffsets(), strides)) {
      if (ShapedType::isDynamic(stride))
        return view.emitOpError(
            "subview source has a dynamic stride, so its offset cannot be "
            "computed at compile time");
      // Skip the zero terms: they contribute nothing and the generated C is
      // read by humans.
      if (isConstantIntValue(off, 0))
        continue;
      const int64_t multiplier = stride * elementBytes;
      offsetAlign = std::gcd(offsetAlign, knownMultipleOf(off) * multiplier);
      offsetExpr.append(" + (");
      emitter.appendNameOrInt(off, offsetExpr);
      offsetExpr.append(" * ");
      offsetExpr.append(std::to_string(multiplier));
      offsetExpr.append(")");
    }
    v = source;
  }

  return getBasePtrOfAlloc(v.getDefiningOp(), basePtr);
}

/// Granularity of an MRAM DMA, in bytes. `mram_read`/`mram_write` require the
/// MRAM address, the WRAM address and the length to all be multiples of this.
static constexpr int64_t kDmaAlignBytes = 8;

/// Rejects a transfer endpoint whose address is not provably DMA-aligned.
///
/// A tile smaller than the DMA granule is the usual way to get here: the
/// per-tasklet slice of a shared MRAM buffer sits at `tasklet * tileBytes`,
/// which is only 8-byte aligned when the tile itself is a whole number of
/// granules. The hardware drops the low bits of a misaligned MRAM address, so
/// the tasklets of a DPU would silently overwrite each other's slices --
/// hence an error here rather than a best-effort address.
static LogicalResult checkDmaAlignment(upmem::LocalTransferOp op,
                                       StringRef side, int64_t offsetAlign,
                                       StringRef offsetExpr) {
  if (offsetAlign % kDmaAlignBytes == 0)
    return success();
  return op->emitOpError("cannot emit a DMA whose ")
         << side << " address is not " << kDmaAlignBytes
         << "-byte aligned: the byte offset `" << offsetExpr
         << "` is only known to be a multiple of " << offsetAlign
         << ". An MRAM DMA drops the low bits of a misaligned address, so this "
            "would move the right bytes to the wrong place. Size the staged "
            "tile so that its footprint is a multiple of "
         << kDmaAlignBytes << " bytes";
}

/// Diagnoses a transfer whose region is not packed, naming the shape and
/// strides so the offending layout is identifiable from the message alone.
static LogicalResult reportNonContiguous(upmem::LocalTransferOp op,
                                         StringRef side, MemRefType ty) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  InFlightDiagnostic diag =
      op->emitOpError("cannot emit a DMA for a non-contiguous ")
      << side << ": " << ty
      << ". A transfer lowers to one flat copy of its whole element count, "
         "which would move the right number of bytes from the wrong "
         "addresses. Give the buffer a layout whose staged slices are packed";
  if (succeeded(ty.getStridesAndOffset(strides, offset)))
    diag << " (strides " << strides << ")";
  return diag;
}

static LogicalResult printLocalTransfer(CppEmitter &emitter,
                                        upmem::LocalTransferOp memcpyOp) {
  using upmem::DpuMemSpace::MRAM;
  using upmem::DpuMemSpace::WRAM;
  raw_ostream &os = emitter.ostream();
  auto from = memcpyOp.getSource();
  auto to = memcpyOp.getTarget();
  upmem::DpuMemSpace fromSpace;
  bool withinWram = false;
  if (isInMemspace(from.getType(), MRAM) && isInMemspace(to.getType(), WRAM)) {
    fromSpace = MRAM;
  } else if (isInMemspace(from.getType(), WRAM) &&
             isInMemspace(to.getType(), MRAM)) {
    fromSpace = WRAM;
  } else if (isInMemspace(from.getType(), WRAM) &&
             isInMemspace(to.getType(), WRAM)) {
    // Both ends in WRAM: a plain copy, not a DMA. This is what a tasklet
    // writing its result into its slot of a pooled buffer becomes, and the
    // reason the granule rules below do not apply to it -- WRAM is addressed
    // by the core, byte by byte, so neither the length nor the offset has
    // anything to be aligned to.
    fromSpace = WRAM;
    withinWram = true;
  } else {
    return memcpyOp->emitOpError(
        "TODO only supports transfers from mram to wram or the reverse");
  }
  if (!from.getType().hasStaticShape() || !to.getType().hasStaticShape())
    return memcpyOp->emitOpError("Unsupported: dynamic count transfer");

  if (from.getType().getNumElements() != to.getType().getNumElements())
    return memcpyOp->emitOpError(
        "Copy source and target don't have same number of elements");

  if (!memrefIsContiguous(from.getType()))
    return reportNonContiguous(memcpyOp, "source", from.getType());
  if (!memrefIsContiguous(to.getType()))
    return reportNonContiguous(memcpyOp, "target", to.getType());

  auto remainingBytes = from.getType().getNumElements() *
                        from.getType().getElementTypeBitWidth() / 8;

  Value fromBaseW, toBaseW;
  std::string fromOffsetW, toOffsetW;
  int64_t fromAlignW, toAlignW;
  if (withinWram) {
    if (failed(getBasePtrAndOffset(emitter, from, fromBaseW, fromOffsetW,
                                   fromAlignW)) ||
        failed(getBasePtrAndOffset(emitter, to, toBaseW, toOffsetW, toAlignW)))
      return failure();
    os << "memcpy(&((char*) " << emitter.getOrCreateName(toBaseW) << ")["
       << toOffsetW << "], &((const char*) "
       << emitter.getOrCreateName(fromBaseW) << ")[" << fromOffsetW << "], "
       << remainingBytes << ")";
    return success();
  }

  Value fromBase, toBase;
  std::string fromOffset, toOffset;
  int64_t fromAlign, toAlign;
  if (failed(getBasePtrAndOffset(emitter, from, fromBase, fromOffset,
                                 fromAlign)) ||
      failed(getBasePtrAndOffset(emitter, to, toBase, toOffset, toAlign)))
    return failure();

  // A short read may take the granule it cannot avoid; a short write may not.
  //
  // Both round the length up, but only the write does damage: the bytes it
  // rounds up over belong to the next tile, and it overwrites them. A read
  // merely fetches a few bytes nobody looks at, and they have somewhere to
  // land -- every buffer is declared padded to a whole granule (see
  // printBufferDecl and the memref.alloca emitter), so as long as the
  // destination is a whole one rather than a slice of it, the tail of the
  // rounded-up read stays inside it. That is what a broadcast scalar staged
  // into WRAM is: geva's coefficients are four bytes at offset zero.
  bool isRead = fromSpace == MRAM;
  bool destIsWholeBuffer = toAlign == 0;
  if (remainingBytes % kDmaAlignBytes != 0 && !(isRead && destIsWholeBuffer))
    return memcpyOp->emitOpError("cannot emit a DMA of ")
           << remainingBytes << " bytes: a transfer length must be a multiple "
           << "of " << kDmaAlignBytes
           << ", and rounding it up would move bytes belonging to the next "
              "tile. Size the staged tile so that its footprint is a multiple "
              "of "
           << kDmaAlignBytes << " bytes";

  // Both ends of the DMA are addressed by the engine, so both have to be
  // aligned. The allocations themselves are `__dma_aligned`; only the offset
  // within them can break it.
  if (failed(checkDmaAlignment(memcpyOp, "source", fromAlign, fromOffset)) ||
      failed(checkDmaAlignment(memcpyOp, "target", toAlign, toOffset)))
    return failure();

  size_t offsetBytes = 0;
  while (remainingBytes > 0) {
    int64_t chunkSizeBytes = std::min(2048l, remainingBytes);
    // Chunks stay whole granules, so `offsetBytes` keeps both addresses
    // aligned across the loop. A tail shorter than one granule can only be
    // the permitted short read, which rounds up: there is nothing to round
    // down to, and rounding down would not terminate.
    chunkSizeBytes = chunkSizeBytes >= kDmaAlignBytes
                         ? llvm::alignDown(chunkSizeBytes, kDmaAlignBytes)
                         : llvm::alignTo(chunkSizeBytes, kDmaAlignBytes);

    if (printMRAMCopyBytes(emitter, fromSpace, fromBase, toBase, chunkSizeBytes,
                           fromOffset, toOffset, offsetBytes)
            .failed()) {
      return failure();
    }
    offsetBytes += chunkSizeBytes;
    remainingBytes -= std::min(remainingBytes, chunkSizeBytes);
    if (remainingBytes > 0) {
      os << ";\n";
    }
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValueAttr();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printBinaryOperation(CppEmitter &emitter,
                                          Operation *operation,
                                          StringRef binaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  if (printValueOrConstant(emitter, operation->getOperand(0)).failed()) {
    return failure();
  }
  os << " " << binaryOperator << " ";
  if (printValueOrConstant(emitter, operation->getOperand(1)).failed()) {
    return failure();
  }

  return success();
}

// arith ops

static LogicalResult printOperation(CppEmitter &emitter, arith::AddFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "+");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::AddIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "+");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::AddUIExtendedOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "+");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::AndIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "&");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::BitcastOp op) {
  // memcpy-based bitcast: (dst_type)(*(src_type*)&operand)
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op.getOperation())))
    return failure();
  os << "*(";
  if (failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  os << "*)&" << emitter.getOrCreateName(op.getIn());
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::CeilDivSIOp op) {
  // (a / b) + (((a % b) != 0) & ((a ^ b) >= 0))
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op.getOperation())))
    return failure();
  StringRef a = emitter.getOrCreateName(op.getLhs());
  StringRef b = emitter.getOrCreateName(op.getRhs());
  os << "(" << a << " / " << b << ") + (((" << a << " % " << b << ") != 0) & (("
     << a << " ^ " << b << ") >= 0))";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::CeilDivUIOp op) {
  // (a + b - 1) / b  (unsigned, no overflow risk when a>0)
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op.getOperation())))
    return failure();
  StringRef a = emitter.getOrCreateName(op.getLhs());
  StringRef b = emitter.getOrCreateName(op.getRhs());
  os << "(" << a << " + " << b << " - 1) / " << b;
  return success();
}

static LogicalResult printCmpOp(CppEmitter &emitter, Operation *op,
                                StringRef cmpOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  if (failed(printValueOrConstant(emitter, op->getOperand(0))))
    return failure();
  os << " " << cmpOp << " ";
  return printValueOrConstant(emitter, op->getOperand(1));
}

static LogicalResult printOperation(CppEmitter &emitter, arith::CmpFOp op) {
  StringRef cmpOp;
  switch (op.getPredicate()) {
  // Ordered comparisons (false if either is NaN)
  case arith::CmpFPredicate::OEQ:
    cmpOp = "==";
    break;
  case arith::CmpFPredicate::OGT:
    cmpOp = ">";
    break;
  case arith::CmpFPredicate::OGE:
    cmpOp = ">=";
    break;
  case arith::CmpFPredicate::OLT:
    cmpOp = "<";
    break;
  case arith::CmpFPredicate::OLE:
    cmpOp = "<=";
    break;
  case arith::CmpFPredicate::ONE:
    cmpOp = "!=";
    break;
  // Unordered: map to same C ops (NaN handling not preserved)
  case arith::CmpFPredicate::UEQ:
    cmpOp = "==";
    break;
  case arith::CmpFPredicate::UGT:
    cmpOp = ">";
    break;
  case arith::CmpFPredicate::UGE:
    cmpOp = ">=";
    break;
  case arith::CmpFPredicate::ULT:
    cmpOp = "<";
    break;
  case arith::CmpFPredicate::ULE:
    cmpOp = "<=";
    break;
  case arith::CmpFPredicate::UNE:
    cmpOp = "!=";
    break;
  default:
    return op->emitOpError("unsupported CmpF predicate");
  }
  return printCmpOp(emitter, op.getOperation(), cmpOp);
}

static LogicalResult printOperation(CppEmitter &emitter, arith::CmpIOp op) {
  StringRef cmpOp;
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    cmpOp = "==";
    break;
  case arith::CmpIPredicate::ne:
    cmpOp = "!=";
    break;
  case arith::CmpIPredicate::slt:
    cmpOp = "<";
    break;
  case arith::CmpIPredicate::sle:
    cmpOp = "<=";
    break;
  case arith::CmpIPredicate::sgt:
    cmpOp = ">";
    break;
  case arith::CmpIPredicate::sge:
    cmpOp = ">=";
    break;
  case arith::CmpIPredicate::ult:
    cmpOp = "<";
    break;
  case arith::CmpIPredicate::ule:
    cmpOp = "<=";
    break;
  case arith::CmpIPredicate::ugt:
    cmpOp = ">";
    break;
  case arith::CmpIPredicate::uge:
    cmpOp = ">=";
    break;
  }
  return printCmpOp(emitter, op.getOperation(), cmpOp);
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ConstantOp op) {
  return printConstantOp(emitter, op.getOperation(), op.getValue());
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivSIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivUIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "/");
}

// Helper for any C-cast expression: (result_type)operand
static LogicalResult printCastOp(CppEmitter &emitter, Operation *op) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  os << "(";
  if (failed(emitter.emitType(op->getLoc(), op->getResult(0).getType())))
    return failure();
  os << ")";
  return printValueOrConstant(emitter, op->getOperand(0));
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ExtFOp op) {
  return printCastOp(emitter, op.getOperation());
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ExtSIOp op) {
  return printCastOp(emitter, op.getOperation());
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ExtUIOp op) {
  return printCastOp(emitter, op.getOperation());
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::FloorDivSIOp op) {
  // Signed floor div: (a - (((a % b) != 0) & ((a ^ b) < 0))) / b
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op.getOperation())))
    return failure();
  StringRef a = emitter.getOrCreateName(op.getLhs());
  StringRef b = emitter.getOrCreateName(op.getRhs());
  os << "(" << a << " - (((" << a << " % " << b << ") != 0) & ((" << a << " ^ "
     << b << ") < 0))) / " << b;
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, arith::FPToSIOp op) {
  return printCastOp(emitter, op.getOperation());
}

static LogicalResult printOperation(CppEmitter &emitter, arith::FPToUIOp op) {
  return printCastOp(emitter, op.getOperation());
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::IndexCastOp op) {
  return printCastOp(emitter, op.getOperation());
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::IndexCastUIOp op) {
  return printCastOp(emitter, op.getOperation());
}

// Helper for integer min/max via ternary: (a OP b) ? a : b
static LogicalResult printMinMaxOp(CppEmitter &emitter, Operation *op,
                                   StringRef cmpOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  StringRef a = emitter.getOrCreateName(op->getOperand(0));
  StringRef b = emitter.getOrCreateName(op->getOperand(1));
  os << "(" << a << " " << cmpOp << " " << b << ") ? " << a << " : " << b;
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaximumFOp op) {
  return printMinMaxOp(emitter, op.getOperation(), ">=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaxNumFOp op) {
  return printMinMaxOp(emitter, op.getOperation(), ">=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaxSIOp op) {
  return printMinMaxOp(emitter, op.getOperation(), ">=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaxUIOp op) {
  return printMinMaxOp(emitter, op.getOperation(), ">=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinimumFOp op) {
  return printMinMaxOp(emitter, op.getOperation(), "<=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinNumFOp op) {
  return printMinMaxOp(emitter, op.getOperation(), "<=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinSIOp op) {
  return printMinMaxOp(emitter, op.getOperation(), "<=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinUIOp op) {
  return printMinMaxOp(emitter, op.getOperation(), "<=");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MulFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "*");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MulIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "*");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::MulSIExtendedOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::MulUIExtendedOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::NegFOp op) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op.getOperation())))
    return failure();
  os << "-" << emitter.getOrCreateName(op.getOperand());
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, arith::OrIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "|");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::RemFOp op) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op.getOperation())))
    return failure();
  os << "fmodf(" << emitter.getOrCreateName(op.getLhs()) << ", "
     << emitter.getOrCreateName(op.getRhs()) << ")";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, arith::RemSIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "%");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::RemUIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "%");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SelectOp op) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*op.getOperation())))
    return failure();
  os << emitter.getOrCreateName(op.getCondition()) << " ? "
     << emitter.getOrCreateName(op.getTrueValue()) << " : "
     << emitter.getOrCreateName(op.getFalseValue());
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ShLIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "<<");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ShRSIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), ">>");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ShRUIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), ">>");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SIToFPOp op) {
  return printCastOp(emitter, op.getOperation());
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SubFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "-");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SubIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "-");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::TruncFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::TruncIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::UIToFPOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::XOrIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "^");
}

static LogicalResult printOperation(CppEmitter &emitter, LLVM::ExpOp op) {
  if (emitter.emitAssignPrefix(*op.getOperation()).failed()) {
    return failure();
  }

  emitter.ostream() << "expf(" << emitter.getOrCreateName(op.getOperand())
                    << ")";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    cf::BranchOp branchOp) {
  raw_ostream &os = emitter.ostream();
  Block &successor = *branchOp.getSuccessor();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = ";
    if (printValueOrConstant(emitter, operand).failed())
      return failure();
    os << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(successor)))
    return branchOp.emitOpError("unable to find label for successor block");
  os << emitter.getOrCreateName(successor);
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    cf::CondBranchOp condBranchOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
     << ") {\n";

  os.indent();

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
                             trueSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = ";
    if (printValueOrConstant(emitter, operand).failed())
      return failure();
    os << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError("unable to find label for successor block");
  }
  os << emitter.getOrCreateName(trueSuccessor) << ";\n";
  os.unindent() << "} else {\n";
  os.indent();
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
                             falseSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = ";
    if (printValueOrConstant(emitter, operand).failed())
      return failure();
    os << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "unable to find label for successor block";
  }
  os << emitter.getOrCreateName(falseSuccessor) << ";\n";
  os.unindent() << "}";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, func::CallOp callOp) {
  if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callOp.getCallee() << "(";
  if (failed(emitter.emitOperands(*callOp.getOperation())))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printDPUReset(CppEmitter &emitter) {
  raw_ostream &os = emitter.ostream();

  os << "dpu_reset();\n";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, scf::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  OperandRange operands = forOp.getInitArgs();
  Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
  Operation::result_range results = forOp.getResults();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : results) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    if (printValueOrConstant(emitter, std::get<1>(pair)).failed())
      return failure();
    os << ";";
    os << "\n";
  }

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if (printValueOrConstant(emitter, forOp.getLowerBound()).failed()) {
    return failure();
  }
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  if (printValueOrConstant(emitter, forOp.getUpperBound()).failed()) {
    return failure();
  }
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if (printValueOrConstant(emitter, forOp.getStep()).failed()) {
    return failure();
  }
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = ";
    if (printValueOrConstant(emitter, operand).failed())
      return failure();
    os << ";\n";
  }

  os.unindent() << "}";

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    OpResult result = std::get<0>(pair);
    BlockArgument iterArg = std::get<1>(pair);
    os << "\n"
       << emitter.getOrCreateName(result) << " = "
       << emitter.getOrCreateName(iterArg) << ";";
  }

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, scf::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : ifOp.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &thenRegion = ifOp.getThenRegion();
  for (Operation &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op has
    // zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();

    for (Operation &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, scf::YieldOp yieldOp) {
  raw_ostream &os = emitter.ostream();
  Operation &parentOp = *yieldOp.getOperation()->getParentOp();

  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return yieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            // Constants are inlined at every use (see printValueOrConstant)
            // rather than being assigned a declared variable name, so the
            // scope check below only applies to non-constant operands.
            if (!isa_and_nonnull<arith::ConstantOp>(operand.getDefiningOp()) &&
                !emitter.hasValueInScope(operand))
              return yieldOp.emitError("operand value not in scope");
            return printValueOrConstant(emitter, operand);
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

// static LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
//   CppEmitter::Scope scope(emitter);

//   for (Operation &op : moduleOp) {
//     if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
//       return failure();
//   }
//   return success();
// }

static LogicalResult printBufferDecl(CppEmitter &emitter,
                                     upmem::StaticAllocOp op) {
  StringRef qualifier;
  if (op.isWram()) {
    qualifier = "__dma_aligned";
  } else if (op.getNoinit()) {
    qualifier = "__mram_noinit __dma_aligned";
  } else {
    qualifier = "__mram __dma_aligned";
  }

  // MRAM buffers are emitted as arrays of bytes to be able to pad them; their
  // bytes only ever move through mram_read/mram_write, which index in bytes
  // anyway. A WRAM buffer is dereferenced element by element (memref.load /
  // memref.store print `name[i]` with `i` an *element* index), so it keeps
  // its element type -- as a byte array those accesses would read one byte at
  // an offset scaled wrong by the element width. The transfer emitters cast
  // the WRAM side to char* themselves, so byte addressing still works there.
  auto &out = emitter.ostream();
  auto bufferType = op.getBuffer().getType();
  auto eltWidthBytes = bufferType.getElementTypeBitWidth() / 8;
  if (op.isWram()) {
    if (failed(emitter.emitType(op->getLoc(), bufferType.getElementType())))
      return failure();
    out << " " << qualifier << " ";
  } else {
    out << "char " << qualifier << " ";
  }
  if (auto name = op.getSymName()) {
    if (failed(emitter.recordStaticName(op.getBuffer(), *name)))
      return failure();
    out << *name;
  } else {
    name = emitter.getOrCreateName(op.getBuffer());
    out << *name;
  }

  if (op.isWram()) {
    // Padded to the 8-byte transfer granularity, in elements.
    out << "["
        << llvm::alignTo(bufferType.getNumElements(),
                         std::max<int64_t>(1, 8 / eltWidthBytes))
        << "]";
  } else {
    auto sizeInBytes = bufferType.getNumElements() * eltWidthBytes;
    sizeInBytes = llvm::alignTo(sizeInBytes, 8);
    out << "[" << sizeInBytes << "]";
  }
  if (op.getZeroinit()) {
    out << " {0}";
  }

  out << "; // ";
  // add real type as comment
  if (failed(emitter.emitType(op->getLoc(), bufferType.getElementType())))
    return failure();

  for (auto dim : bufferType.getShape()) {
    out << '[' << dim << ']';
  }
  out << "\n";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::DpuProgramOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBody().getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  // walk and declare all static buffers
  WalkResult result =
      functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        LogicalResult result = LogicalResult::success();
        if (auto alloc = llvm::dyn_cast_or_null<upmem::StaticAllocOp>(op)) {
          result = printBufferDecl(emitter, alloc);
        }
        if (failed(result))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (result.wasInterrupted())
    return failure();

  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  // if (failed(emitter.emitTypes(functionOp.getLoc(),
  //  functionOp.getFunctionType().getResults())))
  // return failure();
  os << "void " << functionOp.getName() << "(void) {\n";

  os.indent();
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  Region::BlockListType &blocks = functionOp.getBody().getBlocks();
  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (Block &block : llvm::drop_begin(blocks)) {
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an scf.if or cf.cond_br op no semicolon needs
      // to be printed after the closing brace.
      // When generating code for an scf.for op, printing a trailing semicolon
      // is handled within the printOperation function.
      bool trailingSemicolon =
          !isa<cf::CondBranchOp, scf::IfOp, scf::ForOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os.unindent() << "}\n";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, upmem::BarrierOp) {
  emitter.ostream() << "barrier_wait(&my_barrier)";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, upmem::ReturnOp) {
  emitter.ostream() << "return";
  return success();
}

static void printCompilationVar(upmem::DpuProgramOp kernel, raw_ostream &os) {
  os << "COMPILE_" << kernel.getSymName();
}

/*
  Upmem's default stack size is very small (2048 bytes),
  but we allocate all private wram buffers on the stack.
  Global wram allocations are not tasklet-private so we
  have to allocate on the stack. We need to estimate how
  much stack each tasklet will require though and write
  that out as a compiler argument.

  The estimate lives in UPMEMOccupancy.h, shared with
  --upmem-check-occupancy: that pass decides whether a program fits the device
  using this same number, so the two must not drift. It used to be computed
  here from unpadded element counts, which under-estimated the arrays printed
  by printOperation(AllocaOp) below -- those are padded.
*/

static LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  llvm::SmallVector<upmem::DpuProgramOp> kernels;
  for (Operation &op : moduleOp) {
    if (auto prog = llvm::dyn_cast_or_null<upmem::DpuProgramOp>(op)) {
      kernels.push_back(prog);
    }
  }

  if (kernels.empty())
    return failure();

  raw_ostream &os = emitter.ostream();

  os << "// UPMEM-TRANSLATE: ";
  for (auto kernel : kernels) {
    // The compilation var is used to compile only one of
    // the kernels when many can be generated into the
    // same C file, with different tasklet numbers and other
    // parameters.
    printCompilationVar(kernel, os);
    // Whether this actually fits WRAM is checked by --upmem-check-occupancy,
    // which has the platform in hand; by the time we get here the capacity is
    // no longer reachable from the IR.
    auto stackSize = upmem::taskletStackBytes(kernel);
    os << ":" << kernel.getNumTasklets();
    os << ":" << stackSize;
    os << ":" << kernel.getSymName(); // name of the binary
    os << ";";
  }

  os << "\n\n";

  os << "#include <alloc.h>\n"
        "#include <barrier.h>\n"
        "#include <defs.h>\n"
        "#include <mram.h>\n"
        "#include <perfcounter.h>\n\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n\n"
        // "#include \"expf.c\"\n"
        "\n\n";

  os << "BARRIER_INIT(my_barrier, NR_TASKLETS);\n\n";

  for (auto kernel : kernels) {
    os << "#ifdef ";
    printCompilationVar(kernel, os);
    os << "\n";
    if (failed(printOperation(emitter, kernel)))
      return failure();
    os << "#endif\n\n";
  }

  os << "int main(void) {\n";
  // os << "  barrier_wait(&my_barrier);\n";
  // os << "  mem_reset();\n";
  for (auto kernel : kernels) {
    os << "#ifdef ";
    printCompilationVar(kernel, os);
    os << "\n";
    os << "  " << kernel.getName() << "();\n";
    os << "#endif\n";
  }
  // os << "  mem_reset();\n";
  os << "  return 0;\n";
  os << "}";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::FuncOp functionOp) {
  return success();
}

CppEmitter::CppEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef CppEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}
LogicalResult CppEmitter::recordStaticName(Value val, StringRef name) {
  if (valueMapper.count(val) && valueMapper.lookup(val) != name)
    return failure();
  std::string str(name);
  valueMapper.insert(val, std::move(str));
  return success();
}

/// Return the existing or a new label for a Block.
StringRef CppEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool CppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool CppEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CppEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult CppEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "(float)";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        os << "(double)";
        break;
      default:
        break;
      };
      os << strValue;
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print symbolic reference attributes.
  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand value not in scope";
    os << getOrCreateName(result);
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
CppEmitter::emitOperandsAndAttributes(Operation &op,
                                      ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result,
                                                  bool trailingSemicolon) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
    return failure();
  os << " " << getOrCreateName(result);
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CppEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  if (isa<arith::ConstantOp>(op) || isa<upmem::StaticAllocOp>(op)) {
    return success();
  }

  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // CF ops.
          .Case<cf::BranchOp, cf::CondBranchOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arith ops
          .Case<arith::MulIOp, arith::AddIOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::CallOp, func::ConstantOp, func::FuncOp,
                upmem::DpuProgramOp, func::ReturnOp, upmem::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // SCF ops.
          .Case<scf::ForOp, scf::IfOp, scf::YieldOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arithmetic ops.
          .Case<arith::AddFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AddIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AddUIExtendedOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AndIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::BitcastOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CeilDivSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CeilDivUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CmpFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CmpIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ConstantOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::DivFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::DivSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::DivUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ExtFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ExtSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ExtUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::FloorDivSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::FPToSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::FPToUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::IndexCastOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::IndexCastUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaximumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaxNumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaxSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaxUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinimumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinNumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulSIExtendedOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulUIExtendedOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::NegFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::OrIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::RemFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::RemSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::RemUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SelectOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ShLIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ShRSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ShRUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SIToFPOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SubFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SubIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::TruncFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::TruncIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::UIToFPOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::XOrIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<LLVM::ExpOp>([&](auto op) { return printOperation(*this, op); })
          .Case<upmem::TaskletDimOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::AllocaOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<upmem::LocalTransferOp>(
              [&](auto op) { return printLocalTransfer(*this, op); })
          .Case<memref::LoadOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::StoreOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<upmem::BarrierOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::SubViewOp, memref::ExpandShapeOp,
                memref::CollapseShapeOp, memref::CastOp,
                memref::ReinterpretCastOp>([&](auto) -> LogicalResult {
            // fine, handled by local transfer printer
            return success();
          })
          // [&](auto op) { skipSemicolon = true; return success(); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "int32_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  if (auto pType = dyn_cast<MemRefType>(type)) {
    Type type = pType.getElementType();
    if (auto t = dyn_cast<IntegerType>(type)) {
      os << "int ";
    } else if (auto t = dyn_cast<FloatType>(type)) {
      os << "float ";
    }
    os << "*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult CppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult upmem_emitc::UPMEMtranslateToCpp(Operation *op, raw_ostream &os,
                                               bool declareVariablesAtTop) {
  CppEmitter emitter(os, declareVariablesAtTop);
  LogicalResult res = success();
  op->walk<WalkOrder::PreOrder>([&](Operation *child) {
    // todo we should not hardcode the module name
    if (auto mod = llvm::dyn_cast_or_null<ModuleOp>(child)) {
      if (mod.getSymName() == "dpu_kernels") {
        res = emitter.emitOperation(*child, /*trailingSemicolon=*/false);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    }
    if (auto dpuProg = llvm::dyn_cast_or_null<upmem::DpuProgramOp>(child)) {
      res = emitter.emitOperation(*child->getParentOfType<ModuleOp>(),
                                  /*trailingSemicolon=*/false);
      return WalkResult::interrupt();
    }
    return WalkResult::skip();
  });
  return res;
}

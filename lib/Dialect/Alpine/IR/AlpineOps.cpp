#include <cinm-mlir/Dialect/Alpine/IR/AlpineOps.h>

#include <cinm-mlir/Dialect/Alpine/IR/AlpineTypes.h>
#include <cinm-mlir/Utils/CinmUtils.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>

#define DEBUG_TYPE "alpine-ops"

using namespace mlir;
using namespace mlir::alpine;


#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Alpine/IR/AlpineOps.cpp.inc"


void AlpineDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Alpine/IR/AlpineOps.cpp.inc"
      >();
}


void AllocTileOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "alpine_tile");
}

static void appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = dyn_cast<MemRefType>(t)) {
    ss << "view";
    for (auto size : memref.getShape())
      if (ShapedType::isDynamic(size))
        ss << "sx";
      else
        ss << size << "x";
    appendMangledType(ss, memref.getElementType());
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
  } else {
    llvm_unreachable("Invalid type for alpine library name mangling");
  }
}

std::string mlir::alpine::generateLibraryCallName(Operation *op) {
  std::string name(op->getName().getStringRef().str());
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);

  ss << "_";
  auto types = op->getOperandTypes();
  llvm::interleave(
      types.begin(), types.end(), [&](Type t) { appendMangledType(ss, t); },
      [&]() { ss << "_"; });

  ss << "_";
  auto attrs = op->getAttrs();
  llvm::interleave(
      attrs.begin(), attrs.end(),
      [&](NamedAttribute attr) { ss << attr.getName().getValue(); },
      [&]() { ss << "_"; });

  return ss.str();
}

void mlir::alpine::appendOperandPrecision(llvm::raw_string_ostream &ss,
                                          Type t) {
  if (auto memref = dyn_cast<MemRefType>(t)) {
    appendOperandPrecision(ss, memref.getElementType());
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << "_";
    ss << t;
  } else {
    llvm_unreachable("Invalid type for alpine library precision mangling");
  }
}

static bool isI8MemRef(Type t) {
  if (auto mem = dyn_cast<MemRefType>(t))
    return mem.getElementType().isInteger(8);
  return false;
}

LogicalResult WriteWeightsOp::verify() {
  if (!isI8MemRef(getW().getType()))
    return emitOpError("expects weights memref element type i8");
  return success();
}

LogicalResult EnqueueVecOp::verify() {
  if (!isI8MemRef(getX().getType()))
    return emitOpError("expects input vector memref element type i8");
  return success();
}

LogicalResult DequeueVecOp::verify() {
  if (!isI8MemRef(getY().getType()))
    return emitOpError("expects output vector memref element type i8");
  return success();
}

LogicalResult MVMOp::verify() {
  if (!isI8MemRef(getX().getType()) || !isI8MemRef(getY().getType()))
    return emitOpError("expects x and y to be memref<?xi8>");
  return success();
}

LogicalResult ProcessOp::verify() {
  if (auto act = getActivationAttr()) {
    StringRef s = act.getValue();
    if (s != "none" && s != "relu" && s != "clamp")
      return emitOpError("activation must be one of: 'none', 'relu', 'clamp'");
  }

  if (auto cnt = getCountAttr()) {
    int64_t v = cnt.getInt();
    if (v <= 0)
      return emitOpError("count must be > 0 when specified");
  }

  return success();
}

static bool isF32MemRef(Type t) {
  if (auto mem = dyn_cast<MemRefType>(t))
    return mem.getElementType().isF32();
  return false;
}

static LogicalResult verifySameRankAndShape(Operation *op, MemRefType a,
                                            MemRefType b, StringRef aName,
                                            StringRef bName) {
  if (a.getRank() != b.getRank())
    return op->emitOpError() << "expects " << aName << " and " << bName
                             << " to have the same rank, got " << a.getRank()
                             << " and " << b.getRank();

  auto aShape = a.getShape();
  auto bShape = b.getShape();
  if (aShape != bShape)
    return op->emitOpError() << "expects " << aName << " and " << bName
                             << " to have the same shape";

  return success();
}

static LogicalResult verifyScaleZero(Operation *op, FloatAttr scaleAttr,
                                     IntegerAttr zeroAttr) {
  if (!scaleAttr)
    return op->emitOpError("requires 'scale' (f32) attribute");
  if (!zeroAttr)
    return op->emitOpError("requires 'zero' (i32) attribute");

  auto ap = scaleAttr.getValue();
  if (!ap.isFinite() || ap.isZero() || ap.isNegative())
    return op->emitOpError("scale must be finite and > 0");

  int64_t z = zeroAttr.getValue().getSExtValue();
  if (z < -128 || z > 127)
    return op->emitOpError("zero must be in [-128, 127] for i8");

  return success();
}

LogicalResult QuantizeOp::verify() {
  auto srcTy = dyn_cast<MemRefType>(getSrc().getType());
  auto outTy = dyn_cast<MemRefType>(getOut().getType());
  if (!srcTy || !outTy)
    return emitOpError("expects memref operands 'src' and 'out'");
  if (!srcTy.getElementType().isF32())
    return emitOpError("expects element type of 'src' to be f32");
  if (!outTy.getElementType().isInteger(8))
    return emitOpError("expects element type of 'out' to be i8");

  if (failed(verifySameRankAndShape(*this, srcTy, outTy, "src", "out")))
    return failure();

  return verifyScaleZero(*this, getScaleAttr(), getZeroAttr());
}

LogicalResult DequantizeOp::verify() {
  auto srcTy = dyn_cast<MemRefType>(getSrc().getType());
  auto outTy = dyn_cast<MemRefType>(getOut().getType());
  if (!srcTy || !outTy)
    return emitOpError("expects memref operands 'src' and 'out'");
  if (!srcTy.getElementType().isInteger(8))
    return emitOpError("expects element type of 'src' to be i8");
  if (!outTy.getElementType().isF32())
    return emitOpError("expects element type of 'out' to be f32");

  if (failed(verifySameRankAndShape(*this, srcTy, outTy, "src", "out")))
    return failure();

  return verifyScaleZero(*this, getScaleAttr(), getZeroAttr());
}

LogicalResult ReluOp::verify() {
  auto srcTy = dyn_cast<MemRefType>(getSrc().getType());
  auto outTy = dyn_cast<MemRefType>(getOut().getType());

  if (!srcTy || !outTy)
    return emitOpError("expects memref operands 'src' and 'out'");

  if (!srcTy.getElementType().isF32() || !outTy.getElementType().isF32())
    return emitOpError("expects element types of 'src' and 'out' to be f32");

  return verifySameRankAndShape(*this, srcTy, outTy, "src", "out");
}
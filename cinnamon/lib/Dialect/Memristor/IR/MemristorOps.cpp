/// Implements the Memristor dialect ops.
///
/// @file

#include <cinm-mlir/Dialect/Memristor/IR/MemristorOps.h>

#include <cinm-mlir/Dialect/Memristor/IR/MemristorTypes.h>
#include <cinm-mlir/Utils/CinmUtils.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "memristor-ops"

using namespace mlir;
using namespace mlir::memristor;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Memristor/IR/MemristorOps.cpp.inc"

//===----------------------------------------------------------------------===//
// MemristorDialect
//===----------------------------------------------------------------------===//

void MemristorDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Memristor/IR/MemristorOps.cpp.inc"
      >();
}

static void appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = dyn_cast<MemRefType>(t)) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    appendMangledType(ss, memref.getElementType());
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
  } else {
    llvm_unreachable("Invalid type for memristor library name mangling");
  }
}

std::string mlir::memristor::generateLibraryCallName(Operation *op) {
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

void mlir::memristor::appendOperandPrecision(llvm::raw_string_ostream &ss,
                                             Type t) {
  if (auto memref = dyn_cast<MemRefType>(t)) {
    appendOperandPrecision(ss, memref.getElementType());
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << "_";
    ss << t;
  } else {
    llvm_unreachable("Invalid type for memristor library precision mangling");
  }
}
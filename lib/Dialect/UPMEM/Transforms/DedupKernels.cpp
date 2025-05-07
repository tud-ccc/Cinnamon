//===- DuplicateFunctionElimination.cpp - Duplicate function elimination --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Transforms/Passes.h"
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir {

#define GEN_PASS_DEF_UPMEMDEDUPKERNELSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

// Define a notion of function equivalence that allows for reuse. Ignore the
// symbol name for this purpose.
struct DuplicateUPMEMFuncOpEquivalenceInfo
    : public llvm::DenseMapInfo<upmem::DpuProgramOp> {

  static unsigned getHashValue(const upmem::DpuProgramOp cFunc) {
    if (!cFunc) {
      return DenseMapInfo<upmem::DpuProgramOp>::getHashValue(cFunc);
    }

    // Aggregate attributes, ignoring the symbol name.
    llvm::hash_code hash = {};
    upmem::DpuProgramOp func = const_cast<upmem::DpuProgramOp &>(cFunc);
    StringAttr symNameAttrName = func.getSymNameAttrName();
    for (NamedAttribute namedAttr : cFunc->getAttrs()) {
      StringAttr attrName = namedAttr.getName();
      if (attrName == symNameAttrName)
        continue;
      hash = llvm::hash_combine(hash, namedAttr);
    }

    // Also hash the func body.
    func.getBody().walk([&](Operation *op) {
      hash = llvm::hash_combine(
          hash, OperationEquivalence::computeHash(
                    op, /*hashOperands=*/OperationEquivalence::ignoreHashValue,
                    /*hashResults=*/OperationEquivalence::ignoreHashValue,
                    OperationEquivalence::IgnoreLocations));
    });

    return hash;
  }

  static bool isEqual(upmem::DpuProgramOp lhs, upmem::DpuProgramOp rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    // Check discardable attributes equivalence
    if (lhs->getDiscardableAttrDictionary() !=
        rhs->getDiscardableAttrDictionary())
      return false;

    // Check properties equivalence, ignoring the symbol name.
    // Make a copy, so that we can erase the symbol name and perform the
    // comparison.
    auto pLhs = lhs.getProperties();
    auto pRhs = rhs.getProperties();
    pLhs.sym_name = nullptr;
    pRhs.sym_name = nullptr;
    if (pLhs != pRhs)
      return false;

    // Compare inner workings.
    return OperationEquivalence::isRegionEquivalentTo(
        &lhs.getBody(), &rhs.getBody(), OperationEquivalence::IgnoreLocations);
  }
};

static llvm::FailureOr<SymbolRefAttr> getSymbolPath(SymbolTable fromTable,
                                                    SymbolOpInterface target) {
  if (!fromTable.getOp()->isAncestor(target))
    return failure();

  StringAttr rootPath = target.getNameAttr();
  llvm::SmallVector<FlatSymbolRefAttr> path;
  Operation *table = SymbolTable::getNearestSymbolTable(target->getParentOp());
  while (table != fromTable.getOp()) {
    if (auto asSymbol = llvm::dyn_cast_or_null<SymbolOpInterface>(table)) {
      auto next = SymbolTable::getNearestSymbolTable(table->getParentOp());
      path.push_back(FlatSymbolRefAttr::get(rootPath));
      rootPath = asSymbol.getNameAttr();
      table = next;
    } else {
      return failure();
    }
  }
  return SymbolRefAttr::get(rootPath, path);
}

struct UPMEMDedupKernelsPass
    : public impl::UPMEMDedupKernelsPassBase<UPMEMDedupKernelsPass> {

  using impl::UPMEMDedupKernelsPassBase<
      UPMEMDedupKernelsPass>::UPMEMDedupKernelsPassBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Find unique representant per equivalent func ops.
    DenseSet<upmem::DpuProgramOp, DuplicateUPMEMFuncOpEquivalenceInfo>
        uniqueUPMEMFuncOps;
    DenseMap<StringAttr, upmem::DpuProgramOp> getRepresentant;
    DenseSet<upmem::DpuProgramOp> toBeErased;
    module.walk([&](upmem::DpuProgramOp f) {
      auto [repr, inserted] = uniqueUPMEMFuncOps.insert(f);
      getRepresentant[f.getSymNameAttr()] = *repr;
      if (!inserted) {
        toBeErased.insert(f);
      }
    });

    // Update call ops to call unique func op representants.
    llvm::SmallVector<StringRef, 2> flatRef;
    module->getParentOp()->walk([&](upmem::DpuSetOp setOp) {
      auto symtable = SymbolTable::getNearestSymbolTable(setOp);
      auto prog = setOp.resolveDpuProgram();
      if (!prog)
        return;
      upmem::DpuProgramOp callee = getRepresentant[prog.getSymNameAttr()];
      if (!callee)
        return;
      auto ref = getSymbolPath(symtable, callee);
      if (llvm::succeeded(ref)) {
        setOp.setDpuProgramAttr(*ref);
      }
    });

    // Erase redundant func ops.
    for (auto it : toBeErased) {
      it.erase();
    }
  }
};

} // namespace mlir

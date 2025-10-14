//===- DuplicateFunctionElimination.cpp - Duplicate function elimination --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Transforms/Passes.h"
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir {

#define GEN_PASS_DEF_UPMEMDEDUPKERNELSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

// Define a notion of function equivalence that allows for reuse. Ignore the
// symbol name for this purpose.
struct DuplicateUPMEMFuncOpEquivalenceInfo
    : public llvm::DenseMapInfo<upmem::UPMEMFuncOp> {

  static unsigned getHashValue(const upmem::UPMEMFuncOp cFunc) {
    if (!cFunc) {
      return DenseMapInfo<upmem::UPMEMFuncOp>::getHashValue(cFunc);
    }

    // Aggregate attributes, ignoring the symbol name.
    llvm::hash_code hash = {};
    upmem::UPMEMFuncOp func = const_cast<upmem::UPMEMFuncOp &>(cFunc);
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

  static bool isEqual(upmem::UPMEMFuncOp lhs, upmem::UPMEMFuncOp rhs) {
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

struct UPMEMDedupKernelsPass
    : public impl::UPMEMDedupKernelsPassBase<UPMEMDedupKernelsPass> {

  using impl::UPMEMDedupKernelsPassBase<
      UPMEMDedupKernelsPass>::UPMEMDedupKernelsPassBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Find unique representant per equivalent func ops.
    DenseSet<upmem::UPMEMFuncOp, DuplicateUPMEMFuncOpEquivalenceInfo>
        uniqueUPMEMFuncOps;
    DenseMap<StringAttr, upmem::UPMEMFuncOp> getRepresentant;
    DenseSet<upmem::UPMEMFuncOp> toBeErased;
    module.walk([&](upmem::UPMEMFuncOp f) {
      auto [repr, inserted] = uniqueUPMEMFuncOps.insert(f);
      getRepresentant[f.getSymNameAttr()] = *repr;
      if (!inserted) {
        toBeErased.insert(f);
      }
    });

    // Update call ops to call unique func op representants.
    llvm::SmallVector<StringRef, 2> flatRef;
    module->getParentOp()->walk([&](upmem::LaunchFuncOp callOp) {
      if (callOp.getKernelModuleName() != module.getSymName())
        return;
      upmem::UPMEMFuncOp callee = getRepresentant[callOp.getKernelName()];
      auto leaf = SymbolRefAttr::get(callee);
      callOp.setKernelAttr(
          SymbolRefAttr::get(callOp.getKernelModuleName(), {leaf}));
    });

    // Erase redundant func ops.
    for (auto it : toBeErased) {
      it.erase();
    }
  }
};

} // namespace mlir

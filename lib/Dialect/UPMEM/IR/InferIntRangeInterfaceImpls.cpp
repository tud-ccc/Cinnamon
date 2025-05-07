//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for upmem -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

using namespace mlir;
using namespace mlir::upmem;

static ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = IndexType::kInternalStorageBitWidth;
  return ConstantIntRanges::fromUnsigned(APInt(width, umin),
                                         APInt(width, umax));
}

// void RankDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
//                                   SetIntRangeFn setResultRange) {
//   auto parent = (*this)->getParentOfType<upmem::DpuProgramOp>();
//   auto max = parent.getWgShape().getNumRanks();
//   setResultRange(getResult(), getIndexRange(0, max - 1ULL));
// }

// void DPUDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
//                                  SetIntRangeFn setResultRange) {
//   auto parent = (*this)->getParentOfType<upmem::DpuProgramOp>();
//   auto max = parent.getWgShape().getNumDpusPerRank();
//   setResultRange(getResult(), getIndexRange(0, max - 1ULL));
// }
void TaskletDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                     SetIntRangeFn setResultRange) {
  auto parent = (*this)->getParentOfType<upmem::DpuProgramOp>();
  auto max = parent.getNumTasklets();
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}
//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for upmem -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace mlir::upmem;

static constexpr uint64_t kMaxRankDim = std::numeric_limits<uint32_t>::max();
static constexpr uint64_t kMaxDPUDim = 64;
static constexpr uint64_t kMaxTaskletDim = 24;

static constexpr uint64_t kMaxDim = std::numeric_limits<uint32_t>::max();

static ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = IndexType::kInternalStorageBitWidth;
  return ConstantIntRanges::fromUnsigned(APInt(width, umin),
                                         APInt(width, umax));
}

namespace {
enum class LaunchDims : uint32_t { Rank = 0, DPU = 1, Tasklet= 2 };
} // end namespace


static Value valueByDim(KernelDim dims) {
  return dims.x;
}

static uint64_t zext(uint32_t arg) { return static_cast<uint64_t>(arg); }

template <typename Op>
static std::optional<uint64_t> getKnownLaunchDim(Op op, LaunchDims type) {
  if (auto launch = op->template getParentOfType<LaunchOp>()) {
    KernelDim bounds;
    switch (type) {
    case LaunchDims::Rank:
      bounds = launch.getRankSizeOperandValue();
      break;
    case LaunchDims::DPU:
      bounds = launch.getDPUSizeOperandValue();
      break;
    case LaunchDims::Tasklet:
      bounds = launch.getTaskletSizeOperandValue();
      break;
    }
    Value maybeBound = valueByDim(bounds);
    APInt value;
    if (matchPattern(maybeBound, m_ConstantInt(&value)))
      return value.getZExtValue();
  }
  return std::nullopt;
}


void RankDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  uint64_t max = getKnownLaunchDim(*this, LaunchDims::Rank).value_or(kMaxRankDim);
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}

void DPUDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  uint64_t max = getKnownLaunchDim(*this, LaunchDims::DPU).value_or(kMaxDPUDim);
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}
void TaskletDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  uint64_t max = getKnownLaunchDim(*this, LaunchDims::Tasklet).value_or(kMaxTaskletDim);
  setResultRange(getResult(), getIndexRange(0, max - 1ULL));
}


void LaunchOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                 SetIntRangeFn setResultRange) {
  auto setRange = [&](const ConstantIntRanges &argRange, Value dimResult,
                      Value idxResult) {
    if (argRange.umin().getBitWidth() != IndexType::kInternalStorageBitWidth)
      return;
    ConstantIntRanges dimRange =
        argRange.intersection(getIndexRange(1, kMaxDim));
    setResultRange(dimResult, dimRange);
    ConstantIntRanges idxRange =
        getIndexRange(0, dimRange.umax().getZExtValue() - 1);
    setResultRange(idxResult, idxRange);
  };

  argRanges = argRanges.drop_front(getAsyncDependencies().size());
  KernelDim rankDim = getRankSizeClass();
  KernelDim rankId = getRankIdClass();
  KernelDim dpuDim = getDPUSizeClass();
  KernelDim dpuId = getDPUIdClass();
  KernelDim taskletDim = getTaskletSizeClass();
  KernelDim taskletId = getTaskletIdClass();
  setRange(argRanges[0], rankDim.x, rankId.x);
  setRange(argRanges[1], dpuDim.x, dpuId.x);
  setRange(argRanges[2], taskletDim.x, taskletId.x);
}

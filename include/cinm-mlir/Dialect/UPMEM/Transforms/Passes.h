/// Declaration of the transform pass within UPMEM dialect.
///
/// @file

#pragma once

// The inference pass options name types from both: the simulator backend, and
// InferenceOptions::Acquisition.
#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::upmem {

/// Which fragmented transfers -- those arriving as several blocks per leaf --
/// are repacked into one whole-buffer block, trading the blocks for a repack
/// of the host value. See --cnm-ensure-scatter-gather-contiguous.
enum class PackFragmentedTransfers {
  /// Leave every fragmented transfer in the block form.
  NONE = 0,
  /// Repack only operands whose data is the same on every inference, so the
  /// repack amortizes over the serving lifetime.
  STATIC = 1,
  /// Repack per-inference operands too: a copy on every call, in exchange for
  /// every scatter staying in the flat whole-array form.
  ALL = 2
};

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

} // namespace mlir::upmem

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

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

} // namespace mlir::upmem

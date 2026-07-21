/// Registration for generic, dialect-independent debug passes.
///
/// @file

#pragma once

namespace mlir::cinm {

/// Registers the `-print-module-ir` pass: a debug-only pass that prints the
/// current module to stdout without modifying it. Has no dialect
/// dependencies, so it can be scheduled anywhere in a pass pipeline.
void registerPrintModuleIRPass();

} // namespace mlir::cinm

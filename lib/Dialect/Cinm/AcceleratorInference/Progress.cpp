#include "Progress.h"

#include <llvm/Support/CommandLine.h>
#include <unistd.h>

/// True when cinm-opt was invoked with `-o <file>` (result IR is written to
/// that file rather than stdout). In that case stdout carries no IR and is free
/// for live progress rendering.
static bool resultGoesToFile() {
  auto &opts = llvm::cl::getRegisteredOptions();
  auto it = opts.find("o");
  if (it == opts.end() || !it->second)
    return false;
  // The `-o` option registered by MlirOptMain is a cl::opt<std::string>
  // defaulting to "-" (stdout).
  auto *opt = static_cast<llvm::cl::opt<std::string> *>(it->second);
  const std::string &v = opt->getValue();
  return !v.empty() && v != "-";
}

bool canRenderProgress() {
  return resultGoesToFile();// && ::isatty(fileno(stdout));
}

//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Target/UPMEMCpp/UPMEMCppEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::upmem_emitc;

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void mlir::upmem_emitc::registerUPMEMCppTranslation() {
  static llvm::cl::opt<bool> declareVarsAtTop(
      "declare-vars-at-top",
      llvm::cl::desc("Declare variables at top when emitting C/C++"),
      llvm::cl::init(false));

  TranslateFromMLIRRegistration reg(
      "mlir-to-upmem-cpp", "translate from upmem's mlir to upmem's cpp",
      [](Operation *op, raw_ostream &output) {
        return upmem_emitc::UPMEMtranslateToCpp(
            op, output,
            /*declareVariablesAtTop=*/declareVarsAtTop);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<LLVM::LLVMDialect>();
        registry.insert<math::MathDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<upmem::UPMEMDialect>();
        // clang-format on
      });
}


//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimDialect.h"
#include "cinm-mlir/Target/HbmpimCpp/HbmpimCppEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;
using namespace mlir::hbmpim_emitc;

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void mlir::hbmpim_emitc::registerHbmpimCppTranslation() {

  TranslateFromMLIRRegistration reg(
      "mlir-to-hbmpim-cpp", "translate from hbmpim's mlir to hbpim's cpp",
      [](Operation *op, raw_ostream &output) {
        return hbmpim_emitc::HbmpimTranslateToCpp(
            op, output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registerAllDialects(registry);

        registry.insert<hbmpim::HbmpimDialect>();
        // clang-format on
      });
}


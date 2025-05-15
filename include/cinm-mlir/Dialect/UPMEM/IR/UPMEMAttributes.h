/// Declaration of the UPMEM dialect attributes.
///
/// @file

#pragma once

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h>
#include <mlir/IR/Attributes.h>
#include <tilefirst-mlir/Dialect/Btfl/IR/BtflAttributes.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstDialect.h>

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMEnumAttrs.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMT1Attributes.h.inc"
//===----------------------------------------------------------------------===//

namespace mlir::upmem {

enum class BufferMemspace { WRAM, MRAM, HOST };
tilefirst::Maybe<BufferMemspace>
getUpmemMemspace(Location loc, tilefirst::TfMemSpaceAttr memspace,
                 bool allowHost);
} // namespace mlir::upmem
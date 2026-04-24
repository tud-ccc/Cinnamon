

#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Transform/Interfaces/TransformInterfaces.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <optional>
// #include <tilefirst-mlir/Dialect/Btfl/IR/BtflOps.h>
// #include <tilefirst-mlir/Dialect/TileFirst/IR/TfSchedulerDriver.h>
// #include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstAttributes.h>

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h>

// import custom directive
// using namespace mlir::tilefirst::detail::parsing;

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.cpp.inc"

using namespace mlir;
// using namespace mlir::tilefirst;
using namespace mlir::upmem;

void UPMEMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.cpp.inc"
      >();
}

// let assemblyFormat = "`<` `ranks` `(` $num_ranks `)` `,` `dpus` `(`
// $num_dpus_per_rank `)` `,` `tasklets` `(` $num_tasklets_per_dpu `)`
// custom<PopulateUpmemLevels>($wramLevel, $mramLevel, ref($num_ranks),
// ref($num_dpus_per_rank)) `>`";
static ParseResult parseNamedVar(AsmParser &p, llvm::StringLiteral name,
                                 cinm::CinmVarDefAttr &result) {
  if (p.parseKeyword(name) || p.parseLParen() ||
      p.parseCustomAttributeWithFallback(result) || p.parseRParen())
    return failure();
  return success();
}

Attribute UpmemAcceleratorAttr::parse(::mlir::AsmParser &p, ::mlir::Type) {
  cinm::CinmVarDefAttr ranks;
  cinm::CinmVarDefAttr dpus;
  cinm::CinmVarDefAttr tasklets;
  if (p.parseLess() || parseNamedVar(p, "ranks", ranks) || p.parseComma() ||
      parseNamedVar(p, "dpus", dpus) || p.parseComma() ||
      parseNamedVar(p, "tasklets", tasklets))
    return {};

  UpmemPlatformAttr platform = UpmemPlatformAttr::getDefault(p.getContext());
  if (p.parseOptionalComma().succeeded()) {
    if (p.parseCustomAttributeWithFallback(platform))
      return {};
  }

  return UpmemAcceleratorAttr::get(platform, ranks, dpus, tasklets);
}

static void printNamedVar(AsmPrinter &out, llvm::StringLiteral name,
                          cinm::CinmVarDefAttr var) {
  out << name << "(";
  out.printStrippedAttrOrType(var);
  out << ")";
}

void UpmemAcceleratorAttr::print(::mlir::AsmPrinter &out) const {
  out << "<";
  printNamedVar(out, "ranks", getNumRanks());
  out << ", ";
  printNamedVar(out, "dpus", getNumDpusPerRank());
  out << ", ";
  printNamedVar(out, "tasklets", getNumTaskletsPerDpu());
  out << ">";
}

Attribute UpmemPlatformAttr::parse(::mlir::AsmParser &p, ::mlir::Type) {
  SmallVector<int64_t> dims;
  if (p.parseLess() || p.parseKeyword("dimensions") || p.parseEqual() ||
      p.parseDimensionList(dims, false, false))
    return {};
  cinm::CinmLevelArrayAttr levels;
  if (p.parseOptionalComma().succeeded()) {
    if (p.parseKeyword("levels") || p.parseEqual() ||
        p.parseCustomAttributeWithFallback(levels))
      return {};
  } else {
    levels = UpmemPlatformAttr::getDefault(p.getContext()).getLevels();
  }
  if (p.parseGreater())
    return {};

  if (dims.size() != 2 && dims.size() != 3) {
    p.emitError(p.getNameLoc(), "Expected ranks x dpus (x tasklets)?, got ")
        << dims.size() << " dimensions";
    return {};
  }

  int ranks = dims[0], dpus = dims[1], tasklets = 32;
  if (dims.size() == 3)
    tasklets = dims[2];

  return UpmemPlatformAttr::get(p.getContext(), levels, ranks, dpus, tasklets);
}

void UpmemPlatformAttr::print(::mlir::AsmPrinter &out) const {
  out << "<dimensions = ";

  out.printDimensionList(
      {getMaxNumRanks(), getMaxNumDpusPerRank(), getMaxNumTasklets()});
  if (getLevels() != UpmemPlatformAttr::getDefault(getContext()).getLevels()) {
    out << ", levels = ";
    out.printStrippedAttrOrType(getLevels());
  }
  out << ">";
}

UpmemPlatformAttr UpmemPlatformAttr::getDefault(MLIRContext *ctx) {
  using namespace cinm;

  Builder builder(ctx);
  auto indices = 2;

  cinm::CinmLevelDefAttr mram =
      builder.getAttr<cinm::CinmLevelDefAttr>(builder.getStringAttr("mram"),
                                              /*size_in_bytes*/ 67108864,
                                              /*alignment*/ 8, indices);
  cinm::CinmLevelDefAttr wram =
      builder.getAttr<cinm::CinmLevelDefAttr>(builder.getStringAttr("wram"),
                                              /*size_in_bytes*/ 65536,
                                              /*alignment*/ 8, indices);

  return UpmemPlatformAttr::get(
      ctx, CinmLevelArrayAttr::get(builder.getContext(), {mram, wram}), 8, 64,
      16);
}

Attribute
UpmemPlatformAttr::getMemrefMemspace(cinm::CinmLevelDefAttr level) const {
  return level.getName();
}

cinm::CinmAcceleratorAttrInterface
UpmemAcceleratorAttr::instantiateDesignParams(
    const llvm::MapVector<StringRef, long> &instantiations) const {

  return UpmemAcceleratorAttr::get(
      getContext(), getImpl()->platform,
      cinm::detail::instantiateDesignParams(getImpl()->designParams,
                                            instantiations));
}

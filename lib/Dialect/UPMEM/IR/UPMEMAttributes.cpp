

#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
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

#include "cinm-mlir/Dialect/Cinm/IR/CinmTilingFactors.h"

using namespace mlir;
// using namespace mlir::tilefirst;
using namespace mlir::upmem;

void UPMEMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.cpp.inc"
      >();
}

Attribute UPMEMDialect::parseAttribute(DialectAsmParser &parser,
                                       Type type) const {
  if (parser.parseOptionalKeyword("wram").succeeded())
    return parser.getBuilder().getAttr<DpuMemSpaceAttr>(DpuMemSpace::WRAM);
  if (parser.parseOptionalKeyword("mram").succeeded())
    return parser.getBuilder().getAttr<DpuMemSpaceAttr>(DpuMemSpace::MRAM);

  StringRef mnemonic;
  Attribute result;
  auto res = generatedAttributeParser(parser, &mnemonic, type, result);
  if (res.has_value() && res.value().succeeded())
    return result;
  parser.emitError(parser.getNameLoc(), "Unknown attribute ") << mnemonic;
  return {};
}

void UPMEMDialect::printAttribute(Attribute attr,
                                  DialectAsmPrinter &out) const {
  if (auto a = llvm::dyn_cast_or_null<DpuMemSpaceAttr>(attr)) {
    out << stringifyDpuMemSpace(a.getValue());
    return;
  }
  (void)generatedAttributePrinter(attr, out);
}

void DpuMemSpaceAttr::print(AsmPrinter &out) const {
  out << stringifyEnum(getValue());
}

Attribute DpuMemSpaceAttr::parse(AsmParser &parser, Type) {
  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalKeyword("wram").succeeded())
    return parser.getBuilder().getAttr<DpuMemSpaceAttr>(DpuMemSpace::WRAM);
  if (parser.parseOptionalKeyword("mram").succeeded())
    return parser.getBuilder().getAttr<DpuMemSpaceAttr>(DpuMemSpace::MRAM);
  parser.emitError(loc, "Expected one of 'wram' or 'mram'");
  return {};
}

::llvm::StringRef UpmemPlatformAttr::getName() const { return "upmem"; }

Attribute UpmemAcceleratorAttr::parse(::mlir::AsmParser &p, ::mlir::Type) {
  if (p.parseLess())
    return {};
  SmallVector<int64_t> dims;
  if (p.parseDimensionList(dims, false, false) || dims.size() != 3) {
    return {};
  }

  // int64_t ranks;
  // int64_t dpus;
  // int64_t tasklets;
  // if (parseNamedVar(p, "ranks", ranks) || p.parseComma() ||
  //     parseNamedVar(p, "dpus", dpus) || p.parseComma() ||
  //     parseNamedVar(p, "tasklets", tasklets))
  //   return {};

  UpmemPlatformAttr platform = UpmemPlatformAttr::getDefault(p.getContext());
  if (p.parseOptionalComma().succeeded()) {
    if (p.parseCustomAttributeWithFallback(platform))
      return {};
  }
  if (p.parseGreater())
    return {};

  return UpmemAcceleratorAttr::getChecked(
      [&] { return p.emitError(p.getNameLoc()); }, p.getContext(), platform,
      SmallVector<int64_t>{dims[0], dims[1], dims[2]});
}

LogicalResult UpmemAcceleratorAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    UpmemPlatformAttr platform, ArrayRef<int64_t> workgroupShape) {
  if (workgroupShape.size() != 3)
    return emitError() << "expected a ranks x dpus x tasklets workgroup shape";
  // The DPU count is what the platform can actually run out of: an
  // allocation request beyond it fails at runtime, so reject it at compile
  // time. (Tasklet bounds are not checked yet: the rank/DPU split itself is
  // a historical artifact under repair, and existing specs disagree with
  // their platforms about tasklets.)
  const int64_t dpus = workgroupShape[0] * workgroupShape[1];
  const int64_t maxDpus =
      int64_t(platform.getMaxNumRanks()) * platform.getMaxNumDpusPerRank();
  if (dpus > maxDpus)
    return emitError() << "workgroup uses " << dpus
                       << " DPUs but the platform has only " << maxDpus;
  return success();
}

// static void printNamedVar(AsmPrinter &out, llvm::StringLiteral name,
//                           int64_t var) {
//   out << name << "(";
//   // out.printStrippedAttrOrType(var);
//   out << var;
//   out << ")";
// }

void UpmemAcceleratorAttr::print(::mlir::AsmPrinter &out) const {
  out << "<";
  out.printDimensionList(
      {getNumRanks(), getNumDpusPerRank(), getNumTaskletsPerDpu()});
  // printNamedVar(out, "ranks", getNumRanks());
  // out << ", ";
  // printNamedVar(out, "dpus", getNumDpusPerRank());
  // out << ", ";
  // printNamedVar(out, "tasklets", getNumTaskletsPerDpu());
  if (getPlatform() != UpmemPlatformAttr::getDefault(getContext())) {
    out << ", ";
    if (failed(out.printAlias(getPlatform())))
      out.printStrippedAttrOrType(getPlatform());
  }
  out << ">";
}

static cinm::CinmLevelArrayAttr upmemLevels(mlir::MLIRContext *ctx,
                                            bool isV1A) {
  int indices = 2;
  int wramSize = isV1A ? 65536 : 63488;
  // Don't let the whole wram be partitioned
  // for the application, some of it is required for data structures such as the
  // barrier. Let's estimate this as 8192 bytes
  wramSize -= 8192;

  Builder builder(ctx);
  cinm::CinmLevelDefAttr mram =
      builder.getAttr<cinm::CinmLevelDefAttr>(builder.getStringAttr("mram"),
                                              /*size_in_bytes*/ 67108864,
                                              /*alignment*/ 8, indices);
  cinm::CinmLevelDefAttr wram =
      builder.getAttr<cinm::CinmLevelDefAttr>(builder.getStringAttr("wram"),
                                              /*size_in_bytes*/ wramSize,
                                              /*alignment*/ 8, indices);
  return cinm::CinmLevelArrayAttr::get(builder.getContext(), {mram, wram});
}

Attribute UpmemPlatformAttr::parse(::mlir::AsmParser &p, ::mlir::Type) {
  SmallVector<int64_t> dims;
  if (p.parseLess() || p.parseKeyword("type") || p.parseEqual())
    return {};
  auto typeLoc = p.getCurrentLocation();
  llvm::FailureOr<bool> type =
      AsmParser::KeywordSwitch<llvm::FailureOr<bool>>(p)
          .Case("v1A", true)
          .Case("v1B", false)
          .Default(llvm::failure());
  if (llvm::failed(type)) {
    p.emitError(typeLoc, "expected one of v1A, v1B");
    return {};
  }
  bool isV1A = *type;

  if (p.parseComma() || p.parseKeyword("dimensions") || p.parseEqual() ||
      p.parseDimensionList(dims, false, false))
    return {};
  cinm::CinmLevelArrayAttr levels;
  if (p.parseOptionalComma().succeeded()) {
    if (p.parseKeyword("levels") || p.parseEqual() ||
        p.parseCustomAttributeWithFallback(levels))
      return {};
  } else {
    levels = upmemLevels(p.getContext(), isV1A);
  }
  if (p.parseGreater())
    return {};

  if (dims.size() != 2 && dims.size() != 3) {
    p.emitError(p.getNameLoc(), "Expected ranks x dpus (x tasklets)?, got ")
        << dims.size() << " dimensions";
    return {};
  }

  int ranks = dims[0], dpus = dims[1], tasklets = isV1A ? 24 : 16;
  if (dims.size() == 3)
    tasklets = dims[2];

  return UpmemPlatformAttr::get(p.getContext(), levels, isV1A, ranks, dpus,
                                tasklets);
}

void UpmemPlatformAttr::print(::mlir::AsmPrinter &out) const {
  out << "<type = " << (getIsV1a() ? "v1A" : "v1B") << ", dimensions = ";

  out.printDimensionList(
      {getMaxNumRanks(), getMaxNumDpusPerRank(), getMaxNumTasklets()});
  if (getLevels() != UpmemPlatformAttr::getDefault(getContext()).getLevels()) {
    out << ", levels = ";
    out.printStrippedAttrOrType(getLevels());
  }
  out << ">";
}

UpmemPlatformAttr UpmemPlatformAttr::getDefault(MLIRContext *ctx) {
  return UpmemPlatformAttr::get(ctx, upmemLevels(ctx, true), true, 8, 64, 24);
}

cinm::CinmLevelAttrInterface
UpmemPlatformAttr::getMemrefMemspace(cinm::CinmLevelDefAttr level) const {
  auto space = symbolizeDpuMemSpace(level.getName().getValue());
  if (!space)
    return {};
  return DpuMemSpaceAttr::get(getContext(), *space);
}

cinm::CinmAcceleratorAttrInterface
UpmemAcceleratorAttr::instantiateDesignParams(
    const llvm::MapVector<StringRef, long> &) const {
  return *this;

  // return UpmemAcceleratorAttr::get(
  //     getContext(), getImpl()->platform,
  //     cinm::detail::instantiateDesignParams(getImpl()->designParams,
  //                                           instantiations));
}
DiagnosedSilenceableFailure UpmemAcceleratorAttr::computeTilingFactors(
    Operation *op, SmallVectorImpl<int64_t> &tilingFactors) const {
  return cinm::computeTilingFactorsForOp(
      bufferSizeOfLeaf(), getWorkgroupShape(), op, tilingFactors);
}

::llvm::SmallVector<::mlir::cinm::CinmLevelArrayAttr>
UpmemAcceleratorAttr::getWorkgroupMemoryLevels() const {
  auto empty = cinm::CinmLevelArrayAttr::get(getContext(), {});
  return {empty,
          cinm::CinmLevelArrayAttr::get(getContext(),
                                        {getMramLevel(), getWramLevel()}),
          empty};
}

ArrayRef<cinm::CinmVarDefAttr> UpmemAcceleratorAttr::getDesignParams() const {
  return {};
}

int64_t UpmemAcceleratorAttr::bufferSizeOfLeaf() const {
  return getWramLevel().getSizeInBytes() / getNumTaskletsPerDpu();
}

bool UpmemPlatformAttr::isOffloadingTarget(Operation *op) const {
  if (isa<cinm::BatchGemmOp, cinm::BatchGemvOp, cinm::GemmOp, cinm::GemvOp,
          cinm::ReduceOp, cinm::ElementwiseOp>(op))
    return true;

  if (auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
    // A fill is just a broadcast of one value into the output, not worth
    // offloading.
    return !linalg::isaFillOpInterface(generic).has_value();
  }
  return false;
}

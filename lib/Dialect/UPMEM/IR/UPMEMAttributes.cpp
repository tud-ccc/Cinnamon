

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
#include <tilefirst-mlir/Dialect/Btfl/IR/BtflAttributes.h>
#include <tilefirst-mlir/Dialect/Btfl/IR/BtflOps.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TfSchedulerDriver.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstAttributes.h>

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h>

// import custom directive
using namespace mlir::tilefirst::detail::parsing;

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.cpp.inc"

using namespace mlir;
using namespace mlir::tilefirst;
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
                                 TfVarDefAttr &result) {
  if (p.parseKeyword(name) || p.parseLParen() ||
      p.parseCustomAttributeWithFallback(result) || p.parseRParen())
    return failure();
  return success();
}

Attribute UpmemAcceleratorAttr::parse(::mlir::AsmParser &p, ::mlir::Type) {
  TfVarDefAttr ranks;
  TfVarDefAttr dpus;
  TfVarDefAttr tasklets;
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
                          TfVarDefAttr var) {
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

UpmemPlatformAttr UpmemPlatformAttr::getDefault(MLIRContext *ctx) {
  using namespace tilefirst;

  Builder builder(ctx);
  auto indices = 2;

  TfLevelDefAttr mram =
      builder.getAttr<TfLevelDefAttr>(builder.getStringAttr("mram"),
                                      /*size_in_bytes*/ 67108864,
                                      /*alignment*/ 8, indices);
  TfLevelDefAttr wram =
      builder.getAttr<TfLevelDefAttr>(builder.getStringAttr("wram"),
                                      /*size_in_bytes*/ 65536,
                                      /*alignment*/ 8, indices);

  return UpmemPlatformAttr::get(
      ctx, TfLevelArrayAttr::get(builder.getContext(), {mram, wram}), 8, 64,
      16);
}

std::optional<TfVarDefAttr> UpmemAcceleratorAttr::getThreadCountVarDef() const {
  return getNumTaskletsPerDpu();
}

TfAcceleratorAttrInterface UpmemAcceleratorAttr::instantiateDesignParams(
    const llvm::MapVector<StringRef, long> &instantiations) const {

  return UpmemAcceleratorAttr::get(
      getContext(), getImpl()->platform,
      tilefirst::detail::instantiateDesignParams(getImpl()->designParams,
                                                 instantiations));
}

bool UpmemLaunchSchedulerAttr::isParallel() const { return true; }
std::optional<btfl::BtflLoopSchedulerAttrInterface>
UpmemLaunchSchedulerAttr::fuse(
    btfl::BtflLoopSchedulerAttrInterface other) const {
  if (other == *this || other.isBuiltin(btfl::BuiltinSchedulerKind::PARALLEL) ||
      other.isBuiltin(btfl::BuiltinSchedulerKind::HWPARALLEL))
    return *this;
  return std::nullopt;
}

DiagnosedSilenceableFailure
mlir::upmem::UpmemLaunchSchedulerAttr::lowerIntoKernel(
    mlir::RewriterBase &, mlir::btfl::TileOp const &loop,
    mlir::btfl::KernelOp const &) const {
  return emitSilenceableFailure(
      loop->getLoc(),
      "cannot be pushed into kernel, only supports top-down lowering");
}
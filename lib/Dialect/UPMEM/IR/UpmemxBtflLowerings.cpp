

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include <tilefirst-mlir/Dialect/TileFirst/IR/SchedulingSupport.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TfSchedulerDriver.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstAttributes.h>

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>

using namespace mlir;

using namespace mlir::tilefirst;
using namespace mlir::upmem;

TfMemSpaceAttr
UpmemAcceleratorAttr::getGenericMemSpace(TfLevelDefAttr level) const {
  if (level == getMramLevel() || level == getWramLevel()) {
    return level.toMemSpace(
        {getNumRanks().getIndexVar(), getNumDpusPerRank().getIndexVar()});
  }
  return {};
}

Attribute UpmemPlatformAttr::getMemrefMemspace(
    mlir::tilefirst::TfLevelDefAttr level) const {

  if (level.getName() == "mram") {
    return StringAttr::get(level.getContext(), "mram");
  } else if (level.getName() == "wram") {
    return StringAttr::get(level.getContext(), "wram");
  }
  return {};
}

Maybe<BufferMemspace>
upmem::getUpmemMemspace(Location loc, TfMemSpaceAttr memspace, bool allowHost) {
  auto name = memspace.getName();

  if (memspace.isHostMemspace() && allowHost)
    return BufferMemspace::HOST;
  else if (name == "wram")
    return BufferMemspace::WRAM;
  else if (name == "mram")
    return BufferMemspace::MRAM;

  return emitDefiniteFailure(loc, "Buffer is not part of the upmem platform")
         << memspace;
}

DiagnosedSilenceableFailure UpmemAcceleratorAttr::materializeBufferAllocation(
    ImplicitLocOpBuilder &builder, TfBufferType bufTy, MemRefType memrefType,
    Value &result) const {

  bool isWram =
      TRY_GET(getUpmemMemspace(builder.getLoc(), bufTy.getMemorySpace(),
                               false)) == BufferMemspace::WRAM;

  result = builder.create<upmem::StaticAllocOp>(memrefType, isWram);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure UpmemAcceleratorAttr::materializeTransfer(
    ImplicitLocOpBuilder &builder, TfBufferType sourceTy, TfBufferType destTy,
    TypedValue<MemRefType> sourceMemref,
    TypedValue<MemRefType> destMemRef) const {

  BufferMemspace sourceMs = TRY_GET(
      getUpmemMemspace(builder.getLoc(), sourceTy.getMemorySpace(), true));
  BufferMemspace destMs = TRY_GET(
      getUpmemMemspace(builder.getLoc(), destTy.getMemorySpace(), true));

  if (sourceMs == BufferMemspace::HOST || destMs == BufferMemspace::HOST) {
    // todo scatter/gather
    return emitDefiniteFailure(builder.getLoc(),
                               "TODO scatter/gather not implemented");
  } else if (sourceMs == destMs) {
    builder.create<memref::CopyOp>(sourceMemref, destMemRef);
    return DiagnosedSilenceableFailure::success();
  }

  builder.create<upmem::LocalTransferOp>(sourceMemref, destMemRef);

  return DiagnosedSilenceableFailure::success();
}
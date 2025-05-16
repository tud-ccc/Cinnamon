

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/InliningUtils.h>

#include <tilefirst-mlir/Dialect/Btfl/IR/BtflInterfaces.h>
#include <tilefirst-mlir/Dialect/Btfl/IR/BtflOps.h>
#include <tilefirst-mlir/Dialect/Btfl/IR/BtflTopDownLoweringContext.h>
#include <tilefirst-mlir/Dialect/Threads/IR/ThreadsDialect.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/SchedulingSupport.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TfSchedulerDriver.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstAttributes.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstOps.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstTypes.h>

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Utils.h>

#define DEBUG_TYPE "upmem"

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

static DiagnosedSilenceableFailure
createBuffersInDpuProgram(btfl::TopDownLoweringContext &ctx,
                          upmem::DpuProgramOp dpuProgram,
                          btfl::KernelOp dpuBtflKernel, btfl::ScopeOp scope,
                          llvm::SmallVectorImpl<Value> &kernelArgReplacements) {

  auto &rewriter = ctx.rewriter;
  OpBuilder::InsertionGuard _(rewriter);

  IRMapping oldBlockArgsToNewAllocs;

  rewriter.setInsertionPointToStart(&dpuProgram.getBody().front());
  llvm::SmallString<20> nextName("buf");
  const int resetLen = nextName.size();
  int nextNameNum = 0;
  for (auto [outer, inner] : scope.getOperandsWithBlockArgs()) {
    if (!inner.hasOneUse() || *inner.getUsers().begin() != dpuBtflKernel)
      return emitSilenceableFailure(
          inner.getLoc(), "Should be used exactly once by the kernel");

    auto upmemMemspace = TRY_GET(upmem::getUpmemMemspace(
        outer.getLoc(),
        llvm::dyn_cast<tilefirst::TfBufferType>(outer.getType())
            .getMemorySpace(),
        false));

    auto kernelOpNum = inner.getUses().begin()->getOperandNumber();
    auto memrefTy = llvm::cast<MemRefType>(
        dpuBtflKernel.getKernelImplSig().getInput(kernelOpNum));

    nextName.append(std::to_string(nextNameNum++));
    StringAttr symbolName = rewriter.getStringAttr(nextName);
    nextName.truncate(resetLen);

    // MemRefType structuredType, bool isWram, StringRef sym_name = {}, bool
    // noinit = false);
    auto alloc = rewriter.create<upmem::StaticAllocOp>(
        outer.getLoc(), memrefTy, upmemMemspace == upmem::BufferMemspace::WRAM,
        symbolName);

    oldBlockArgsToNewAllocs.map(dpuBtflKernel.getBufferBlockArgs()[kernelOpNum],
                                alloc.getBuffer());
    // record this to be able to use it in transfers
    ctx.recordValueToSymbolMapping(outer, alloc);
  }

  // compute the replacement list for the block arguments
  for (auto [outer, inner] : dpuBtflKernel.getOperandsWithBlockArgs()) {
    kernelArgReplacements.push_back(oldBlockArgsToNewAllocs.lookup(inner));
  }

  return DiagnosedSilenceableFailure::success();
}

tilefirst::Maybe<upmem::DeviceHierarchyType>
toWgType(UpmemAcceleratorAttr upmem, Location loc) {

  auto ranks = upmem.getNumRanks().asInt();
  auto dpus = upmem.getNumDpusPerRank().asInt();
  auto tasklets = upmem.getNumTaskletsPerDpu().asInt();
  if (ranks && dpus && tasklets) {
    return upmem::DeviceHierarchyType::get(upmem.getContext(), *ranks, *dpus,
                                           *tasklets);
  }
  return emitDefiniteFailure(loc,
                             "worgroup design params are not fully resolved");
}

DiagnosedSilenceableFailure
UpmemAcceleratorAttr::lowerAcceleratorOp(btfl::TopDownLoweringContext &ctx,
                                         tilefirst::AcceleratorOp acc) const {

  // We start from an accelerator that has a scope (assumed unique),
  // and we turn the buffers inputs/outputs and the relevant transfers into.
  // For all arguments of the scope:
  // - Push them into the kernel as a static_alloc (with a buffer ID)
  // - Associate the value (outside scope) with that ID, use that to turn the
  // transfers host<->dpu into upmem.scatter/gather.

  // For scatter/gather I need a scatter map (r,d) -> (dimensions...). The
  // scatter map needs to use values for (r,d) that are found in the enclosing
  // loops. This means we either need to lower all enclosing loops
  // simultaneously or we have to

  if (!acc.getResult().hasOneUse())
    return emitSilenceableFailure(
        acc->getLoc(), "Not implemented, accelerator has multiple uses");

  Operation *user = *acc.getResult().getUsers().begin();
  auto scope = llvm::dyn_cast_or_null<btfl::ScopeOp>(user);
  if (!scope)
    return emitSilenceableFailure(user->getLoc(),
                                  "Not implemented, user is not a scope");

  auto dpuBtflKernel =
      llvm::dyn_cast_or_null<btfl::KernelOp>(&scope.getBodyBlock()->front());
  if (!dpuBtflKernel ||
      !llvm::isa<btfl::YieldOp>(dpuBtflKernel->getNextNode())) {
    return emitSilenceableFailure(
        scope->getLoc(), "Scope requires single child that is a btfl.kernel");
  }

  auto &rewriter = ctx.rewriter;
  upmem::DpuProgramOp dpuProgram;
  ModuleOp module = acc->getParentOfType<ModuleOp>();
  SymbolTable modSymbols(module);
  {
    OpBuilder::InsertionGuard _(rewriter);

    // get or create the dpu_kernels module, insert the new upmem.dpu_program in
    // there.

    auto kernelsMod = modSymbols.lookup<ModuleOp>("dpu_kernels");
    if (!kernelsMod) {
      rewriter.setInsertionPointToEnd(&module.getBodyRegion().front());
      kernelsMod = rewriter.create<ModuleOp>(
          scope->getLoc(), rewriter.getStringAttr("dpu_kernels"));
    }

    auto numThreadsAttr =
        scope->getAttrOfType<IntegerAttr>(threads::ThreadCountAttr::name);
    int numThreads = numThreadsAttr ? numThreadsAttr.getInt() : 1;
    rewriter.setInsertionPointToStart(&kernelsMod.getBodyRegion().front());
    dpuProgram = rewriter.create<upmem::DpuProgramOp>(scope->getLoc(), "prog",
                                                      numThreads);
    dpuProgram.getBody().emplaceBlock();

    // Create named buffers in the upmem.dpu_program to replace the arguments of
    // the scope/kernel. This also adds the remappings Value->Symbol to the
    // lowering context.
    SmallVector<Value> kernelArgReplacements;
    TRY(createBuffersInDpuProgram(ctx, dpuProgram, dpuBtflKernel, scope,
                                  kernelArgReplacements));

    rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
    auto terminator = rewriter.create<upmem::ReturnOp>(scope->getLoc());
    // erase terminator so that it is not inlined with the rest
    rewriter.eraseOp(dpuBtflKernel.getBodyBlock()->getTerminator());

    // Inline the body of the kernel into the new dpu_program, replacing kernel
    // arguments with the new allocations.
    rewriter.inlineBlockBefore(dpuBtflKernel.getBodyBlock(), terminator,
                               kernelArgReplacements);
    rewriter.eraseOp(dpuBtflKernel);
  }

  // Finally create the alloc_dpus op, and remap the old !tilefirst.accelerator
  // value, so that it can be accessed at transfer sites for instance.

  auto wgType = TRY_GET(toWgType(*this, acc->getLoc()));
  auto dpuProgRef = upmem::getSymbolPath(modSymbols, dpuProgram);

  auto allocOp =
      rewriter.create<upmem::AllocDPUsOp>(acc->getLoc(), wgType, *dpuProgRef);

  ctx.recordValueMapping(acc.getResult(), allocOp);
  LLVM_DEBUG(llvm::errs() << "[upmem] Created op " << dpuProgram << "\n");

  return DiagnosedSilenceableFailure::success();
}

mlir::DiagnosedSilenceableFailure
UpmemAcceleratorAttr::lowerOpTopDown(::mlir::btfl::TopDownLoweringContext &ctx,
                                     ::mlir::Operation *op) const {
  if (!cast<TfAcceleratorAttrInterface>(*this).ownsOperation(op))
    return btfl::detail::defaultLowerOpTopDown(ctx, op);
  auto acc = ctx.getAcceleratorOp(*this);
  auto wgValue = ctx.getMappedValue(acc.getResult());

  return llvm::TypeSwitch<Operation *, DiagnosedSilenceableFailure>(op)
      .Case([&](btfl::ScopeOp scope) {
        ctx.rewriter.create<upmem::WaitForOp>(scope->getLoc(), wgValue);
        return DiagnosedSilenceableFailure::success();
      })
      .Default([&](Operation *op) {
        return btfl::detail::defaultLowerOpTopDown(ctx, op);
      });
}
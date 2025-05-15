#include "tilefirst-mlir/Dialect/Btfl/IR/BtflAttributes.h"
#include "tilefirst-mlir/Dialect/Btfl/IR/BtflOps.h"
#include "tilefirst-mlir/Dialect/TileFirst/IR/SchedulingSupport.h"
#include "tilefirst-mlir/Dialect/TileFirst/IR/TfScheduleSpec.h"
#include "tilefirst-mlir/Dialect/TileFirst/IR/TfSchedulerDriver.h"
#include "tilefirst-mlir/Dialect/TileFirst/IR/TileFirstAttributesBase.h"
#include "tilefirst-mlir/Dialect/TileFirst/IR/TileFirstOps.h"
#include "tilefirst-mlir/Dialect/TileFirst/IR/TileFirstTypes.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/LocationSnapshot.h>

#include <optional>
#include <tilefirst-mlir/Dialect/Btfl/Transforms/BtflCompositeTransforms.h>
#include <tilefirst-mlir/Dialect/Btfl/Transforms/BtflSchedulingApi.h>
#include <tilefirst-mlir/Dialect/Btfl/Transforms/BtflSchedulingTransforms.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/SymExpr.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TfInferenceContext.h>
#include <tilefirst-mlir/Dialect/TileFirst/IR/TileFirstAttributes.h>

#include <tilefirst-mlir/Dialect/Threads/IR/ThreadsDialect.h>

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>

#define DEBUG_TYPE "upmemx"

using namespace mlir;
using namespace mlir::btfl::scheduling;

static DiagnosedSilenceableFailure
coarsenLastDim(btfl::ScheduleOp schedule, BtflPlatformApi &api,
               SymExpr coarseningFactor, btfl::TilingOptions options = {}) {

  auto dim = schedule.getScheduleSpec().getKdims();
  if (dim.empty())
    return DiagnosedSilenceableFailure::success();

  int dimIx = dim.size() - 1;
  TRY(api.coarsenDim(schedule, dimIx, coarseningFactor, options));
  (void)api.removeRedundantDims(schedule);

  return DiagnosedSilenceableFailure::success();
}

// using namespace upmem;

mlir::DiagnosedSilenceableFailure
performSmartSchedulingForUpmem(RewriterBase &rewriter, btfl::BlockOp blockOp,
                               upmem::UpmemAcceleratorAttr upmem) {
  BtflPlatformApi api(blockOp, rewriter);

  SymVariableExpr blockingFactor = api.createFreshVariable("Q");
  SymVariableExpr taskletCount = upmem.getNumTaskletsPerDpu().getBoundVar();
  TRY(performInitialParallelismDetection(
      blockOp, rewriter, blockingFactor, taskletCount,
      rewriter.getAttr<threads::BtflThreadsSchedulerAttr>()));

  // Todo we could solve in different waves depending on liveness.
  // Design parameters like tasklets need to be solved globally though.
  TRY(api.solveGreedily({taskletCount, blockingFactor}));

  // dim for the mram loop
  SymVariableExpr mmFactor = api.createFreshVariable("M");

  for (auto schedule : collect<btfl::ScheduleOp>(blockOp)) {
    auto wram =
        TRY_GET(api.getMemSpaceInstance(upmem.getWramLevel(), schedule));
    auto mram = TRY_GET(api.getTransferMemSpace(upmem.getMramLevel(), wram));

    TRY(api.transferFromInside(schedule, mram, wram));
    // scheduling::debugPrintBufferSize(schedule);
    if (succeeded(api.removeRedundantDims(schedule, true)))
      continue;

    TRY(coarsenLastDim(
        schedule, api, mmFactor,
        btfl::TilingOptions{.scheduler = api.getBuiltinSeqScheduler()}));
  }

  TRY(api.solveGreedily({mmFactor}));

  for (auto schedule : collect<btfl::ScheduleOp>(blockOp)) {
    auto mram =
        TRY_GET(api.getMemSpaceInstance(upmem.getMramLevel(), schedule));

    TRY(api.transferFromInside(schedule, api.getHostMemSpace(), mram));
    // scheduling::debugPrintBufferSize(schedule);

    (void)api.removeRedundantDims(schedule, true);
  }

  (void)generateLocationsFromIR("", blockOp, OpPrintingFlags{});

  for (auto kernel : collect<btfl::KernelOp>(blockOp)) {
    (void)runKernelCleanupPipeline(kernel);
  }

  TRY(api.performBufferReuseAnalysis());
  TRY(api.performConstantPropagation());

  return emitSilenceableFailure(blockOp, "Should not make progress anymore");
}

static TfVarDefAttr getVarDef(NameInventor &namer, llvm::StringRef ixHint,
                              llvm::StringRef boundHint, long upperBound) {
  auto ixName = namer.getUniqueName(ixHint);
  auto boundsName = namer.getUniqueName(boundHint);
  return TfVarDefAttr::get(ixName, boundsName, 1, upperBound);
}
static upmem::UpmemAcceleratorAttr
getAccelerator(upmem::UpmemPlatformAttr platform, btfl::BlockOp block,
               long maxRanks, long maxDpus, long maxTasklets) {
  NameInventor namer = NameInventor::getNameInventor(block, "");
  return upmem::UpmemAcceleratorAttr::get(
      platform, getVarDef(namer, "r", "R", maxRanks),
      getVarDef(namer, "d", "D", maxDpus),
      getVarDef(namer, "t", "T", maxTasklets));
}

// static llvm::LogicalResult
// insertTemplateForSingleDpu(UpmemPlatformAttr platform, RewriterBase
// &rewriter,
//                            btfl::BlockOp block, btfl::TileOp parallelLoop) {

//   rewriter.setInsertionPointToStart(&block.getBodyBlock());

//   // An accelerator that uses a single DPU tile, but may use several threads.
//   auto upmemAccelerator =
//       getAccelerator(platform, block, 1, 1, platform.getMaxNumTasklets());

//   rewriter.create<tilefirst::AcceleratorOp>(parallelLoop->getLoc(),
//                                             upmemAccelerator);

//   // start scheduling
//   BtflPlatformApi api(block, rewriter);
//   SymVariableExpr blockingFactor = api.createFreshVariable("Q");
//   SymVariableExpr taskletCount =
//       upmemAccelerator.getNumTaskletsPerDpu().getBoundVar();
//   TRY_REPORT(performInitialParallelismDetection(
//       block, rewriter, blockingFactor, taskletCount,
//       rewriter.template getAttr<btfl::BuiltinLoopSchedulerAttr>(
//           btfl::BuiltinSchedulerKind::THREADED)));

//   // Todo we could solve in different waves depending on liveness.
//   // Design parameters like tasklets need to be solved globally though.
//   TRY_REPORT(api.solveGreedily({taskletCount, blockingFactor}));

//   parallelLoop.setSchedulerAttr(
//       rewriter.getAttr<btfl::BuiltinLoopSchedulerAttr>(
//           btfl::BuiltinSchedulerKind::THREADED));
//   parallelLoop.setTemporalSymbolAttr(
//       SymExprAttr::get(upmemAccelerator.getNumTaskletsPerDpu().getIndexVar()));

//   // todo set memory spaces

//   return llvm::success();
// }

// static btfl::TileOp findFirstParallelLoop(btfl::BlockOp block) {
//   btfl::TileOp targetLoop;
//   block.walk([&](btfl::TileOp op) {
//     if (auto builtin =
//     llvm::dyn_cast_or_null<btfl::BuiltinLoopSchedulerAttr>(
//             op.getScheduler())) {
//       // todo this scheduler is the one used by expose_parallelism. It should
//       be
//       // something else, hwparallel is not specific enough.
//       if (builtin.getKind() == btfl::BuiltinSchedulerKind::HWPARALLEL) {
//         targetLoop = op;
//         return WalkResult::interrupt();
//       }
//     }
//     return WalkResult::advance();
//   });
//   return targetLoop;
// }

static bool hasFinalSize(const TfScheduleSpec &spec, Type genericTy) {
  if (auto bufTy = llvm::dyn_cast_or_null<TfBufferType>(genericTy)) {
    return !bufTy.getTileShape().anyVar([&](SymVariableExpr var) {
      // return whether this variable will make the buffer grow. It is assumed
      // that if the dim goes up, the buffer grows (this might not be true if
      // the var appears in divisor position for instance).
      if (auto kdim = spec.getKdim(var.getId())) {
        return kdim->pdimTarget != kdim->kdimValue;
      }
      return false;
    });
  }
  return true;
}

static DiagnosedSilenceableFailure
makeProgressOnMemspaceAssignment(btfl::BlockOp block) {
  DiagnosedSilenceableFailure failure = DiagnosedSilenceableFailure::success();
  block->walk([&](Operation *op) -> WalkResult {
    if (auto schedule = llvm::dyn_cast_or_null<btfl::ScheduleOp>(op)) {
      const TfScheduleSpec &spec = schedule.getScheduleSpec();
      for (auto [outer, genTy, inner] : schedule.getBuffersWithGenericTypes()) {
        auto outerTy = llvm::dyn_cast_or_null<TfBufferType>(outer.getType());
        auto innerTy = llvm::dyn_cast_or_null<TfBufferType>(inner.getType());
        if (outerTy && outerTy.getMemorySpace().getName() == "wram") {
          // if the outer space is wram then the inner space must be wram
          TRY_IN_WALK(failure,
                      btfl::setMemorySpace(inner, outerTy.getMemorySpace()));
        } else if (innerTy && hasFinalSize(spec, genTy) &&
                   innerTy.getMemorySpace().getName() == "wram") {
          // If the inner space is wram, and this buffer may not grow more
          // through coarsening of this schedule, then we may place the outer
          // buffer also in wram to cut down on transfer times.

          TRY_IN_WALK(failure,
                      btfl::setMemorySpace(outer, innerTy.getMemorySpace()));
        }
      }
    }
    return WalkResult::advance();
  });
  return failure;
}

static DiagnosedSilenceableFailure
moveMramToWram(btfl::BlockOp blockOp, RewriterBase &rewriter,
               upmem::UpmemAcceleratorAttr upmem) {

  BtflPlatformApi api(blockOp, rewriter);

  // dim for the mram loop
  SymVariableExpr mmFactor = api.createFreshVariable("M");

  for (auto schedule : collect<btfl::ScheduleOp>(blockOp)) {
    auto wram =
        TRY_GET(api.getMemSpaceInstance(upmem.getWramLevel(), schedule));
    auto mram = TRY_GET(api.getTransferMemSpace(upmem.getMramLevel(), wram));

    TRY(api.transferFromInside(schedule, mram, wram));
    // scheduling::debugPrintBufferSize(schedule);
    if (succeeded(api.removeRedundantDims(schedule, true)))
      continue;

    TRY(coarsenLastDim(
        schedule, api, mmFactor,
        btfl::TilingOptions{.scheduler = api.getBuiltinSeqScheduler()}));
  }

  TRY(api.solveGreedily({mmFactor}));

  return DiagnosedSilenceableFailure::success();
}

static llvm::LogicalResult
insertTemplateForSingleDpu(upmem::UpmemPlatformAttr platform,
                           RewriterBase &rewriter, btfl::BlockOp block) {

  rewriter.setInsertionPointToStart(&block.getBodyBlock());

  // An accelerator that uses a single DPU tile, but may use several threads.
  auto upmemAccelerator =
      getAccelerator(platform, block, 1, 1, platform.getMaxNumTasklets());

  rewriter.create<tilefirst::AcceleratorOp>(block->getLoc(), upmemAccelerator);

  // start scheduling
  BtflPlatformApi api(block, rewriter);
  SymVariableExpr blockingFactor = api.createFreshVariable("Q");
  SymVariableExpr taskletCount =
      upmemAccelerator.getNumTaskletsPerDpu().getBoundVar();
  SymVariableExpr mmFactor = api.createFreshVariable("M");
  auto threadsScheduler = rewriter.getAttr<threads::BtflThreadsSchedulerAttr>();
  TRY_REPORT(performInitialParallelismDetection(
      block, rewriter, blockingFactor, taskletCount, threadsScheduler));

  auto wram =
      upmemAccelerator.getGenericMemSpace(upmemAccelerator.getWramLevel());
  auto mram =
      upmemAccelerator.getGenericMemSpace(upmemAccelerator.getMramLevel());

  // Set all the inputs and buffers used inside of threaded tile loops as in
  // WRAM. This sets constraints on the greedy solving.
  TRY_REPORT(api.setMemorySpaceForScheduler(threadsScheduler, wram));
  TRY_REPORT(makeProgressOnMemspaceAssignment(block));

  // Todo we could solve in different waves depending on liveness.
  // Design parameters like tasklets need to be solved globally though.
  TRY_REPORT(api.solveGreedily({taskletCount, blockingFactor}));

  // Transfer host buffers to MRAM. Since we're doing everything on a single DPU
  // we want the inputs/outputs to be in MRAM.
  // TODO this assumes that the full host buffers fit in MRAM. We should
  // actually tile
  //  down and resolve the tile size
  auto hostMemspace = api.getHostMemSpace();

  for (auto schedule : collect<btfl::ScheduleOp>(block)) {
    TRY_REPORT(api.transferFromInside(schedule, mram, wram));
    TRY_REPORT(api.transferFromOutside(schedule, hostMemspace, mram));
    // todo do coarsening from mram to wram.
    // In the simple case of softmax we may just pick any dimension to coarsen.

    if (succeeded(api.removeRedundantDims(schedule, true)))
      continue;

    TRY_REPORT(coarsenLastDim(
        schedule, api, mmFactor,
        btfl::TilingOptions{.scheduler = api.getBuiltinSeqScheduler()}));
  }
  TRY_REPORT(api.performBufferReuseAnalysis());
  TRY_REPORT(api.performConstantPropagation());

  TRY_REPORT(api.solveGreedily({mmFactor}));

  for (auto schedule : collect<btfl::ScheduleOp>(block)) {
    if (succeeded(api.removeRedundantDims(schedule, true)))
      continue;
  }

  return llvm::success();
}

void upmem::UpmemPlatformAttr::collectSchedulingStartingPoints(
    TfSchedulingStarter &driver) const {

  // There are several hwparallel loops.
  // We want to pick one and spawn multiple alternatives from that one.

  driver.forkAlternativeIf<btfl::BlockOp>([&](auto &rewriter, auto block) {
    return insertTemplateForSingleDpu(*this, rewriter, block);
  });
}

static ::mlir::DiagnosedSilenceableFailure
tileDownLocalTransfer(RewriterBase &rewriter, btfl::TransferOp transfer,
                      bool &change) {
  auto source = TRY_GET(upmem::getUpmemMemspace(
      transfer->getLoc(), transfer.getSourceMemorySpace(), true));
  auto dest = TRY_GET(upmem::getUpmemMemspace(
      transfer->getLoc(), transfer.getDestMemorySpace(), true));

  if (source != upmem::BufferMemspace::HOST &&
      dest != upmem::BufferMemspace::HOST) {
    // this is a local transfer
    if (auto numBytes = asConstant(transfer.getNumBytesTransferred())) {
      if (*numBytes > 2048 && *numBytes % 2048 == 0) {
        auto factor = *numBytes / 2048;
        for (auto [i, dim] :
             llvm::enumerate(toTensorDimensions(transfer.getTileShape()))) {
          if (factor < dim && dim % factor == 0) {
            btfl::TileOp result;
            change = true;
            return transfer.tileDown(
                rewriter, i, getSymConstantExpr(factor, transfer->getContext()),
                result, btfl::TilingOptions{});
          }
        }
      }
    }
  }
  return DiagnosedSilenceableFailure::success();
}

::mlir::DiagnosedSilenceableFailure
upmem::UpmemAcceleratorAttr::makeSchedulingProgress(
    TfAcceleratorScheduler &api) const {

  auto block = llvm::dyn_cast_or_null<btfl::BlockOp>(api.blockOp());
  if (!block) {
    return emitDefiniteFailure(api.getLoc(), "Not a btfl.block");
  }

  // todo this is a placeholder
  // auto res = performSmartSchedulingForUpmem(api.rewriter, block, *this);
  TRY(makeProgressOnMemspaceAssignment(block));

  // bool hasScheduleOpsLeft = false;
  for (auto schedule : collect<btfl::ScheduleOp>(block)) {
    // hasScheduleOpsLeft |=
    llvm::failed(btfl::removeRedundantDims(api.rewriter, schedule, true));
  }

  bool tiledAnyTransfer = false;
  for (auto transfer : collect<btfl::TransferOp>(block)) {
    TRY(tileDownLocalTransfer(api.rewriter, transfer, tiledAnyTransfer));
  }
  if (tiledAnyTransfer) {
    btfl::FusionOptions fusionOptions;
    btfl::doGreedyFusion(api.rewriter, block, fusionOptions);
  }

  api.markSchedulingDone();

  return DiagnosedSilenceableFailure::success();
}
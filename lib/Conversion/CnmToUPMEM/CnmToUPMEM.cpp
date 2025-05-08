#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <cinm-mlir/Dialect/UPMEM/Transforms/Utils.h>
#include <cstdint>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#define GEN_PASS_DEF_CONVERTCNMTOUPMEMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace mlir::cnm {
namespace {

template <typename T> T reduceMul(ArrayRef<T> arr) {
  T result{1};
  for (const T &elem : arr) {
    result *= elem;
  }
  return result;
}

MemRefType convertTensorToMemref(ShapedType ty) {
  if (isa<MemRefType>(ty))
    return cast<MemRefType>(ty);

  return MemRefType::get(ty.getShape(), ty.getElementType());
}

inline static constexpr size_t alignTo(size_t v, size_t alignment) {
  return v % alignment == 0 ? v : v + alignment - v % alignment;
}

// In CNM the affine map has 1 dim for rank, 1 for dpu, 1 for tasklet.
// In upmem it has only one dim for rank and another for dpu. Dimensions
// of the buffer shape are zero (they are the offset of the buffer start).
static AffineMap adaptAffineMapCnmToUpmem(AffineMap map,
                                          cnm::BufferType bufTy) {
  assert(map.getNumDims() == 3);
  auto rankDim = getAffineDimExpr(0, map.getContext());
  auto dpuDim = getAffineDimExpr(1, map.getContext());
  auto cst0 = getAffineConstantExpr(0, map.getContext());
  SmallVector<AffineExpr> exprs;
  for (auto e : map.getResults()) {
    exprs.push_back(e.replaceDims({rankDim, dpuDim, cst0}));
  }
  for (auto _ : bufTy.getShape()) {
    exprs.push_back(cst0);
  }
  return AffineMap::get(2, 0, std::move(exprs), map.getContext());
}

static LogicalResult convertCnmGatherToUpmem(RewriterBase &rewriter,
                                             cnm::GatherOp op,
                                             upmem::AllocDPUsOp upmemWgAlloc,
                                             SymbolRefAttr refToBuffer) {

  rewriter.setInsertionPoint(op);
  Value outputBuf = op.getOutputBuf();
  bool isBufferized = isa<BaseMemRefType>(op.getOutputBuf().getType());
  if (!isBufferized) {
    outputBuf = rewriter.create<memref::AllocOp>(
        op->getLoc(), convertTensorToMemref(op.getOutputBuf().getType()));
  }

  const size_t numTasklets = upmemWgAlloc.getType().getNumTaskletsPerDpu();
  const int64_t transferCount = op.getTransferCountInItems() * numTasklets;

  rewriter.create<upmem::GatherOp>(
      op->getLoc(), outputBuf, refToBuffer, transferCount,
      adaptAffineMapCnmToUpmem(op.getGatherMap(), op.getBuffer().getType()),
      upmemWgAlloc.getResult());

  if (!isBufferized) {
    Value outputAsTensor = createOrFoldUnrealizedConversionCast(
        op->getLoc(), rewriter, op.getOutput().getType(), outputBuf);

    rewriter.replaceAllUsesWith(op.getOutput(), outputAsTensor);
  }
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult convertCnmScatterToUpmem(RewriterBase &rewriter,
                                              cnm::ScatterOp op,
                                              upmem::AllocDPUsOp upmemWgAlloc,
                                              SymbolRefAttr refToBuffer) {

  rewriter.setInsertionPoint(op);
  const Value tensor = op.getInput();
  const ShapedType inputTy = op.getInput().getType();

  const Value inputAsMemref = createOrFoldUnrealizedConversionCast(
      op.getLoc(), rewriter, convertTensorToMemref(inputTy), tensor);

  const size_t numTasklets = upmemWgAlloc.getType().getNumTaskletsPerDpu();
  const int64_t transferCount = op.getTransferCountInItems() * numTasklets;

  rewriter.create<upmem::ScatterOp>(
      op->getLoc(), inputAsMemref, refToBuffer, transferCount,
      adaptAffineMapCnmToUpmem(op.getScatterMap(), op.getBuffer().getType()),
      upmemWgAlloc.getResult());

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult convertCnmLaunchToUpmem(cnm::LaunchOp launch,
                                             RewriterBase &rewriter,
                                             SymbolTable rootModule,
                                             ModuleOp dpuKernelModule) {

  rewriter.clearInsertionPoint();

  auto wg = launch.getWg().getType().getShape();
  if (wg.size() != 3)
    return launch.emitOpError("Should have working group with 3 entries");

  const auto upmemTy =
      rewriter.getType<upmem::DeviceHierarchyType>(wg[0], wg[1], wg[2]);

  auto dpuProgram = rewriter.create<upmem::DpuProgramOp>(
      launch->getLoc(), "program", upmemTy.getNumTaskletsPerDpu());
  dpuProgram.getBody().emplaceBlock();
  SymbolTable symTable(dpuKernelModule);
  symTable.insert(dpuProgram);

  auto programPath = upmem::getSymbolPath(rootModule, dpuProgram);
  assert(llvm::succeeded(programPath));

  auto wgAlloc = cast<cnm::WorkgroupOp>(launch.getWg().getDefiningOp());
  rewriter.setInsertionPoint(wgAlloc);
  auto upmemWgAlloc = rewriter.create<upmem::AllocDPUsOp>(
      wgAlloc->getLoc(), upmemTy, *programPath);

  llvm::MapVector<Value, upmem::StaticAllocOp> buffersToMramBuf;
  llvm::MapVector<Value, upmem::PrivateWRAMAllocOp> buffersToPwramBuf;

  rewriter.setInsertionPointToStart(&dpuProgram.getBody().front());

  // create named static MRAM buffers for each cnm.alloc operation, put them in
  // the dpu program
  SmallVector<AllocOp> allocsToDelete;

  SymbolTable dpuProgramSymTable(dpuProgram);
  for (auto user : launch.getWg().getUsers()) {
    if (auto alloc = llvm::dyn_cast_or_null<cnm::AllocOp>(user)) {
      allocsToDelete.push_back(alloc);

      auto bufferType = alloc.getType();

      SmallVector<int64_t> bufShape(bufferType.getShape());

      // the pwram buffer has the shape we expect
      MemRefType memrefTy =
          MemRefType::get(bufShape, bufferType.getElementType(),
                          MemRefLayoutAttrInterface{}, bufferType.getLevel());
      auto pwramBuf =
          rewriter.create<upmem::PrivateWRAMAllocOp>(alloc.getLoc(), memrefTy);

      buffersToPwramBuf[alloc.getResult()] = pwramBuf;

      // the mram buffer type has tasklet dimension prepended.
      bufShape.insert(bufShape.begin(), upmemTy.getNumTaskletsPerDpu());

      memrefTy =
          MemRefType::get(bufShape, bufferType.getElementType(),
                          MemRefLayoutAttrInterface{}, bufferType.getLevel());

      auto mrambuf = rewriter.create<upmem::StaticAllocOp>(
          alloc->getLoc(), memrefTy, false, false,
          rewriter.getStringAttr("buf"));
      dpuProgramSymTable.insert(mrambuf); // this renames it to a unique name
      buffersToMramBuf[alloc.getResult()] = mrambuf;
    }
  }

  // then, replace all scatter/gather with the upmem equivalents

  for (auto user : launch.getWg().getUsers()) {

    if (auto scatter = llvm::dyn_cast_or_null<cnm::ScatterOp>(user)) {
      auto alloc = buffersToMramBuf[scatter.getBuffer()];
      if (alloc) {
        auto ref = upmem::getSymbolPath(
            SymbolTable::getNearestSymbolTable(scatter), alloc);
        if (succeeded(ref)) {
          if (failed(convertCnmScatterToUpmem(rewriter, scatter, upmemWgAlloc,
                                              *ref)))
            return failure();
          continue;
        }
      }
    }
    if (auto gather = llvm::dyn_cast_or_null<cnm::GatherOp>(user)) {
      auto alloc = buffersToMramBuf[gather.getBuffer()];
      if (alloc) {
        auto ref = upmem::getSymbolPath(
            SymbolTable::getNearestSymbolTable(gather), alloc);
        if (succeeded(ref)) {
          if (failed(convertCnmGatherToUpmem(rewriter, gather, upmemWgAlloc,
                                             *ref)))
            return failure();
          continue;
        }
      }
    }
  }

  // At this point we have replaced (and deleted) the scatter and gather.
  // We still need to move the body of the launch into the new DPU program,
  // and delete all remaining CNM ops.

  // these memrefs map to the pwram bufs. TODO we need to transfer from mram to
  // pwram
  IRMapping mapping;
  for (auto [cnmBuf, memref] : llvm::zip_equal(
           llvm::concat<Value>(launch.getInputs(), launch.getOutBuffers()),
           launch.getBody().getArguments())) {

    mapping.map(memref, buffersToPwramBuf[cnmBuf].getResult());
  }

  rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  for (auto &op : launch.getBody().front().without_terminator()) {
    rewriter.clone(op, mapping);
  }
  upmem::DpuProgramOp::ensureTerminator(dpuProgram.getBody(), rewriter,
                                        launch->getLoc());

  rewriter.setInsertionPoint(launch);
  rewriter.create<upmem::WaitForOp>(launch->getLoc(), upmemWgAlloc.getResult());
  rewriter.eraseOp(launch);
  for (auto op : allocsToDelete) {
    rewriter.eraseOp(op);
  }

  for (auto user : wgAlloc.getResult().getUsers()) {
    if (auto free = llvm::dyn_cast_or_null<cnm::FreeWorkgroupOp>(user)) {
      rewriter.setInsertionPoint(free);
      rewriter.create<upmem::FreeDPUsOp>(free->getLoc(), upmemWgAlloc.getResult());
      rewriter.eraseOp(free);
    }
  }

  if (wgAlloc.getResult().use_empty()) {
    rewriter.eraseOp(wgAlloc);
    return success();
  } else {
    return wgAlloc.getResult().user_begin()->emitOpError(
        "Unexpected workgroup usage");
  }
}

struct ConvertCnmTerminatorToUPMEM
    : public OpConversionPattern<cnm::TerminatorOp> {
  using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::TerminatorOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op); // gets generated by ConvertCnmLaunchToUPMEM
    return success();
  }
};

} // namespace

struct ConvertCnmToUPMEMPass
    : public ::impl::ConvertCnmToUPMEMPassBase<ConvertCnmToUPMEMPass> {
  void runOnOperation() final {

    auto rootModule = getOperation();
    auto sym = SymbolTable::lookupSymbolIn(rootModule, "dpu_kernels");
    ModuleOp dpuKernelModule = llvm::dyn_cast_or_null<ModuleOp>(sym);
    if (!dpuKernelModule && sym) {
      mlir::emitError(sym->getLoc(), "Should be a module");
      signalPassFailure();
      return;
    }
    if (!dpuKernelModule) {
      OpBuilder builder(&getContext());
      builder.setInsertionPointToEnd(&rootModule.getBodyRegion().front());
      dpuKernelModule =
          builder.create<ModuleOp>(rootModule->getLoc(), "dpu_kernels");
    }

    SmallVector<LaunchOp> launchOps;
    rootModule->walk(
        [&](cnm::LaunchOp launch) { launchOps.push_back(launch); });

    SymbolTable rootSymTable(rootModule);

    IRRewriter rewriter(&getContext());
    for (auto launch : launchOps) {
      if (failed(convertCnmLaunchToUpmem(launch, rewriter, rootSymTable,
                                         dpuKernelModule))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<Pass> createConvertCnmToUPMEMPass() {
  return std::make_unique<ConvertCnmToUPMEMPass>();
}

} // namespace mlir::cnm

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/CinmUtils.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"
#include "upmem_cost_model/Types.h"

#include <cstdint>
#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

#include <upmem_cost_model/ProgramBuilder.h>

#define DEBUG_TYPE "upmem-cpp-sim"

namespace mlir::upmem {

/// Estimate the cost of the host side of a tiled GEMV (mv2) kernel.
///
/// Transfer costs use the same formula as OpCountSimulator's ScatterOp/GatherOp
/// case via scatterGatherCost().
double UpmemSimulator::simulateFullGemv(std::chrono::milliseconds timeout,
                                        int64_t M, int64_t K, int64_t mramRows,
                                        int64_t mramCols, int64_t wramRows,
                                        int64_t wramCols, int64_t dpuRows,
                                        int64_t dpuCols, int64_t tasklets,
                                        upmem_cm::DType dty) {
  // Cost of one scatter/gather of `elemsPerDpu` i32 elements across all DPUs.
  auto xferCost = [&](int64_t elemsPerDpu) {
    return scatterGatherCost(elemsPerDpu, upmem_cm::dtypeBytes(dty),
                             std::max(1L, dpuCols * dpuRows / 64), 64);
  };

  // DPU compute cost (one DPU, accounts for tasklet parallelism inside).
  double dpuCost =
      this->simulateGemv(timeout, static_cast<int>(tasklets), mramRows,
                         mramCols, wramRows, wramCols, dty);

  // Per inner-loop (col-tile) iteration: 3 scatters + wait + 1 gather.
  double innerIterCost = xferCost(mramRows * mramCols) // scatter A tile
                         + xferCost(mramCols)          // scatter x tile
                         + xferCost(mramRows)          // scatter y (init)
                         + dpuCost                     // DPU kernel
                         + xferCost(mramRows);         // gather y (result)

  int64_t innerTrips = K / (dpuCols * mramCols);
  int64_t outerTrips = M / (dpuRows * mramRows);
  return static_cast<double>(outerTrips * innerTrips) * innerIterCost;
}

/// Estimate the cost of the host side of a tiled reduction.
/// This is like reducing an <MxK> tensor into <M>.
double UpmemSimulator::simulateTailReduction(
    std::chrono::milliseconds timeoutMs, int64_t M, int64_t K,
    cinm::ReduceMethod reduction, int64_t mramRows, int64_t mramCols,
    int64_t wramRows, int64_t wramCols, int64_t dpuRows, int64_t dpuCols,
    int64_t taskletRows, int64_t taskletCols, upmem_cm::DType dty) {

  // Cost of one scatter/gather of `elemsPerDpu` i32 elements across all DPUs.
  auto xferCost = [&](int64_t elemsPerDpu) {
    return scatterGatherCost(elemsPerDpu, upmem_cm::dtypeBytes(dty),
                             std::max(1L, (dpuRows * dpuCols) / 64), 64);
  };

  // DPU compute cost (one DPU, accounts for tasklet parallelism inside).
  double dpuCost =
      this->simulateReduction(timeoutMs, reduction, taskletRows, taskletCols,
                              mramRows, mramCols, wramRows, wramCols, dty);

  // Per inner-loop (col-tile) iteration: scatter A + scatter y + wait + gather
  // y.
  double innerIterCost = xferCost(mramRows * mramCols) // scatter A tile
                         + xferCost(mramRows)  // scatter y (running partial)
                         + dpuCost             // DPU kernel
                         + xferCost(mramRows); // gather y (result)

  int64_t innerTrips = K / (dpuCols * mramCols);
  int64_t outerTrips = M / (dpuRows * mramRows);
  return static_cast<double>(outerTrips * innerTrips) * innerIterCost;
}

} // namespace mlir::upmem

// ===----------------------------------------------------------------------===//
// IR emission template for tail reduction
// ===----------------------------------------------------------------------===//

namespace mlir {
namespace {

// Pack input tiles into aStage for scatter. Loop steps are (mramRows, mramCols)
// so the IVs directly represent the row/col offsets within the current M/K
// tile. flatDpu is recovered via affine floordiv on the IVs.
static void packATile(OpBuilder &b, Location loc, Value input, Value aStage,
                      Value mOff, Value kOff, int64_t dpuRows, int64_t dpuCols,
                      int64_t mramRows, int64_t mramCols) {
  MLIRContext *ctx = b.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  AffineMap flatDpuMap = AffineMap::get(
      2, 0, {d0.floorDiv(mramRows) * dpuCols + d1.floorDiv(mramCols)}, ctx);

  cinm::createNestedAffineForLoops(
      b, loc, {dpuRows * mramRows, dpuCols * mramCols}, {mramRows, mramCols},
      {},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value rowBase = arith::AddIOp::create(b, loc, mOff, ivs[0]);
        Value colBase = arith::AddIOp::create(b, loc, kOff, ivs[1]);
        Value flatDpu = affine::AffineApplyOp::create(
            b, loc, flatDpuMap, ValueRange{ivs[0], ivs[1]});

        Value src = memref::SubViewOp::create(
            b, loc, input, ArrayRef<OpFoldResult>{rowBase, colBase},
            ArrayRef<OpFoldResult>{b.getIndexAttr(mramRows),
                                   b.getIndexAttr(mramCols)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        Value dst3 = memref::SubViewOp::create(
            b, loc, aStage,
            ArrayRef<OpFoldResult>{flatDpu, b.getIndexAttr(0),
                                   b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(mramRows),
                                   b.getIndexAttr(mramCols)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                   b.getIndexAttr(1)});
        Value dst = memref::CollapseShapeOp::create(
            b, loc, dst3, ArrayRef<ReassociationIndices>{{0, 1}, {2}});
        memref::CopyOp::create(b, loc, src, dst);
        return {};
      });
}

// Broadcast output row slices into yStage for all DPUs. Outer step is mramRows
// so iv0 is the row offset directly; iv1 is the dc index (step 1).
static void packYTile(OpBuilder &b, Location loc, Value output, Value yStage,
                      Value mOff, int64_t dpuRows, int64_t dpuCols,
                      int64_t mramRows) {
  MLIRContext *ctx = b.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  AffineMap flatDpuMap =
      AffineMap::get(2, 0, {d0.floorDiv(mramRows) * dpuCols + d1}, ctx);

  cinm::createNestedAffineForLoops(
      b, loc, {dpuRows * mramRows, dpuCols}, {mramRows, 1}, {},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value rowBase = arith::AddIOp::create(b, loc, mOff, ivs[0]);
        Value flatDpu = affine::AffineApplyOp::create(
            b, loc, flatDpuMap, ValueRange{ivs[0], ivs[1]});
        Value srcRow = memref::SubViewOp::create(
            b, loc, output, ArrayRef<OpFoldResult>{rowBase},
            ArrayRef<OpFoldResult>{b.getIndexAttr(mramRows)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        Value dst2 = memref::SubViewOp::create(
            b, loc, yStage, ArrayRef<OpFoldResult>{flatDpu, b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(mramRows)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        Value dst = memref::CollapseShapeOp::create(
            b, loc, dst2, ArrayRef<ReassociationIndices>{{0, 1}});
        memref::CopyOp::create(b, loc, srcRow, dst);
        return {};
      });
}

// Merge yStage back into output after gather. Outer step is mramRows so iv0
// is the row offset directly; firstDpu is recovered via affine floordiv.
static void mergeYTile(OpBuilder &b, Location loc, Value yStage, Value output,
                       Value mOff, int64_t dpuRows, int64_t dpuCols,
                       int64_t mramRows) {
  MLIRContext *ctx = b.getContext();
  AffineMap firstDpuMap = AffineMap::get(
      1, 0, {getAffineDimExpr(0, ctx).floorDiv(mramRows) * dpuCols}, ctx);

  cinm::createNestedAffineForLoops(
      b, loc, {dpuRows * mramRows}, {mramRows}, {},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value rowBase = arith::AddIOp::create(b, loc, mOff, ivs[0]);
        Value firstDpu = affine::AffineApplyOp::create(b, loc, firstDpuMap,
                                                       ValueRange{ivs[0]});
        Value outRow = memref::SubViewOp::create(
            b, loc, output, ArrayRef<OpFoldResult>{rowBase},
            ArrayRef<OpFoldResult>{b.getIndexAttr(mramRows)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        Value first2 = memref::SubViewOp::create(
            b, loc, yStage, ArrayRef<OpFoldResult>{firstDpu, b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(mramRows)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
        Value first1 = memref::CollapseShapeOp::create(
            b, loc, first2, ArrayRef<ReassociationIndices>{{0, 1}});
        memref::CopyOp::create(b, loc, first1, outRow);

        if (dpuCols > 1) {
          cinm::createNestedAffineForLoops(
              b, loc, {mramRows}, {1}, {},
              [&](OpBuilder &b, Location loc, ValueRange ivs2,
                  ValueRange) -> SmallVector<Value> {
                Value mr = ivs2[0];
                Value acc =
                    memref::LoadOp::create(b, loc, outRow, ValueRange{mr});
                for (int64_t dc = 1; dc < dpuCols; ++dc) {
                  Value flatDpu = arith::AddIOp::create(
                      b, loc, firstDpu,
                      arith::ConstantIndexOp::create(b, loc, dc));
                  acc = arith::AddIOp::create(
                      b, loc, acc,
                      memref::LoadOp::create(b, loc, yStage,
                                             ValueRange{flatDpu, mr}));
                }
                memref::StoreOp::create(b, loc, acc, outRow, ValueRange{mr});
                return {};
              });
        }
        return {};
      });
}

} // namespace

upmem::DpuProgramOp createDpuTailReductionKernel(
    Location loc, RewriterBase &rewriter, ModuleOp target, int64_t mramRows,
    int64_t mramCols, int64_t wramRows, int64_t wramCols, int64_t taskletRows,
    int64_t taskletCols, FlatSymbolRefAttr &aBufSym,
    FlatSymbolRefAttr &yBufSym) {

  rewriter.clearInsertionPoint();
  const auto taskletCount = taskletRows * taskletCols;
  auto kernl = upmem::DpuProgramOp::create(rewriter, loc, "red", taskletCount);
  SymbolTable symTable(target);
  symTable.insert(kernl);

  rewriter.setInsertionPointToStart(&kernl.getBody().front());

  // clang-format off
  // TODO generate buffers and loop nests:
  /*
   %abuf_mram = upmem.static_alloc @bufa(mram): memref<mramRows x mramCols xi32, #upmem.mram>
   %ybuf_mram = upmem.static_alloc @bufy(mram): memref<mramRows xi32, #upmem.mram>

   %abuf_wram = upmem.static_alloc @bufaw(wram): memref<taskletRows x wramRows x taskletCols x wramCols xi32, #upmem.wram>
   %ybuf_wram = upmem.static_alloc @bufyw(wram): memref<taskletRows x taskletCols x wramRows xi32, #upmem.wram>
   %ybuf_wram_2 = upmem.static_alloc @bufyw2(wram): memref<taskletRows * wramRows xi32, #upmem.wram>
  
   %tid = upmem.tasklet_dim()  // thread id
   %tcolix = (arith ops that does tid modulo taskletCols)
   %trowix = (arith ops that does (tid - tcolIx) floordiv taskletCols)
   
  affine.for %mr = 0 to mramRows step (taskletRows * wramRows) {
    affine.for %mc = 0 to mramCols step (taskletCols * wramCols) {
      if %tcolix == 0 {
        for %i in 0 to wramRows {
          // transfer contiguous rows (need to loop unless wramRows == mramRows?)
          %myMramCols = subview of the %abuf_mram with 
            sizes [1, taskletCols * wramCols] offsets [%mr + %trowix + %i * taskletRows, %mc] strides [1, 1]
            : memref<taskletCols * wramCols x eltTy>
          %myMramReshaped = memref.reshape %myMramCols 
            : memref<taskletCols * wramCols x eltTy> into memref<taskletCols x wramCols x eltTy>
          %myWramCols = subview of the %abuf_wram with
            sizes [1, 1, taskletCols, wramCols] offsets [%trowix, %i, 0, 0] strides [1, 1, 1, 1]
            : memref<taskletCols x wramCols x eltTy>  // rank-reduced, drops the two size-1 dims

          upmem.local_transfer %myMramReshaped into %myWramCols
        }
      }
      upmem.barrier()

      %myA = subview of the %abuf_wram with
            sizes [1, wramRows, 1, wramCols] offsets [%trowix, 0, %tcolix, 0] strides [1, 1, 1, 1]
            : memref<wramRows x wramCols x eltTy>  // rank-reduced; result strides are [taskletCols*wramCols, 1]

      %myY = subview of the %ybuf_wram with
            sizes [1, 1, wramRows] offsets [%trowix, %tcolix, 0] strides [1, 1, 1]
            : memref<wramRows x eltTy>  // rank-reduced; contiguous

      for %i = 0 to wramRows { // inner reduction loop
        for %j = 0 to wramCols {
          %myY[i] += %myA[i, j]
        }
      }
    }

    upmem.barrier()
    if %tcolix == 0 {
      %myMram = subview of %ybuf_mram with
            sizes [wramRows] offsets [%mr + %trowix * wramRows] strides [1]
            : memref<wramRows x eltTy>
      %myWram2 = subview of %ybuf_wram_2 with
            sizes [wramRows] offsets [%trowix * wramRows] strides [1]
            : memref<wramRows x eltTy>
      upmem.local_transfer %myMram into %myWram2 // transfer initial values

      for %i in 0 to wramRows {
        // Partial result accumulation into the buffer
        for %ci in 0 to taskletCols {
          %ybuf_wram2[%trowix * wramRows + %i] += %ybuf_wram[%trowix, %ci, %i]
        }
      }
      upmem.local_transfer %myWram2 into %myMram
    }
  }
  

  */
  // clang-format on


  return kernl;
}

/// Emit the host-side tiled loop nest for a tail reduction of input (memref
/// collapsed to <M x K>) into output (memref<M x elt>). Generates:
///
///   for m = 0 to M step dpuRows*mramRows:
///     for k = 0 to K step dpuCols*mramCols:
///       [pack A tile and y into staging buffers]
///       upmem.scatter aStage → @aBufSym
///       upmem.scatter yStage → @yBufSym
///       upmem.wait_for dpus
///       upmem.gather  yStage ← @yBufSym
///       [merge yStage back into output]
///
/// The hierarchy must have type <1 x (dpuRows*dpuCols) x tasklets>.
/// The DPU-side kernel (and the upmem.dpu_program containing aBufSym/yBufSym)
/// must be created separately before calling this function.
void generateTailReduction(cinm::ReduceOp op, RewriterBase &rewriter,
                           int64_t dpuRows, int64_t dpuCols, int64_t mramRows,
                           int64_t mramCols, int64_t wramRows, int64_t wramCols,
                           int64_t taskletRows, int64_t taskletCols) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();
  auto inputTy = cast<MemRefType>(op.getInput().getType());
  Type eltTy = inputTy.getElementType();
  int64_t numDpus = dpuRows * dpuCols;

  const int64_t M = computeProduct(inputTy.getShape().drop_back());
  const int64_t K = inputTy.getShape().back();

  auto reshapedInput =
      reshapeStatic(rewriter, op->getLoc(), op.getInput(), {M, K});

  auto output =
      memref::AllocOp::create(rewriter, loc, MemRefType::get({M}, eltTy));

  // Flat-DPU-major staging buffers: one contiguous mramRows×mramCols slice
  // per DPU for A, and one mramRows slice per DPU for the running y
  // partial.
  Value aStage = memref::AllocOp::create(
      rewriter, loc, MemRefType::get({numDpus, mramRows, mramCols}, eltTy));
  Value yStage = memref::AllocOp::create(
      rewriter, loc, MemRefType::get({numDpus, mramRows}, eltTy));
  FlatSymbolRefAttr aBufSym, yBufSym;
  Value dpus;

  auto parentMod = op->getParentOfType<ModuleOp>();

  upmem::DpuProgramOp krnlOp = createDpuTailReductionKernel(
      loc, rewriter, parentMod, mramRows, mramCols, wramRows, wramCols,
      taskletRows, taskletCols, aBufSym, yBufSym);

  // Scatter maps for hierarchy <1 x numDpus x tasklets>.
  // The flat DPU index is rank*numDpus + dpu; since numRanks=1, rank=0
  // always, so flat = dpu. Map: (rank, dpu) -> (dpu, 0, ...).
  auto dpuDim = getAffineDimExpr(1, ctx);
  auto zero = getAffineConstantExpr(0, ctx);
  AffineMap aMap = AffineMap::get(2, 0, {dpuDim, zero, zero}, ctx);
  AffineMap yMap = AffineMap::get(2, 0, {dpuDim, zero}, ctx);

  cinm::createNestedAffineForLoops(
      rewriter, loc, {M, K}, {dpuRows * mramRows, dpuCols * mramCols},
      /*iterArgInit=*/ValueRange{},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value mOff = ivs[0];
        Value kOff = ivs[1];

        packATile(b, loc, reshapedInput, aStage, mOff, kOff, dpuRows, dpuCols,
                  mramRows, mramCols);
        packYTile(b, loc, output, yStage, mOff, dpuRows, dpuCols, mramRows);

        upmem::ScatterOp::create(b, loc, aStage, aBufSym,
                                 static_cast<uint64_t>(mramRows * mramCols),
                                 aMap, dpus);
        upmem::ScatterOp::create(b, loc, yStage, yBufSym,
                                 static_cast<uint64_t>(mramRows), yMap, dpus);
        upmem::WaitForOp::create(b, loc, dpus);
        upmem::GatherOp::create(b, loc, yStage, yBufSym,
                                static_cast<uint64_t>(mramRows), yMap, dpus);

        mergeYTile(b, loc, yStage, output, mOff, dpuRows, dpuCols, mramRows);
        return {};
      });

  memref::DeallocOp::create(rewriter, loc, aStage);
  memref::DeallocOp::create(rewriter, loc, yStage);
}

} // namespace mlir

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/CinmUtils.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"
#include "upmem_cost_model/Types.h"

#include <cstdint>
#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
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

} // namespace

upmem::DpuProgramOp createDpuTailReductionKernel(
    Location loc, RewriterBase &rewriter, ModuleOp target, int64_t mramRows,
    int64_t mramCols, int64_t wramRows, int64_t wramCols, int64_t taskletRows,
    int64_t taskletCols, Type eltTy, llvm::StringRef &aBufSym,
    llvm::StringRef &yBufSym) {

  rewriter.clearInsertionPoint();
  const auto taskletCount = taskletRows * taskletCols;
  auto kernl = upmem::DpuProgramOp::create(rewriter, loc, "red", taskletCount);
  SymbolTable symTable(target);
  symTable.insert(kernl);
  auto *block = &kernl.getBody().emplaceBlock();
  rewriter.setInsertionPointToEnd(block);
  upmem::ReturnOp::create(rewriter, loc);

  rewriter.setInsertionPointToStart(block);
  MLIRContext *ctx = rewriter.getContext();

  auto mramMS =
      rewriter.getAttr<upmem::DpuMemSpaceAttr>(upmem::DpuMemSpace::MRAM);
  auto wramMS =
      rewriter.getAttr<upmem::DpuMemSpaceAttr>(upmem::DpuMemSpace::WRAM);

  // MRAM buffers (named so host scatter/gather can reference them)
  auto abufMram = upmem::StaticAllocOp::create(
      rewriter, loc,
      MemRefType::get({mramRows, mramCols}, eltTy, MemRefLayoutAttrInterface{},
                      mramMS),
      upmem::DpuMemSpace::MRAM, "bufa");
  aBufSym = *abufMram.getSymName();

  auto ybufMram = upmem::StaticAllocOp::create(
      rewriter, loc,
      MemRefType::get({mramRows}, eltTy, MemRefLayoutAttrInterface{}, mramMS),
      upmem::DpuMemSpace::MRAM, "bufy");
  yBufSym = *ybufMram.getSymName();

  // WRAM staging buffers (shared across tasklets)
  Value abufWram =
      upmem::StaticAllocOp::create(
          rewriter, loc,
          MemRefType::get({taskletRows, wramRows, taskletCols, wramCols}, eltTy,
                          MemRefLayoutAttrInterface{}, wramMS),
          upmem::DpuMemSpace::WRAM)
          .getBuffer();
  Value ybufWram =
      upmem::StaticAllocOp::create(
          rewriter, loc,
          MemRefType::get({taskletRows, taskletCols, wramRows}, eltTy,
                          MemRefLayoutAttrInterface{}, wramMS),
          upmem::DpuMemSpace::WRAM)
          .getBuffer();
  Value ybufWram2 = upmem::StaticAllocOp::create(
                        rewriter, loc,
                        MemRefType::get({taskletRows * wramRows}, eltTy,
                                        MemRefLayoutAttrInterface{}, wramMS),
                        upmem::DpuMemSpace::WRAM)
                        .getBuffer();

  // Tasklet indices: tcolix = tid % taskletCols, trowix = (tid - tcolix) /
  // taskletCols
  Value tid = upmem::TaskletDimOp::create(rewriter, loc).getResult();
  Value tcolsCst = arith::ConstantIndexOp::create(rewriter, loc, taskletCols);
  Value tcolix = arith::RemUIOp::create(rewriter, loc, tid, tcolsCst);
  Value trowix = arith::DivUIOp::create(
      rewriter, loc, arith::SubIOp::create(rewriter, loc, tid, tcolix),
      tcolsCst);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value loadCond = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, tcolix, zero);
  DenseI8ArrayAttr taskletIdsAttr;
  {
    SmallVector<int8_t, 8> taskletIds;
    for (int8_t i = 0; i < taskletCount; i++) {
      if (i % taskletCols == 0)
        taskletIds.push_back(i);
    }
    taskletIdsAttr = rewriter.getDenseI8ArrayAttr(std::move(taskletIds));
  }

  // Precomputed result types for rank-reducing subviews.
  // Strides are derived from the source layout (see comment block above).
  MemRefType myMramFlatTy = MemRefType::get(
      {taskletCols * wramCols}, eltTy,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}), mramMS);
  MemRefType myMramReshapedTy = MemRefType::get(
      {taskletCols, wramCols}, eltTy,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {wramCols, 1}), mramMS);
  MemRefType myWramColsTy = MemRefType::get(
      {taskletCols, wramCols}, eltTy,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {wramCols, 1}), wramMS);
  MemRefType myATy =
      MemRefType::get({wramRows, wramCols}, eltTy,
                      StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                                             {taskletCols * wramCols, 1}),
                      wramMS);
  MemRefType myYTy = MemRefType::get(
      {wramRows}, eltTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}),
      wramMS);
  MemRefType sliceMramTy = MemRefType::get(
      {wramRows}, eltTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}),
      mramMS);
  MemRefType sliceWramTy = MemRefType::get(
      {wramRows}, eltTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}),
      wramMS);

  auto elemAdd = [&](OpBuilder &b, Location loc, Value lhs,
                     Value rhs) -> Value {
    return eltTy.isIntOrIndex()
               ? arith::AddIOp::create(b, loc, lhs, rhs).getResult()
               : arith::AddFOp::create(b, loc, lhs, rhs).getResult();
  };

  // mr loop: iterate MRAM row chunks of size taskletRows*wramRows
  cinm::createNestedAffineForLoops(
      rewriter, loc, {mramRows}, {taskletRows * wramRows}, {},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value mr = ivs[0];

        // mc loop: iterate MRAM col chunks of size taskletCols*wramCols
        cinm::createNestedAffineForLoops(
            b, loc, {mramCols}, {taskletCols * wramCols}, {},
            [&](OpBuilder &b, Location loc, ValueRange ivs2,
                ValueRange) -> SmallVector<Value> {
              Value mc = ivs2[0];

              // Only the tcolix==0 tasklet in each row group loads A into WRAM

              auto loadIf =
                  scf::IfOp::create(b, loc, TypeRange{}, loadCond, false);
              loadIf->setAttr("upmem_cm.const_tasklets", taskletIdsAttr);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(&loadIf.getThenRegion().front());

                // i loop: load one MRAM row per iteration into abufWram[trowix,
                // i, :, :]
                Value taskletRowsCst =
                    arith::ConstantIndexOp::create(b, loc, taskletRows);
                cinm::createNestedAffineForLoops(
                    b, loc, {wramRows}, {1}, {},
                    [&](OpBuilder &b, Location loc, ValueRange ivs3,
                        ValueRange) -> SmallVector<Value> {
                      Value i = ivs3[0];
                      // rowOff = mr + trowix + i * taskletRows  (interleaved
                      // row assignment)
                      Value rowOff = arith::AddIOp::create(
                          b, loc, mr,
                          arith::AddIOp::create(
                              b, loc, trowix,
                              arith::MulIOp::create(b, loc, i,
                                                    taskletRowsCst)));
                      // Flat 1D view of the MRAM row, then reshape to
                      // [taskletCols, wramCols]
                      Value myMramFlat = memref::SubViewOp::create(
                          b, loc, myMramFlatTy, abufMram.getBuffer(),
                          ArrayRef<OpFoldResult>{rowOff, mc},
                          ArrayRef<OpFoldResult>{
                              b.getIndexAttr(1),
                              b.getIndexAttr(taskletCols * wramCols)},
                          ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                                 b.getIndexAttr(1)});
                      Value myMramReshaped = memref::ExpandShapeOp::create(
                          b, loc, myMramReshapedTy, myMramFlat,
                          ArrayRef<ReassociationIndices>{{0, 1}});
                      // Rank-reducing subview: abufWram[trowix, i, :, :] ->
                      // [taskletCols, wramCols]
                      Value myWramCols = memref::SubViewOp::create(
                          b, loc, myWramColsTy, abufWram,
                          ArrayRef<OpFoldResult>{trowix, i, b.getIndexAttr(0),
                                                 b.getIndexAttr(0)},
                          ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                                 b.getIndexAttr(1),
                                                 b.getIndexAttr(taskletCols),
                                                 b.getIndexAttr(wramCols)},
                          ArrayRef<OpFoldResult>{
                              b.getIndexAttr(1), b.getIndexAttr(1),
                              b.getIndexAttr(1), b.getIndexAttr(1)});
                      upmem::LocalTransferOp::create(b, loc, myMramReshaped,
                                                     myWramCols);
                      return {};
                    });
              } // guard restores insertion point to after loadIf

              upmem::BarrierOp::create(b, loc);

              // Each tasklet processes its slice: abufWram[trowix, :, tcolix,
              // :]
              Value myA = memref::SubViewOp::create(
                  b, loc, myATy, abufWram,
                  ArrayRef<OpFoldResult>{trowix, b.getIndexAttr(0), tcolix,
                                         b.getIndexAttr(0)},
                  ArrayRef<OpFoldResult>{
                      b.getIndexAttr(1), b.getIndexAttr(wramRows),
                      b.getIndexAttr(1), b.getIndexAttr(wramCols)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                         b.getIndexAttr(1), b.getIndexAttr(1)});
              Value myY = memref::SubViewOp::create(
                  b, loc, myYTy, ybufWram,
                  ArrayRef<OpFoldResult>{trowix, tcolix, b.getIndexAttr(0)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                         b.getIndexAttr(wramRows)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                         b.getIndexAttr(1)});

              // Reduction: myY[i] += myA[i, j]
              cinm::createNestedAffineForLoops(
                  b, loc, {wramRows, wramCols}, {1, 1}, {},
                  [&](OpBuilder &b, Location loc, ValueRange ivs3,
                      ValueRange) -> SmallVector<Value> {
                    Value i = ivs3[0], j = ivs3[1];
                    Value aij =
                        memref::LoadOp::create(b, loc, myA, ValueRange{i, j});
                    Value yi =
                        memref::LoadOp::create(b, loc, myY, ValueRange{i});
                    memref::StoreOp::create(b, loc, elemAdd(b, loc, yi, aij),
                                            myY, ValueRange{i});
                    return {};
                  });

              return {};
            }); // end mc loop

        // All tasklets done with ybufWram; row-leaders merge and write back
        upmem::BarrierOp::create(b, loc);
        auto accumIf = scf::IfOp::create(b, loc, TypeRange{}, loadCond, false);
        accumIf->setAttr("upmem_cm.const_tasklets", taskletIdsAttr);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&accumIf.getThenRegion().front());

          Value wramRowsCst = arith::ConstantIndexOp::create(b, loc, wramRows);
          Value mramOff = arith::AddIOp::create(
              b, loc, mr, arith::MulIOp::create(b, loc, trowix, wramRowsCst));
          Value myMramSlice = memref::SubViewOp::create(
              b, loc, sliceMramTy, ybufMram.getBuffer(),
              ArrayRef<OpFoldResult>{mramOff},
              ArrayRef<OpFoldResult>{b.getIndexAttr(wramRows)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});

          Value wram2Off = arith::MulIOp::create(b, loc, trowix, wramRowsCst);
          Value myWram2 = memref::SubViewOp::create(
              b, loc, sliceWramTy, ybufWram2, ArrayRef<OpFoldResult>{wram2Off},
              ArrayRef<OpFoldResult>{b.getIndexAttr(wramRows)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});

          // Load running partial sum from MRAM, accumulate ybufWram
          // contributions
          upmem::LocalTransferOp::create(b, loc, myMramSlice, myWram2);
          cinm::createNestedAffineForLoops(
              b, loc, {wramRows, taskletCols}, {1, 1}, {},
              [&](OpBuilder &b, Location loc, ValueRange ivs3,
                  ValueRange) -> SmallVector<Value> {
                Value i = ivs3[0], ci = ivs3[1];
                Value ywi = memref::LoadOp::create(b, loc, ybufWram,
                                                   ValueRange{trowix, ci, i});
                Value w2i =
                    memref::LoadOp::create(b, loc, myWram2, ValueRange{i});
                memref::StoreOp::create(b, loc, elemAdd(b, loc, w2i, ywi),
                                        myWram2, ValueRange{i});
                return {};
              });
          upmem::LocalTransferOp::create(b, loc, myWram2, myMramSlice);
        }

        return {};
      }); // end mr loop

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
void upmem::generateTailReduction(cinm::ReduceOp op, RewriterBase &rewriter,
                                  int64_t dpuRows, int64_t dpuCols,
                                  int64_t mramRows, int64_t mramCols,
                                  int64_t wramRows, int64_t wramCols,
                                  int64_t taskletRows, int64_t taskletCols) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();
  auto inputTy = cast<MemRefType>(op.getInput().getType());
  Type eltTy = inputTy.getElementType();
  int64_t numDpus = dpuRows * dpuCols;
  int64_t numTasklets = taskletRows * taskletCols;

  const int64_t M = computeProduct(inputTy.getShape().drop_back());
  const int64_t K = inputTy.getShape().back();

  auto reshapedInput =
      reshapeStatic(rewriter, loc, op.getInput(), inputTy, {M, K});

  auto output =
      memref::AllocOp::create(rewriter, loc, MemRefType::get({M}, eltTy));

  // Flat-DPU-major staging buffers: one contiguous mramRows×mramCols slice
  // per DPU for A, and one mramRows slice per DPU for the running y
  // partial.
  Value aStage = memref::AllocOp::create(
      rewriter, loc, MemRefType::get({numDpus, mramRows, mramCols}, eltTy));
  Value yStage = memref::AllocOp::create(
      rewriter, loc, MemRefType::get({dpuRows, dpuCols, mramRows}, eltTy));

  llvm::StringRef aBufSym, yBufSym;

  auto parentMod = op->getParentOfType<ModuleOp>();

  upmem::DpuProgramOp krnlOp = createDpuTailReductionKernel(
      loc, rewriter, parentMod, mramRows, mramCols, wramRows, wramCols,
      taskletRows, taskletCols, eltTy, aBufSym, yBufSym);

  rewriter.setInsertionPointAfter(yStage.getDefiningOp());
  // dpus = upmem.alloc_dpus with program (reference to the krnlOp) :
  // !upmem.hierarchy<1 x (dpuRows*dpuCols) x taskletCount>
  auto wgTy = upmem::DeviceHierarchyType::get(ctx, 1, numDpus, numTasklets);
  auto dpuProgramSymbol = SymbolRefAttr::get(krnlOp.getSymNameAttr());
  auto dpus = upmem::AllocDPUsOp::create(rewriter, loc, wgTy, dpuProgramSymbol);

  // Scatter maps for hierarchy <1 x numDpus x tasklets>.
  // The flat DPU index is rank*numDpus + dpu; since numRanks=1, rank=0
  // always, so flat = dpu.
  // aStage: (rank, dpu) -> (dpu, 0, 0)
  // yStage: (rank, dpu) -> (dpu / dpuCols, dpu % dpuCols, 0)
  auto dpuDim = getAffineDimExpr(1, ctx);
  auto zero = getAffineConstantExpr(0, ctx);
  AffineMap aMap = AffineMap::get(2, 0, {dpuDim, zero, zero}, ctx);
  AffineMap yMap = AffineMap::get(
      2, 0, {dpuDim.floorDiv(dpuCols), dpuDim % dpuCols, zero}, ctx);

  auto arithReductionKind = cinm::getArithConstant(op.getMethod(), eltTy);
  auto neutral =
      arith::getIdentityValue(arithReductionKind, eltTy, rewriter, loc);

  cinm::createNestedAffineForLoops(
      rewriter, loc, {M}, {dpuRows * mramRows},
      /*iterArgInit=*/ValueRange{},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value mOff = ivs[0];

        // Fill the yStage output buffer with the neutral
        // element of the reduction.
        linalg::FillOp::create(rewriter, loc, neutral, yStage);

        // Scatter it. We do this only once - if there are several iterations
        // over dpuCols*mramCols, then the future calls to wait_for will reuse
        // the partial results that are already in mram.
        upmem::ScatterOp::create(b, loc, yStage, yBufSym,
                                 static_cast<uint64_t>(mramRows), yMap, dpus);

        cinm::createNestedAffineForLoops(
            b, loc, {K}, {dpuCols * mramCols},
            /*iterArgInit=*/ValueRange{},
            [&](OpBuilder &b, Location loc, ValueRange ivs,
                ValueRange) -> SmallVector<Value> {
              Value kOff = ivs[0];

              packATile(b, loc, reshapedInput, aStage, mOff, kOff, dpuRows,
                        dpuCols, mramRows, mramCols);
              upmem::ScatterOp::create(
                  b, loc, aStage, aBufSym,
                  static_cast<uint64_t>(mramRows * mramCols), aMap, dpus);
              upmem::WaitForOp::create(b, loc, dpus);
              return {};
            });

        // Once we're done with a set of rows, we gather their results.
        // We still need to reduce over dpuCols.
        upmem::GatherOp::create(b, loc, yStage, yBufSym,
                                static_cast<uint64_t>(mramRows), yMap, dpus);
        // Subview of output for this row tile, shaped to match yStage after
        // reducing dpuCols: output[mOff .. mOff + dpuRows*mramRows).
        Value outRows = memref::SubViewOp::create(
            b, loc, output, ArrayRef<OpFoldResult>{mOff},
            ArrayRef<OpFoldResult>{b.getIndexAttr(dpuRows * mramRows)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        linalg::FillOp::create(b, loc, neutral, outRows);

        // Reshape outRows {dpuRows*mramRows} -> {dpuRows, mramRows}.
        // The subview has stride 1 and dynamic offset, so the 2D shape
        // has strides {mramRows, 1} with the same dynamic offset.
        Value outRows2D = memref::ExpandShapeOp::create(
            b, loc,
            MemRefType::get({dpuRows, mramRows}, eltTy,
                            StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                                                   {mramRows, 1})),
            outRows, ArrayRef<ReassociationIndices>{{0, 1}});

        // Reduce yStage {dpuRows, dpuCols, mramRows} over dim 1 (dpuCols)
        // into outRows2D {dpuRows, mramRows}, writing directly into output.
        linalg::ReduceOp::create(
            b, loc, ValueRange{yStage}, ValueRange{outRows2D},
            ArrayRef<int64_t>{1},
            [&](OpBuilder &b, Location loc, ValueRange args) {
              linalg::YieldOp::create(
                  b, loc, arith::getReductionOp(arithReductionKind, b, loc,
                                                args[0], args[1]));
            });
        return {};
      });


  Type resultTy = op.getResult().getType();
  if (isa<MemRefType>(resultTy)) {
    rewriter.replaceOp(op, output);
  } else if (!isa<ShapedType>(resultTy)) {
    // Scalar result: load the single element from output[0].
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
    Value scalar =
        memref::LoadOp::create(rewriter, loc, output, ValueRange{zero});
    rewriter.replaceOp(op, scalar);
  } else {
    op.emitError("generateTailReduction: unexpected result type ") << resultTy;
    rewriter.eraseOp(op);
  }
  memref::DeallocOp::create(rewriter, loc, aStage);
  memref::DeallocOp::create(rewriter, loc, yStage);

  upmem::FreeDPUsOp::create(rewriter, loc, dpus);
}

} // namespace mlir

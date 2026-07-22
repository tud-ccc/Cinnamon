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

#include "SimulatorBase.h"

#include <cstdint>
#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
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
                                        DType dty) {
  // Cost of one scatter/gather of `elemsPerDpu` i32 elements across all DPUs.
  auto xferCost = [&](int64_t elemsPerDpu) {
    return scatterGatherCost(elemsPerDpu, dty,
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
    int64_t taskletRows, int64_t taskletCols, DType dty) {

  // Cost of one scatter/gather of `elemsPerDpu` i32 elements across all DPUs.
  auto xferCost = [&](int64_t elemsPerDpu) {
    return scatterGatherCost(elemsPerDpu, dty,
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

// Emit the upmem.scatter(s) that place the [dpuRows*mramRows,
// dpuCols*mramCols] tile of `input` at (mOff, kOff) into each DPU's
// `aBufSym` MRAM buffer, scattering directly from `input` -- never staging
// through an intermediate host-side copy.
//
// `view` below is a DPU-major [dpuRows, dpuCols, mramRows, mramCols]
// reinterpretation of the tile (no data movement, just a stride swap versus
// `input`'s natural [dpuRows, mramRows, dpuCols, mramCols] split). Every
// upmem.scatter requires the elements it transfers per DPU to be contiguous
// in the host buffer (see upmem::ScatterOp::verify / getContiguousSuffixSize).
// Two cases, distinguished by whether one DPU's whole mramRows x mramCols
// chunk is itself one contiguous span:
//  - It is, when mramRows == 1 (only one row per DPU, trivially contiguous)
//    or the tile's row-to-row stride equals mramCols (no column-tiling: the
//    assigned columns already span the underlying matrix's full row, so
//    consecutive mram rows sit back-to-back in memory). Use the classic
//    (rank, dpu) scatter form, transferring the whole chunk in one shot.
//  - Otherwise, only each individual mram row (mramCols contiguous elements)
//    is guaranteed contiguous -- consecutive rows of the same DPU's tile are
//    not adjacent, since the underlying matrix is wider than this k-tile.
//    Use the UPMEM SDK's scatter transfer API instead, via the (rank, dpu,
//    block) scatter form with one block per mram row.
// Either way, `view`'s own (possibly non-mergeable) strides are used as-is:
// no memref.collapse_shape is needed, since the scatter map itself flattens
// the (dpuRow, dpuCol) grid via floordiv/mod on the flat dpu index.
static void scatterATile(OpBuilder &b, Location loc, Value input, Value mOff,
                         Value kOff, int64_t dpuRows, int64_t dpuCols,
                         int64_t mramRows, int64_t mramCols, StringRef aBufSym,
                         Value dpus) {
  Value tile2D = memref::SubViewOp::create(
      b, loc, input, ArrayRef<OpFoldResult>{mOff, kOff},
      ArrayRef<OpFoldResult>{b.getIndexAttr(dpuRows * mramRows),
                             b.getIndexAttr(dpuCols * mramCols)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});

  // Natural split of the tile: [dpuRows, mramRows, dpuCols, mramCols], with
  // strides inferred by expand_shape from tile2D's own layout (no manual
  // stride arithmetic).
  Value natural = memref::ExpandShapeOp::create(
      b, loc, ArrayRef<int64_t>{dpuRows, mramRows, dpuCols, mramCols}, tile2D,
      ArrayRef<ReassociationIndices>{{0, 1}, {2, 3}});
  auto naturalTy = cast<MemRefType>(natural.getType());
  auto [naturalStrides, naturalOffset] = naturalTy.getStridesAndOffset();
  assert(!ShapedType::isDynamic(naturalStrides[0]) &&
         !ShapedType::isDynamic(naturalStrides[1]) &&
         !ShapedType::isDynamic(naturalStrides[2]) &&
         !ShapedType::isDynamic(naturalStrides[3]) &&
         "scatterATile requires a 2D input with static strides");

  // DPU-major grouping needs [dpuRows, dpuCols, mramRows, mramCols] instead:
  // swap the middle two dims (mramRows, dpuCols). expand_shape can only
  // split dims in place, not reorder them, so the transpose itself still
  // needs a reinterpret_cast -- built on top of the natural split above (via
  // extract_strided_metadata) rather than computed from scratch.
  SmallVector<int64_t> viewStrides = {naturalStrides[0], naturalStrides[2],
                                      naturalStrides[1], naturalStrides[3]};
  auto meta = memref::ExtractStridedMetadataOp::create(b, loc, natural);
  MemRefType viewTy = MemRefType::get(
      {dpuRows, dpuCols, mramRows, mramCols}, naturalTy.getElementType(),
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                             viewStrides));
  Value view = memref::ReinterpretCastOp::create(
      b, loc, viewTy, meta.getBaseBuffer(),
      static_cast<OpFoldResult>(meta.getOffset()),
      getAsIndexOpFoldResult(
          b.getContext(),
          ArrayRef<int64_t>{dpuRows, dpuCols, mramRows, mramCols}),
      getAsIndexOpFoldResult(b.getContext(), viewStrides));

  MLIRContext *ctx = b.getContext();
  auto dpuDim = getAffineDimExpr(1, ctx);
  auto zero = getAffineConstantExpr(0, ctx);
  // `naturalStrides[1]` is the row-to-row stride within one DPU's own tile,
  // inherited from `input`'s row stride regardless of tiling.
  bool wholeChunkContiguous = mramRows == 1 || naturalStrides[1] == mramCols;

  if (wholeChunkContiguous) {
    AffineMap aMap = AffineMap::get(
        2, 0, {dpuDim.floorDiv(dpuCols), dpuDim % dpuCols, zero, zero}, ctx);
    upmem::ScatterOp::create(b, loc, view, aBufSym,
                             static_cast<uint64_t>(mramRows * mramCols), aMap,
                             dpus);
    return;
  }

  // (rank, dpu, block) form: one block per mram row.
  AffineMap aMap = AffineMap::get(3, 0,
                                  {dpuDim.floorDiv(dpuCols), dpuDim % dpuCols,
                                   getAffineDimExpr(2, ctx), zero},
                                  ctx);
  upmem::ScatterOnTaskletsOp::create(b, loc, view, aBufSym,
                                     static_cast<uint64_t>(mramCols), aMap,
                                     dpus, static_cast<int64_t>(mramRows));
}

// Pack the [dpuCols*mramCols] slice of `x` at kOff for scatter. Returns the
// memref to scatter from: a direct view of `x` if it's already contiguous
// enough (same criterion as upmem::ScatterOp::verify), otherwise `xStage`
// after a single memref.copy of the whole slice into it.
static Value packXSlice(OpBuilder &b, Location loc, Value x, Value xStage,
                        Value kOff, int64_t dpuCols, int64_t mramCols) {
  Value slice1D = memref::SubViewOp::create(
      b, loc, x, ArrayRef<OpFoldResult>{kOff},
      ArrayRef<OpFoldResult>{b.getIndexAttr(dpuCols * mramCols)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});

  // No transpose needed here (unlike scatterATile): a straight split.
  Value view = memref::ExpandShapeOp::create(
      b, loc, ArrayRef<int64_t>{dpuCols, mramCols}, slice1D,
      ArrayRef<ReassociationIndices>{{0, 1}});
  auto viewTy = cast<MemRefType>(view.getType());

  if (mlir::getContiguousSuffixSize(viewTy) == dpuCols * mramCols)
    return view;

  memref::CopyOp::create(b, loc, view, xStage);
  return xStage;
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
///       upmem.scatter A tile directly → @aBufSym
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

  // Flat-DPU-major staging buffer: one mramRows slice per DPU for the
  // running y partial. A is scattered directly from `reshapedInput`, no
  // staging (see scatterATile).
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

  // Scatter map for yStage, for hierarchy <1 x numDpus x tasklets>.
  // The flat DPU index is rank*numDpus + dpu; since numRanks=1, rank=0
  // always, so flat = dpu.
  // yStage: (rank, dpu) -> (dpu / dpuCols, dpu % dpuCols, 0)
  auto dpuDim = getAffineDimExpr(1, ctx);
  auto zero = getAffineConstantExpr(0, ctx);
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

              scatterATile(b, loc, reshapedInput, mOff, kOff, dpuRows, dpuCols,
                           mramRows, mramCols, aBufSym, dpus);
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
              linalg::YieldOp::create(b, loc,
                                      arith::getReductionOp(arithReductionKind,
                                                            b, loc, args[0],
                                                            args[1]));
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
  memref::DeallocOp::create(rewriter, loc, yStage);

  upmem::FreeDPUsOp::create(rewriter, loc, dpus);
}

// ===----------------------------------------------------------------------===//
// IR emission template for tiled GEMV
// ===----------------------------------------------------------------------===//

/// Emit the DPU-side kernel for a tiled GEMV y += A*x.
///
/// MRAM layout (per DPU):  abuf[mramRows, mramCols], xbuf[mramCols],
/// ybuf[mramRows]. WRAM layout (shared):   abufW[tasklets, wramRows, wramCols],
///                         xbufW[wramCols], ybufW[tasklets, wramRows].
///
/// Each tasklet tid owns rows [tid*wramRows, (tid+1)*wramRows) per mr-tile.
/// Tasklet 0 loads the shared x slice; all tasklets load their own A slice.
/// Two barriers per mc iteration: one before t0 overwrites xbufW, one after
/// all loads complete.
upmem::DpuProgramOp createDpuGemvKernel(Location loc, RewriterBase &rewriter,
                                        ModuleOp target, int64_t mramRows,
                                        int64_t mramCols, int64_t wramRows,
                                        int64_t wramCols, int64_t tasklets,
                                        Type eltTy, llvm::StringRef &aBufSym,
                                        llvm::StringRef &xBufSym,
                                        llvm::StringRef &yBufSym) {

  rewriter.clearInsertionPoint();
  auto kernl = upmem::DpuProgramOp::create(rewriter, loc, "gemv", tasklets);
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

  // MRAM buffers (named so host scatter/gather can reference them).
  auto abufMram = upmem::StaticAllocOp::create(
      rewriter, loc,
      MemRefType::get({mramRows, mramCols}, eltTy, MemRefLayoutAttrInterface{},
                      mramMS),
      upmem::DpuMemSpace::MRAM, "bufa");
  aBufSym = *abufMram.getSymName();

  auto xbufMram = upmem::StaticAllocOp::create(
      rewriter, loc,
      MemRefType::get({mramCols}, eltTy, MemRefLayoutAttrInterface{}, mramMS),
      upmem::DpuMemSpace::MRAM, "bufx");
  xBufSym = *xbufMram.getSymName();

  auto ybufMram = upmem::StaticAllocOp::create(
      rewriter, loc,
      MemRefType::get({mramRows}, eltTy, MemRefLayoutAttrInterface{}, mramMS),
      upmem::DpuMemSpace::MRAM, "bufy");
  yBufSym = *ybufMram.getSymName();

  // WRAM buffers: A and y are per-tasklet slices; x is shared (t0 loads it).
  Value abufWram = upmem::StaticAllocOp::create(
                       rewriter, loc,
                       MemRefType::get({tasklets, wramRows, wramCols}, eltTy,
                                       MemRefLayoutAttrInterface{}, wramMS),
                       upmem::DpuMemSpace::WRAM)
                       .getBuffer();
  Value xbufWram = upmem::StaticAllocOp::create(
                       rewriter, loc,
                       MemRefType::get({wramCols}, eltTy,
                                       MemRefLayoutAttrInterface{}, wramMS),
                       upmem::DpuMemSpace::WRAM)
                       .getBuffer();
  Value ybufWram = upmem::StaticAllocOp::create(
                       rewriter, loc,
                       MemRefType::get({tasklets, wramRows}, eltTy,
                                       MemRefLayoutAttrInterface{}, wramMS),
                       upmem::DpuMemSpace::WRAM)
                       .getBuffer();

  Value tid = upmem::TaskletDimOp::create(rewriter, loc).getResult();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value isT0 =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, tid, zero);
  Value wramRowsCst = arith::ConstantIndexOp::create(rewriter, loc, wramRows);
  Value myRowBase = arith::MulIOp::create(rewriter, loc, tid, wramRowsCst);

  DenseI8ArrayAttr t0Attr = rewriter.getDenseI8ArrayAttr({0});

  // Precomputed result types for rank-reducing subviews.
  MemRefType myAWramTy = MemRefType::get(
      {wramRows, wramCols}, eltTy,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {wramCols, 1}), wramMS);
  MemRefType myYWramTy = MemRefType::get(
      {wramRows}, eltTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}),
      wramMS);
  MemRefType myYMramTy = MemRefType::get(
      {wramRows}, eltTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}),
      mramMS);
  MemRefType myXMramTy = MemRefType::get(
      {wramCols}, eltTy, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}),
      mramMS);
  // Bulk A tile covering every tasklet's slice at once (tasklet 0 loads it
  // in a single DMA rather than each tasklet issuing its own transfer).
  // abufWram[tasklets, wramRows, wramCols] is contiguous, so it is bit-
  // identical to a flat [tasklets*wramRows, wramCols] view.
  MemRefType bulkAMramTy = MemRefType::get(
      {tasklets * wramRows, wramCols}, eltTy,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {mramCols, 1}), mramMS);
  MemRefType bulkAWramTy =
      MemRefType::get({tasklets * wramRows, wramCols}, eltTy,
                      MemRefLayoutAttrInterface{}, wramMS);
  // Same idea for the y write-back: t0 flushes every tasklet's y slice in
  // one bulk DMA instead of each tasklet writing back its own.
  MemRefType bulkYMramTy = MemRefType::get(
      {tasklets * wramRows}, eltTy,
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1}), mramMS);
  MemRefType bulkYWramTy = MemRefType::get({tasklets * wramRows}, eltTy,
                                           MemRefLayoutAttrInterface{}, wramMS);

  // Per-tasklet WRAM slices: fixed for the lifetime of the kernel invocation.
  // abufWram[tasklets, wramRows, wramCols] → [wramRows, wramCols]
  Value myAWram = memref::SubViewOp::create(
      rewriter, loc, myAWramTy, abufWram,
      ArrayRef<OpFoldResult>{tid, rewriter.getIndexAttr(0),
                             rewriter.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(wramRows),
                             rewriter.getIndexAttr(wramCols)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(1)});
  // ybufWram[tasklets, wramRows] → [wramRows]
  Value myYWram = memref::SubViewOp::create(
      rewriter, loc, myYWramTy, ybufWram,
      ArrayRef<OpFoldResult>{tid, rewriter.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(wramRows)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(1)});

  auto elemAdd = [&](OpBuilder &b, Location loc, Value lhs,
                     Value rhs) -> Value {
    return eltTy.isIntOrIndex()
               ? arith::AddIOp::create(b, loc, lhs, rhs).getResult()
               : arith::AddFOp::create(b, loc, lhs, rhs).getResult();
  };
  auto elemMul = [&](OpBuilder &b, Location loc, Value lhs,
                     Value rhs) -> Value {
    return eltTy.isIntOrIndex()
               ? arith::MulIOp::create(b, loc, lhs, rhs).getResult()
               : arith::MulFOp::create(b, loc, lhs, rhs).getResult();
  };

  // mr loop: tasklets cooperate to cover all mramRows, each handling wramRows.
  cinm::createNestedAffineForLoops(
      rewriter, loc, {mramRows}, {tasklets * wramRows}, {},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value mr = ivs[0];
        Value rowOff = arith::AddIOp::create(b, loc, mr, myRowBase);

        // Load this tasklet's y slice from MRAM before accumulating.
        Value myYMram = memref::SubViewOp::create(
            b, loc, myYMramTy, ybufMram.getBuffer(),
            ArrayRef<OpFoldResult>{rowOff},
            ArrayRef<OpFoldResult>{b.getIndexAttr(wramRows)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        upmem::LocalTransferOp::create(b, loc, myYMram, myYWram);

        // mc loop: iterate over mramCols in wramCols chunks.
        cinm::createNestedAffineForLoops(
            b, loc, {mramCols}, {wramCols}, {},
            [&](OpBuilder &b, Location loc, ValueRange ivs2,
                ValueRange) -> SmallVector<Value> {
              Value mc = ivs2[0];

              // Barrier before t0 overwrites the shared xbufWram.
              upmem::BarrierOp::create(b, loc);

              // Tasklet 0 loads the x slice for this column tile.
              auto xIfOp = scf::IfOp::create(b, loc, TypeRange{}, isT0, false);
              xIfOp->setAttr("upmem_cm.const_tasklets", t0Attr);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(&xIfOp.getThenRegion().front());
                Value myXMram = memref::SubViewOp::create(
                    b, loc, myXMramTy, xbufMram.getBuffer(),
                    ArrayRef<OpFoldResult>{mc},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(wramCols)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                upmem::LocalTransferOp::create(b, loc, myXMram, xbufWram);

                // Tasklet 0 also loads the whole A tile (every tasklet's
                // slice) in one bulk DMA, instead of each tasklet issuing
                // its own transfer for its wramRows x wramCols slice.
                Value bulkAMram = memref::SubViewOp::create(
                    b, loc, bulkAMramTy, abufMram.getBuffer(),
                    ArrayRef<OpFoldResult>{mr, mc},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(tasklets * wramRows),
                                           b.getIndexAttr(wramCols)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                           b.getIndexAttr(1)});
                Value bulkAWram = memref::ReinterpretCastOp::create(
                    b, loc, bulkAWramTy, abufWram, /*offset=*/0,
                    ArrayRef<int64_t>{tasklets * wramRows, wramCols},
                    ArrayRef<int64_t>{wramCols, 1});
                upmem::LocalTransferOp::create(b, loc, bulkAMram, bulkAWram);
              }

              // Barrier: all loads (x and A, both by t0) must complete.
              upmem::BarrierOp::create(b, loc);

              // Compute: myY[i] += myA[i, j] * xbufWram[j]
              cinm::createNestedAffineForLoops(
                  b, loc, {wramRows, wramCols}, {1, 1}, {},
                  [&](OpBuilder &b, Location loc, ValueRange ivs3,
                      ValueRange) -> SmallVector<Value> {
                    Value i = ivs3[0], j = ivs3[1];
                    Value aij = memref::LoadOp::create(b, loc, myAWram,
                                                       ValueRange{i, j});
                    Value xj =
                        memref::LoadOp::create(b, loc, xbufWram, ValueRange{j});
                    Value yi =
                        memref::LoadOp::create(b, loc, myYWram, ValueRange{i});
                    memref::StoreOp::create(
                        b, loc, elemAdd(b, loc, yi, elemMul(b, loc, aij, xj)),
                        myYWram, ValueRange{i});
                    return {};
                  });

              return {};
            }); // end mc loop

        // Barrier: every tasklet must finish writing its y slice before t0
        // bulk-transfers all of them back to MRAM at once.
        upmem::BarrierOp::create(b, loc);

        auto yWritebackIf = scf::IfOp::create(b, loc, TypeRange{}, isT0, false);
        yWritebackIf->setAttr("upmem_cm.const_tasklets", t0Attr);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&yWritebackIf.getThenRegion().front());
          Value bulkYMram = memref::SubViewOp::create(
              b, loc, bulkYMramTy, ybufMram.getBuffer(),
              ArrayRef<OpFoldResult>{mr},
              ArrayRef<OpFoldResult>{b.getIndexAttr(tasklets * wramRows)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          Value bulkYWram = memref::ReinterpretCastOp::create(
              b, loc, bulkYWramTy, ybufWram, /*offset=*/0,
              ArrayRef<int64_t>{tasklets * wramRows}, ArrayRef<int64_t>{1});
          upmem::LocalTransferOp::create(b, loc, bulkYWram, bulkYMram);
        }
        return {};
      }); // end mr loop

  return kernl;
}

// When we know the buffer is not going to be read from,
// we can delete previous updates.
static void deleteUnusedUpdates(const Value aMemrefValue, Operation *stopAt) {
  // getUses() order reflects use-list construction order (newest first), not
  // program order, and it can include uses created *after* stopAt (e.g. a
  // later bufferization.to_tensor of the same buffer) -- those must be
  // ignored entirely rather than merely skipped, since they don't bound the
  // backward walk through the uses that actually precede stopAt. So filter
  // to same-block uses strictly before stopAt and sort explicitly by
  // position, walking from the one closest to stopAt backwards: as long as
  // each is a pure overwrite of the whole buffer, nothing before stopAt
  // reads it, so it's dead and can be erased.
  llvm::SmallVector<OpOperand *, 4> uses;
  for (auto &use : aMemrefValue.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == stopAt || owner->getBlock() != stopAt->getBlock() ||
        !owner->isBeforeInBlock(stopAt))
      continue;
    uses.push_back(&use);
  }
  llvm::sort(uses, [](OpOperand *a, OpOperand *b) {
    return b->getOwner()->isBeforeInBlock(a->getOwner());
  });
  for (auto *use : uses) {

    auto op = use->getOwner();
    if (auto fill = llvm::dyn_cast_or_null<linalg::FillOp>(op)) {
      auto inits = fill.getDpsInits();
      if (inits.size() == 1 && inits.front() == aMemrefValue) {
        fill->erase();
        continue;
      }
    }
    return; // give up
  }
}

/// Emit the host-side tiled loop nest for a GEMV out += lhs * rhs where
/// lhs is memref<M x K x elt>, rhs is memref<K x elt>, out is memref<M x elt>.
///
///   for m = 0 to M step dpuRows*mramRows:
///     fill yStage ← 0
///     for k = 0 to K step dpuCols*mramCols:
///       [pack x slice → xStage]
///       upmem.scatter A tile directly → @aBufSym
///       upmem.scatter xStage → @xBufSym
///       upmem.scatter yStage → @yBufSym   (running partial)
///       upmem.wait_for dpus               (DPU: y += A*x)
///       upmem.gather  yStage ← @yBufSym
///     [reduce yStage over dpuCols, add into out[m..]]
///
/// Matches the cost breakdown in simulateFullGemv().
void upmem::generateGemv(cinm::GemvOp op, RewriterBase &rewriter,
                         int64_t dpuRows, int64_t dpuCols, int64_t mramRows,
                         int64_t mramCols, int64_t wramRows, int64_t wramCols,
                         int64_t tasklets) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();

  auto lhsTy = cast<MemRefType>(op.getLhs().getType());
  Type eltTy = lhsTy.getElementType();
  const int64_t M = lhsTy.getShape()[0];
  const int64_t K = lhsTy.getShape()[1];
  int64_t numDpus = dpuRows * dpuCols;

  Value A = op.getLhs();
  Value x = op.getRhs();
  Value y = op.getOut();

  // Staging buffers on the host. A is scattered directly from `A`, no
  // staging (see scatterATile).
  // xStage: one [mramCols] slice per DPU column group (dpuCols groups).
  // yStage: one [mramRows] partial accumulator per DPU (2D DPU grid).
  // When dpuCols == 1 there is only one partial per DPU row, so it is
  // already the final result: instead of allocating a separate yStage and
  // reducing it into `y` afterwards, we scatter/gather directly into a
  // (expanded) subview of `y`, skipping the host-side reduction entirely.
  bool needsPartialReduction = dpuCols > 1;
  Value xStage = memref::AllocOp::create(
      rewriter, loc, MemRefType::get({dpuCols, mramCols}, eltTy));
  Value yStage;
  if (needsPartialReduction)
    yStage = memref::AllocOp::create(
        rewriter, loc, MemRefType::get({dpuRows, dpuCols, mramRows}, eltTy));
  else
    deleteUnusedUpdates(y, op);

  llvm::StringRef aBufSym, xBufSym, yBufSym;
  auto parentMod = op->getParentOfType<ModuleOp>();
  upmem::DpuProgramOp krnlOp = createDpuGemvKernel(
      loc, rewriter, parentMod, mramRows, mramCols, wramRows, wramCols,
      tasklets, eltTy, aBufSym, xBufSym, yBufSym);

  // Private constant zero-filled global with the shape of a single result tile
  // (<mramRows>), used to reset it every M tile via memref.copy (which lowers
  // to a memcpy) instead of linalg.fill (which lowers to an explicit loop
  // nest).
  auto yZeroTy = MemRefType::get({mramRows}, eltTy);
  memref::GlobalOp yZeroGlobal;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentMod.getBody());
    yZeroGlobal = memref::GlobalOp::create(
        rewriter, loc, "__cinm_gemv_yzero",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/yZeroTy,
        /*initial_value=*/
        DenseElementsAttr::get(RankedTensorType::get(yZeroTy.getShape(), eltTy),
                               rewriter.getZeroAttr(eltTy)),
        /*constant=*/true, /*alignment=*/IntegerAttr{});
    SymbolTable(parentMod).insert(yZeroGlobal);
  }

  // Insert the DPU alloc and host loop right before the op so that op.getOut()
  // (whose alloc precedes the op) is already in scope inside the M loop.
  rewriter.setInsertionPoint(op);
  Value yZero = memref::GetGlobalOp::create(rewriter, loc, yZeroTy,
                                            yZeroGlobal.getSymNameAttr());

  auto wgTy = upmem::DeviceHierarchyType::get(ctx, 1, numDpus, tasklets);
  auto dpuProgramSymbol = SymbolRefAttr::get(krnlOp.getSymNameAttr());
  auto dpus = upmem::AllocDPUsOp::create(rewriter, loc, wgTy, dpuProgramSymbol);

  // Scatter maps for hierarchy <1 x numDpus x tasklets> (numRanks=1, rank=0).
  // xStage: (rank, dpu) → (dpu % dpuCols, 0)   — same x for all DPU rows
  // yStage: (rank, dpu) → (dpu / dpuCols, dpu % dpuCols, 0)
  auto dpuDim = getAffineDimExpr(1, ctx);
  auto zeroExpr = getAffineConstantExpr(0, ctx);
  AffineMap xMap = AffineMap::get(2, 0, {dpuDim % dpuCols, zeroExpr}, ctx);
  AffineMap yMap = AffineMap::get(
      2, 0, {dpuDim.floorDiv(dpuCols), dpuDim % dpuCols, zeroExpr}, ctx);

  auto addKind = eltTy.isIntOrIndex() ? arith::AtomicRMWKind::addi
                                      : arith::AtomicRMWKind::addf;

  cinm::createNestedAffineForLoops(
      rewriter, loc, {M}, {dpuRows * mramRows}, {},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange) -> SmallVector<Value> {
        Value mOff = ivs[0];

        // the zero scatter map is a broadcast of the smaller tile
        AffineMap zeroScatterMap = AffineMap::get(2, 0, {zeroExpr}, ctx);

        // Reset the running y partials to zero for this M tile.
        upmem::ScatterOp::create(b, loc, yZero, yBufSym,
                                 static_cast<uint64_t>(mramRows),
                                 zeroScatterMap, dpus);

        cinm::createNestedAffineForLoops(
            b, loc, {K}, {dpuCols * mramCols}, {},
            [&](OpBuilder &b, Location loc, ValueRange ivs2,
                ValueRange) -> SmallVector<Value> {
              Value kOff = ivs2[0];

              scatterATile(b, loc, A, mOff, kOff, dpuRows, dpuCols, mramRows,
                           mramCols, aBufSym, dpus);
              Value xToScatter =
                  packXSlice(b, loc, x, xStage, kOff, dpuCols, mramCols);

              upmem::ScatterOp::create(b, loc, xToScatter, xBufSym,
                                       static_cast<uint64_t>(mramCols), xMap,
                                       dpus);
              upmem::WaitForOp::create(b, loc, dpus);

              return {};
            }); // end k loop

        // yBuf is the buffer scattered to / gathered from the DPUs for this
        // M tile: either the shared yStage allocation (dpuCols > 1), or a
        // 3D view directly into `y` (dpuCols == 1, see comment above).
        Value yBuf;
        if (needsPartialReduction) {
          yBuf = yStage;
        } else {
          Value outRows = memref::SubViewOp::create(
              b, loc, y, ArrayRef<OpFoldResult>{mOff},
              ArrayRef<OpFoldResult>{b.getIndexAttr(dpuRows * mramRows)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          yBuf = memref::ExpandShapeOp::create(
              b, loc,
              MemRefType::get({dpuRows, 1, mramRows}, eltTy,
                              StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                                                     {mramRows, mramRows, 1})),
              outRows, ArrayRef<ReassociationIndices>{{0, 1, 2}});
        }

        upmem::GatherOp::create(b, loc, yBuf, yBufSym,
                                static_cast<uint64_t>(mramRows), yMap, dpus);

        if (needsPartialReduction) {
          // Reduce yStage[dpuRows, dpuCols, mramRows] over dim 1 (dpuCols)
          // and accumulate into out[mOff .. mOff + dpuRows*mramRows).
          Value outRows = memref::SubViewOp::create(
              b, loc, y, ArrayRef<OpFoldResult>{mOff},
              ArrayRef<OpFoldResult>{b.getIndexAttr(dpuRows * mramRows)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          Value outRows2D = memref::ExpandShapeOp::create(
              b, loc,
              MemRefType::get({dpuRows, mramRows}, eltTy,
                              StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                                                     {mramRows, 1})),
              outRows, ArrayRef<ReassociationIndices>{{0, 1}});
          linalg::ReduceOp::create(
              b, loc, ValueRange{yBuf}, ValueRange{outRows2D},
              ArrayRef<int64_t>{1},
              [&](OpBuilder &b, Location loc, ValueRange args) {
                linalg::YieldOp::create(
                    b, loc,
                    arith::getReductionOp(addKind, b, loc, args[0], args[1]));
              });
        }

        return {};
      }); // end m loop

  rewriter.eraseOp(op);
  memref::DeallocOp::create(rewriter, loc, xStage);
  if (needsPartialReduction)
    memref::DeallocOp::create(rewriter, loc, yStage);
  upmem::FreeDPUsOp::create(rewriter, loc, dpus);
}

} // namespace mlir

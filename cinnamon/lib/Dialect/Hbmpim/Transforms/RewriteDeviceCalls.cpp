#include "cinm-mlir/Conversion/CommonPatterns.h"
// #include "cinm-mlir/Conversion/UPMEMPasses.h"
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimAttributes.h"
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.h"
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimTypes.h"
#include <cinm-mlir/Utils/CinmUtils.h>
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.h"
#include "cinm-mlir/Dialect/Hbmpim/Transforms/Passes.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/Casting.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>

#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace mlir {
#define GEN_PASS_DEF_HBMPIMREWRITEDEVICECALLSPASS
#include "cinm-mlir/Dialect/Hbmpim/Transforms/Passes.h.inc"
// #include "cinm-mlir/Conversion/HbmpimPasses.h.inc"
} // namespace mlir

namespace mlir::hbmpim{
namespace {
struct ElementwiseOpRewriter
    : public OpConversionPattern<hbmpim::ExecuteElementwiseOp> {
public:
  using OpConversionPattern<hbmpim::ExecuteElementwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hbmpim::ExecuteElementwiseOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // auto loc = op.getLoc();
    // int64_t totalPe = op.getConfig().getType().getNumElements();
    // int64_t numGrf = op.getConfig().getType().getNumGrf();
    // Value cst0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Value cst1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // Value cst2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    // Value totalPeVal = rewriter.create<arith::ConstantIndexOp>(loc, totalPe);
    // Value numTile = rewriter.create<arith::DivUIOp>(loc, op.getDim(), totalPeVal);
    // Value jumpsTobeTakenVal = rewriter.create<arith::SubIOp>(loc, numTile, cst1);
    // Value numGrfVal = rewriter.create<arith::ConstantIndexOp>(loc, numGrf);
    // Value cmds = rewriter.create<hbmpim::GetPimCmdsOp>(loc, 
    //     hbmpim::PIMCMDVecType::get(rewriter.getContext(), hbmpim::PimKernelType::ADD), 
    //     hbmpim::PimKernelType::ADD, jumpsTobeTakenVal, cst0, cst0);
    // Value toggleCond = rewriter.create<hbmpim::GetToggleCond>(loc, 
    //     rewriter.getIndexType(), hbmpim::PimBankType::ALL_BANK);
    // rewriter.create<hbmpim::SetControlOp>(loc, hbmpim::BurstType::bst_hab_pim_, 
    //     true, toggleCond, false, false);
    // rewriter.create<hbmpim::SetControlOp>(loc, hbmpim::BurstType::bst_hab_, 
    //     false, toggleCond, false, false);
    // rewriter.create<hbmpim::ParkInOp>(loc);
    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::SB,
    //     hbmpim::DRAMMode::HAB);
    // rewriter.create<hbmpim::ProgramCrfOp>(loc, cmds);
    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::HAB,
    //     hbmpim::DRAMMode::HAB_PIM);

    // auto savedPoint = rewriter.saveInsertionPoint();

    // scf::ForOp forTiles = rewriter.create<scf::ForOp>(loc, cst0, numTile, cst1);
    // rewriter.setInsertionPointToStart(forTiles.getBody());
    // Value c = rewriter.create<arith::MulIOp>(loc, forTiles.getInductionVar() , numGrfVal);
    // scf::ForOp forOddEvenBank = rewriter.create<scf::ForOp>(loc, cst0, cst1, cst2);
    // rewriter.setInsertionPointToStart(forOddEvenBank.getBody());
    // rewriter.create<hbmpim::AddTransactionAllOp>(loc, false, cst0, 
    //     forOddEvenBank.getInductionVar(), op.getInput0row(), c, 
    //     hbmpim::BurstType::null_bst_, true, numGrfVal);
    // rewriter.create<hbmpim::AddTransactionAllOp>(loc, false, cst0, 
    //     forOddEvenBank.getInductionVar(), op.getInput1row(), c, 
    //     hbmpim::BurstType::null_bst_, true, numGrfVal);
    // rewriter.create<hbmpim::AddTransactionAllOp>(loc, true, cst0, 
    //     forOddEvenBank.getInductionVar(), op.getResultRow(), c, 
    //     hbmpim::BurstType::null_bst_, true, numGrfVal);
        
    // rewriter.restoreInsertionPoint(savedPoint);
    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::HAB_PIM, 
    //     hbmpim::DRAMMode::HAB);
    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::HAB, 
    //     hbmpim::DRAMMode::SB);
    rewriter.eraseOp(op);
    return success();
  }
};

static void createAddTransactionAllInstructions(ImplicitLocOpBuilder &builder, Value numPimChan, Value numPimRank, bool isWrite, 
    Value bankgroupIdx, Value bankIdx, Value row, Value col, hbmpim::BurstType bst, bool use_barrier, Value numLoop, Value cst0, Value cst1){
  // scf::ForOp forChan = builder.create<scf::ForOp>(cst0, numPimChan, cst1);
  // builder.setInsertionPointToStart(forChan.getBody());
  // scf::ForOp forRank = builder.create<scf::ForOp>(cst0, numPimRank, cst1);
  // builder.setInsertionPointToStart(forRank.getBody());
  // scf::ForOp forLoop = builder.create<scf::ForOp>(cst0, numLoop, cst1, col);
  // builder.setInsertionPointToStart(forLoop.getBody());
  // auto localCol = forLoop.getRegionIterArg(0);
  // Value addr= builder.create<hbmpim::PimAddrGenSafeOp>(numPimChan, numPimRank,bankgroupIdx, bankIdx, row, localCol);
  // builder.create<hbmpim::AddTransactionOp>(isWrite, addr, bst);
  // Value incLocalCol = builder.create<arith::AddIOp>(localCol, cst1);
  // builder.create<scf::YieldOp>(incLocalCol);

}

static void createComputeGemvLoops(ImplicitLocOpBuilder &builder, Value numPimChan, Value numPimRank, 
  Value numGrf, Value numInputTile, Value numOutputTile, Value batchIdx, Value cst0, Value cst1) {

  // scf::ForOp forChan = builder.create<scf::ForOp>(cst0, numPimChan, cst1);
  // builder.setInsertionPointToStart(forChan.getBody());
  // scf::ForOp forRank = builder.create<scf::ForOp>(cst0, numPimRank, cst1);
  // builder.setInsertionPointToStart(forRank.getBody());
  // scf::ForOp forGrf = builder.create<scf::ForOp>(cst0, numGrf, cst1);
  // builder.setInsertionPointToStart(forGrf.getBody());
  // Value pim_reg_ra = builder.create<arith::ConstantIndexOp>(0x3fff);
  // // uint64_t addr =
  // //                   pim_addr_mgr_->addrGen(ch_idx, ra_idx, 0, 1, pim_reg_ra, 0x8 + gidx);
  // Value cst8 = builder.create<arith::ConstantIndexOp>(0x8);
  // Value offsetGidx = builder.create<arith::AddIOp>(forGrf.getInductionVar(), cst8);
  // Value addr = builder.create<hbmpim::PimAddrGenOp>(forChan.getInductionVar(), forRank.getInductionVar(), cst0, cst1, pim_reg_ra, offsetGidx);
  // // int input_idx =
  // //                   batchIdx * num_grfA_ * num_input_tiles + inputTile * num_grfA_ + gidx;
  // Value t1 = builder.create<arith::MulIOp>(batchIdx, numGrf);
  // Value t2 = builder.create<arith::MulIOp>(t1, numInputTile);
  // Value t3 = builder.create<arith::MulIOp>(numInputTile, numGrf);
  // Value t4 = builder.create<arith::AddIOp>(t2, t3);
  // Value input_idx = builder.create<arith::MulIOp>(t4, forGrf.getInductionVar());
  // // mem_->addTransaction(true, addr, str, &data->bData[input_idx]);
  // builder.create<hbmpim::AddTransactionOp>(true, addr, input_idx);
}

struct ExecuteGemvOpRewriter
    : public OpConversionPattern<hbmpim::ExecuteGemvOp> {
public:
  using OpConversionPattern<hbmpim::ExecuteGemvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hbmpim::ExecuteGemvOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // int64_t numGrf = op.getNumGrf();
    // Value numBankVal = rewriter.create<arith::ConstantIndexOp>(loc, op.getNumBank());
    // Value numChannelVal = rewriter.create<arith::ConstantIndexOp>(loc, op.getNumChannel());
    // Value numRankVal = rewriter.create<arith::ConstantIndexOp>(loc, op.getNumRank());
    // Value numGrfVal = rewriter.create<arith::ConstantIndexOp>(loc, numGrf);
    // Value cst0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Value cst1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // Value cst2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    // Value numInputTiles = op.getNumInputTile();
    // Value numOutputTiles = op.getNumOutputTile();
    // Value numBatch = op.getBatch();
    // Value temp1 = rewriter.create<arith::SubIOp>(loc, numInputTiles, cst1);
    // Value temp2 = rewriter.create<arith::CeilDivUIOp>(loc, numInputTiles, cst2);
    // Value temp3 = rewriter.create<arith::MulIOp>(loc, temp2, numGrfVal);
    // Value numJumpEvenBank = rewriter.create<arith::SubIOp>(loc, temp3, cst1);

    // Value temp4 = rewriter.create<arith::DivUIOp>(loc, numInputTiles, cst2);
    // Value temp5 = rewriter.create<arith::MulIOp>(loc, temp4, numGrfVal);
    // Value numJumpOddBank = rewriter.create<arith::SubIOp>(loc, temp5, cst1);

    // Value cmds = rewriter.create<hbmpim::GetPimCmdsOp>(loc, 
    //     hbmpim::PIMCMDVecType::get(rewriter.getContext(), hbmpim::PimKernelType::GEMV), 
    //     hbmpim::PimKernelType::GEMV, cst0, numJumpOddBank, numJumpEvenBank);
    // Value toggleCond = rewriter.create<hbmpim::GetToggleCondOp>(loc, 
    //     rewriter.getIndexType(), hbmpim::PimBankType::ALL_BANK);
    // rewriter.create<hbmpim::SetControlOp>(loc, hbmpim::BurstType::bst_hab_pim_, 
    //     true, toggleCond, false, true);
    // rewriter.create<hbmpim::ParkInOp>(loc);

    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::SB,
    //     hbmpim::DRAMMode::HAB);
    // rewriter.create<hbmpim::ProgramCrfOp>(loc, cmds);


    // auto savedPoint1 = rewriter.saveInsertionPoint();

    // scf::ForOp forOutputTiles = rewriter.create<scf::ForOp>(loc, cst0, numOutputTiles, cst1);
    // rewriter.setInsertionPointToStart(forOutputTiles.getBody());
    // scf::ForOp forBatchs = rewriter.create<scf::ForOp>(loc, cst0, numBatch, cst1);
    // rewriter.setInsertionPointToStart(forBatchs.getBody());
    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::HAB,
    //     hbmpim::DRAMMode::HAB_PIM);

    // // int col = num_output_tiles * num_input_tiles / 2 * num_grfA_ * num_grfB_ + (j + b) * num_grfB_;
    // Value t1 = rewriter.create<arith::MulIOp>(loc, numOutputTiles, numInputTiles);
    // Value t2 = rewriter.create<arith::MulIOp>(loc, cst2, numGrfVal);
    // Value t3 = rewriter.create<arith::MulIOp>(loc, t2, numGrfVal);
    // Value t4 = rewriter.create<arith::DivUIOp>(loc, t1, t3);
    // Value t5 = rewriter.create<arith::AddIOp>(loc, forOutputTiles.getInductionVar(), forBatchs.getInductionVar());
    // Value t6 = rewriter.create<arith::MulIOp>(loc, t5, numGrfVal);
    // Value col = rewriter.create<arith::AddIOp>(loc, t6, t4);
    // ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    // auto savedPoint2 = rewriter.saveInsertionPoint();
    // scf::ForOp forInputTiles1 = rewriter.create<scf::ForOp>(loc, cst0, numInputTiles, cst2);
    // builder.setInsertionPointToStart(forInputTiles1.getBody());
    // createComputeGemvLoops(builder, numChannelVal, numRankVal, numGrfVal, 
    //   numInputTiles, numOutputTiles, forBatchs.getInductionVar(), cst0, cst1);

    // scf::ForOp forInputTiles2 = rewriter.create<scf::ForOp>(loc, cst1, numInputTiles, cst2);
    // builder.setInsertionPointToStart(forInputTiles2.getBody());
    // createComputeGemvLoops(builder, numChannelVal, numRankVal, numGrfVal, 
    //   numInputTiles, numOutputTiles, forBatchs.getInductionVar(), cst0, cst1);
    // rewriter.restoreInsertionPoint(savedPoint2);
    // createAddTransactionAllInstructions(builder, numChannelVal, numRankVal, true, 
    //   cst0, cst1, cst0, col, hbmpim::BurstType::null_bst_, true, numGrfVal, cst0, cst1);



    // rewriter.create<hbmpim::AddTransactionAllOp>(loc, true, cst0, cst1, 
    //   cst0, col, hbmpim::BurstType::null_bst_, true, numGrfVal);
    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::HAB_PIM, 
    //     hbmpim::DRAMMode::HAB);
    // // Value c = rewriter.create<arith::MulIOp>(loc, forTiles.getInductionVar() , numGrfVal);
    // // scf::ForOp forOddEvenBank = rewriter.create<scf::ForOp>(loc, cst0, cst1, cst2);
    // rewriter.restoreInsertionPoint(savedPoint1);
    // rewriter.create<hbmpim::ChangePIMModeOp>(loc, hbmpim::DRAMMode::HAB, 
    //     hbmpim::DRAMMode::SB);
    // rewriter.create<hbmpim::ParkOutOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};


} // namespace


void populateDeviceRewritePatterns(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
    // patterns.add<ElementwiseOpRewriter>(patterns.getContext());
    patterns.add<ExecuteGemvOpRewriter>(patterns.getContext());

}

struct HbmpimRewriteDeviceCallsPass
    : public impl::HbmpimRewriteDeviceCallsPassBase<HbmpimRewriteDeviceCallsPass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    TypeConverter converter;
    // converter.addSourceMaterialization(addUnrealizedCast);
    // converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateDeviceRewritePatterns(converter, patterns);

    ConversionTarget target(getContext());
    target.addIllegalOp<hbmpim::ExecuteElementwiseOp>();
    target.addIllegalOp<hbmpim::ExecuteGemvOp>();
    

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createHbmpimRewriteDeviceCallsPassPass() {
  return std::make_unique<HbmpimRewriteDeviceCallsPass>();
}

} // namespace mlir::hbmpim

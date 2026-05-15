#include "cinm-mlir/Conversion/MemristorToFunc/MemristorToFunc.h"

#include "cinm-mlir/Dialect/Memristor/IR/MemristorDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/Support/Casting.h>
#include <memory>
#include <mlir/IR/BuiltinTypes.h>

#define DEBUG_TYPE "mlir-memristor-to-func"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::memristor;

namespace mlir::memristor {
#define GEN_PASS_DEF_CONVERTMEMRISTORTOFUNC
#include <cinm-mlir/Conversion/MemristorPasses.h.inc>
} // namespace mlir::memristor

namespace {
template <typename MemristorOp>
static LogicalResult createLibraryCall(MemristorOp &op,
                                       PatternRewriter &rewriter) {
  auto fnName = op.getLibraryCallName();
  if (fnName.empty()) {
    op->emitError("No library call defined for: ") << *op;
    return failure();
  }

  auto fnNameAttr = SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto parentModule = op->template getParentOfType<ModuleOp>();

  SmallVector<Type, 4> parameterTypes{};
  SmallVector<Value, 4> parameters{};

  // Make any memref operands unranked.
  for (auto operand : op->getOperands()) {
    auto operandType = operand.getType();
    if (auto memrefType = llvm::dyn_cast_or_null<MemRefType>(operandType)) {
      auto shape = memrefType.getShape().vec();
      std::fill(shape.begin(), shape.end(), ShapedType::kDynamic);
      auto dynamicMemrefType =
          MemRefType::get(shape, memrefType.getElementType());

      auto unrankedMemref = memref::CastOp::create(rewriter, op.getLoc(),
                                                   dynamicMemrefType, operand);

      parameterTypes.push_back(dynamicMemrefType);
      parameters.push_back(unrankedMemref.getResult());
    } else {
      parameterTypes.push_back(operandType);
      parameters.push_back(operand);
    }
  }

  if (!parentModule.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto insertionPoint = rewriter.saveInsertionPoint();

    auto moduleEnd = std::prev(parentModule.getBody()->end());
    rewriter.setInsertionPoint(parentModule.getBody(), moduleEnd);

    auto libFnType = FunctionType::get(rewriter.getContext(), parameterTypes,
                                       op->getResultTypes());

    // Insert before module terminator.
    auto funcOp = FuncOp::create(rewriter, op->getLoc(), fnNameAttr.getValue(),
                                 libFnType, ArrayRef<NamedAttribute>{});
    funcOp.setVisibility(FuncOp::Visibility::Nested);

    rewriter.restoreInsertionPoint(insertionPoint);
  }

  rewriter.replaceOpWithNewOp<CallOp>(op, fnNameAttr.getValue(),
                                      op->getResultTypes(), parameters);

  return success();
}

template <typename MemristorOp>
class MemristorOpConversion : public OpRewritePattern<MemristorOp> {
public:
  using OpRewritePattern<MemristorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemristorOp op,
                                PatternRewriter &rewriter) const override {
    return createLibraryCall(op, rewriter);
  }
};

} // anonymous namespace

void mlir::memristor::populateMemristorToFuncConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<MemristorOpConversion<memristor::WriteToCrossbarOp>,
                  MemristorOpConversion<memristor::GemmOp>,
                  MemristorOpConversion<memristor::GevmOp>,
                  MemristorOpConversion<memristor::BarrierOp>>(ctx);
}

namespace mlir::memristor {

class ConvertMemristorToFunc
    : public mlir::memristor::impl::ConvertMemristorToFuncBase<
          ConvertMemristorToFunc> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns{&getContext()};
    populateMemristorToFuncConversionPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](auto *) { return true; });
    target.addIllegalDialect<MemristorDialect>();
    target.addLegalDialect<FuncDialect>();
    target.addLegalOp<FuncOp>();

    Operation *operation = getOperation();
    FrozenRewritePatternSet frozenPatterns{std::move(patterns)};
    if (failed(applyPartialConversion(operation, target, frozenPatterns)))
      signalPassFailure();
  }
};

} // namespace mlir::memristor

std::unique_ptr<Pass> memristor::createConvertMemristorToFuncPass() {
  return std::make_unique<ConvertMemristorToFunc>();
}
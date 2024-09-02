#include "cinm-mlir/Conversion/MemristorToFunc/MemristorToFunc.h"

#include "cinm-mlir/Dialect/Memristor/IR/MemristorDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "mlir-memristor-to-func"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::memristor;

#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/MemristorPasses.h.inc>

namespace {
// Get a SymbolRefAttr containing the library function name for the MemristorOp.
// If the library function does not exist, insert a declaration.
template <typename MemristorOp>
static FlatSymbolRefAttr getLibraryCallSymbolRef(Operation *op,
                                                 PatternRewriter &rewriter) {
  auto memristorOp = cast<MemristorOp>(op);
  auto fnName = memristorOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return {};
  }

  // fnName is a dynamic std::String, unique it via a SymbolRefAttr.
  auto fnNameAttr = SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes(op->getOperandTypes());
  auto libFnType = FunctionType::get(rewriter.getContext(), inputTypes,
                                     op->getResultTypes());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  auto funcOp = rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(),
                                        libFnType, ArrayRef<NamedAttribute>{});
  funcOp.setVisibility(FuncOp::Visibility::Nested);

  return fnNameAttr;
}

// MemristorOpConversion<MemristorOp> creates a new call to the
// `MemristorOp::getLibraryCallName()` function.
// The implementation of the function can be either in the same module or in an
// externally linked library.
template <typename MemristorOp>
class MemristorOpConversion : public OpRewritePattern<MemristorOp> {
public:
  using OpRewritePattern<MemristorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemristorOp op,
                                PatternRewriter &rewriter) const override {
    auto libraryCallName = getLibraryCallSymbolRef<MemristorOp>(op, rewriter);
    if (!libraryCallName)
      return failure();

    Operation *memristorOp = op;
    rewriter.replaceOpWithNewOp<CallOp>(memristorOp, libraryCallName.getValue(),
                                        memristorOp->getResultTypes(),
                                        memristorOp->getOperands());

    return success();
  }
};

class ConvertMemristorToFunc
    : public ConvertMemristorToFuncBase<ConvertMemristorToFunc> {
public:
  void runOnOperation() override {

    RewritePatternSet patterns{&getContext()};
    populateMemristorToFuncConversionPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addIllegalDialect<MemristorDialect>();
    target.addLegalDialect<FuncDialect>();
    target.addLegalOp<FuncOp>();

    Operation *operation = getOperation();
    FrozenRewritePatternSet frozenPatterns{std::move(patterns)};
    if (failed(applyPartialConversion(operation, target, frozenPatterns)))
      signalPassFailure();
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

std::unique_ptr<Pass> mlir::memristor::createConvertMemristorToFuncPass() {
  return std::make_unique<ConvertMemristorToFunc>();
}

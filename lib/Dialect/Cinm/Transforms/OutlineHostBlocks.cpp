//===- OutlineHostBlocks.cpp - Hand host blocks to another compiler -------===//
//
// Moves the bodies of the compute blocks pinned to the host into functions
// of a separate module, leaving a call in their place. The device side of
// the program stays with this compiler; the outlined module is meant for a
// host compiler that does the job properly (vectorization, threading,
// buffer reuse), and the two are linked back together through the private
// prototypes this pass leaves in the enclosing module.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMOUTLINEHOSTBLOCKSPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// Whether `op` is a compute op that only the host may run.
bool isHostBlock(Operation *op) {
  if (!isa<ComputeOp, ComputeBlockOp>(op))
    return false;
  if (op->getAttr("accelerator"))
    return false;
  auto platforms =
      op->getAttrOfType<ArrayAttr>(CinmDialect::AVAILABLE_PLATFORMS_NAME);
  return platforms && !platforms.empty() &&
         llvm::all_of(platforms, llvm::IsaPred<HostPlatformAttr>);
}

/// Whether a value used from above is recomputed inside the outlined
/// function rather than passed to it.
bool isClonedIn(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getNumOperands() != 0 || def->getNumRegions() != 0)
    return false;
  return def->hasTrait<OpTrait::ConstantLike>() || isa<tensor::EmptyOp>(def);
}

struct OutlineHostBlocksPass
    : public impl::CinmOutlineHostBlocksPassBase<OutlineHostBlocksPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ModuleOp outlined = getOrCreateOutlinedModule(module);

    SmallVector<Operation *> blocks;
    module->walk([&](Operation *op) {
      if (op->getParentOfType<ModuleOp>() == module && isHostBlock(op))
        blocks.push_back(op);
    });

    llvm::StringMap<unsigned> counters;
    for (Operation *op : blocks)
      if (failed(outline(op, module, outlined, counters)))
        return signalPassFailure();

    if (outlinedFile.empty())
      return;
    std::error_code ec;
    llvm::raw_fd_ostream os(outlinedFile, ec);
    if (ec) {
      module.emitError() << "cannot write the outlined module to '"
                         << outlinedFile << "': " << ec.message();
      return signalPassFailure();
    }
    outlined.print(os);
    os << '\n';
    outlined.erase();
  }

private:
  ModuleOp getOrCreateOutlinedModule(ModuleOp module) {
    for (auto nested : module.getBody()->getOps<ModuleOp>())
      if (nested.getSymName() == StringRef(moduleName))
        return nested;
    auto outlined = ModuleOp::create(module.getLoc(), StringRef(moduleName));
    module.getBody()->push_back(outlined);
    return outlined;
  }

  /// A symbol name free in both modules, derived from the function `op` is
  /// in.
  std::string freshName(Operation *op, ModuleOp module, ModuleOp outlined,
                        llvm::StringMap<unsigned> &counters) {
    auto parent = op->getParentOfType<func::FuncOp>();
    std::string base = (parent ? parent.getSymName() : "main").str() + "_host";
    while (true) {
      std::string name = base + std::to_string(counters[base]++);
      if (!SymbolTable::lookupSymbolIn(module, name) &&
          !SymbolTable::lookupSymbolIn(outlined, name))
        return name;
    }
  }

  /// Copies into `outlined` every function the ops of `body` reference,
  /// and the functions those reference in turn.
  LogicalResult copyCallees(Block &body, Operation *anchor, ModuleOp outlined) {
    SmallVector<Block *> scan{&body};
    OpBuilder builder(outlined.getBodyRegion());
    while (!scan.empty()) {
      Block *block = scan.pop_back_val();
      LogicalResult result = success();
      block->walk([&](Operation *inner) {
        inner->getAttrDictionary().walk([&](SymbolRefAttr ref) {
          if (SymbolTable::lookupSymbolIn(outlined, ref.getRootReference()))
            return;
          Operation *symbol = SymbolTable::lookupNearestSymbolFrom(anchor, ref);
          auto callee = llvm::dyn_cast_or_null<func::FuncOp>(symbol);
          if (!callee) {
            inner->emitError() << "references " << ref
                               << ", which is not a function this pass can "
                                  "copy into the outlined module";
            result = failure();
            return;
          }
          builder.setInsertionPointToEnd(outlined.getBody());
          auto copy = cast<func::FuncOp>(builder.clone(*callee));
          if (!copy.isExternal())
            scan.push_back(&copy.getBody().front());
        });
      });
      if (failed(result))
        return failure();
    }
    return success();
  }

  LogicalResult outline(Operation *op, ModuleOp module, ModuleOp outlined,
                        llvm::StringMap<unsigned> &counters) {
    Region &region = op->getRegion(0);
    Block &body = region.front();
    Operation *yield = body.getTerminator();

    // What the call passes: the region's own arguments (a compute_block's),
    // then whatever the body reaches for from above.
    SmallVector<Value> passed(body.getArguments());
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(region, captured);
    SmallVector<Value> cloned;
    for (Value v : captured)
      (isClonedIn(v) ? cloned : passed).push_back(v);

    MLIRContext *ctx = op->getContext();
    auto type = FunctionType::get(ctx, ValueRange(passed).getTypes(),
                                  yield->getOperandTypes());
    std::string name = freshName(op, module, outlined, counters);
    Location loc = op->getLoc();

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(outlined.getBody());
    auto fn = func::FuncOp::create(builder, loc, name, type);
    Block *entry = fn.addEntryBlock();
    IRMapping mapping;
    mapping.map(passed, entry->getArguments());
    builder.setInsertionPointToStart(entry);
    for (Value v : cloned)
      builder.clone(*v.getDefiningOp(), mapping);
    for (Operation &inner : body.without_terminator())
      builder.clone(inner, mapping);
    func::ReturnOp::create(
        builder, loc, llvm::map_to_vector(yield->getOperands(), [&](Value v) {
          return mapping.lookupOrDefault(v);
        }));
    // A destination pin is advice to this compiler's bufferization; the
    // host compiler plans its own buffers, and on tensors the op's value is
    // just its source.
    fn.walk([](bufferization::MaterializeInDestinationOp pin) {
      if (pin->getNumResults() == 0)
        return;
      pin.getResult().replaceAllUsesWith(pin.getSource());
      pin.erase();
    });

    if (failed(copyCallees(fn.getBody().front(), op, outlined)))
      return failure();

    builder.setInsertionPointToEnd(module.getBody());
    auto decl = func::FuncOp::create(builder, loc, name, type);
    decl.setPrivate();

    // The body is replaced by the call. Its ops only use each other forward,
    // so erasing from the back never leaves a dangling use.
    for (Operation &inner : llvm::make_early_inc_range(llvm::reverse(body)))
      inner.erase();
    builder.setInsertionPointToEnd(&body);
    auto call =
        func::CallOp::create(builder, loc, name, type.getResults(), passed);
    YieldOp::create(builder, loc, call.getResults());
    return success();
  }
};

} // namespace
} // namespace mlir::cinm

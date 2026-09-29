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
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMOUTLINEHOSTBLOCKSPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// Whether a value used from above is recomputed inside the outlined
/// function rather than passed to it.
bool isClonedIn(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getNumOperands() != 0 || def->getNumRegions() != 0)
    return false;
  return def->hasTrait<OpTrait::ConstantLike>() || isa<tensor::EmptyOp>(def);
}

/// A compute op no accelerator was chosen for: host code.
bool isUnplaced(Operation *op) {
  return isa<ComputeOp, ComputeBlockOp>(op) && !op->getAttr("accelerator");
}

/// Whether the boundary of `op` has a C form (see callableAbi): memrefs of
/// static shape and strides, memref results that are arguments written in
/// place. Checked on what outlining passes: the region's arguments, then
/// the values used from above.
bool hasCForm(Operation *op) {
  Region &region = op->getRegion(0);
  llvm::SetVector<Value> captured;
  getUsedValuesDefinedAbove(region, captured);
  auto staticMemref = [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    SmallVector<int64_t> strides;
    int64_t offset;
    return !memref || (memref.hasStaticShape() &&
                       succeeded(memref.getStridesAndOffset(strides, offset)) &&
                       llvm::none_of(strides, ShapedType::isDynamic));
  };
  for (Value v : llvm::concat<Value>(region.getArguments(), captured))
    if (!staticMemref(v.getType()))
      return false;
  Block &body = region.front();
  for (Value v : body.getTerminator()->getOperands())
    if (isa<MemRefType>(v.getType()) &&
        !(isa<BlockArgument>(v) && cast<BlockArgument>(v).getOwner() == &body))
      return false;
  return true;
}

struct OutlineHostBlocksPass
    : public impl::CinmOutlineHostBlocksPassBase<OutlineHostBlocksPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!manifestFile.empty() && !unplaced) {
      module.emitError("manifest-file describes the C boundary of unplaced");
      return signalPassFailure();
    }
    ModuleOp outlined = getOrCreateOutlinedModule(module);

    SmallVector<Operation *> blocks;
    module->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getParentOfType<ModuleOp>() != module)
        return WalkResult::advance();
      if (unplaced ? isUnplaced(op) : isHostComputeOp(op)) {
        // Without a C form, a block stays here with our code.
        if (!unplaced || hasCForm(op))
          blocks.push_back(op);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });

    llvm::StringMap<unsigned> counters;
    for (Operation *op : blocks)
      if (failed(outline(op, module, outlined, counters)))
        return signalPassFailure();

    if (!manifestFile.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(manifestFile, ec);
      if (ec) {
        module.emitError() << "cannot write the manifest to '" << manifestFile
                           << "': " << ec.message();
        return signalPassFailure();
      }
      os << llvm::formatv("{0:1}", llvm::json::Value(std::move(manifest)))
         << '\n';
    }

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
  /// Per outlined function, its entry in the manifest.
  llvm::json::Array manifest;

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

    llvm::json::Array params, results;
    if (unplaced && failed(callableAbi(fn, decl, call, params, results)))
      return failure();
    if (!manifestFile.empty()) {
      llvm::json::Object entry{{"name", name},
                               {"params", std::move(params)},
                               {"results", std::move(results)}};
      if (auto alloc = op->getAttrOfType<DictionaryAttr>(
              CinmDialect::GRAPH_ALLOC_NAME)) {
        if (auto graph = alloc.getAs<StringAttr>("graph"))
          entry["graph"] = graph.getValue();
        if (auto cls = alloc.getAs<IntegerAttr>("class"))
          entry["class"] = cls.getInt();
        if (auto member = alloc.getAs<IntegerAttr>("member"))
          entry["member"] = member.getInt();
      }
      manifest.push_back(std::move(entry));
    }
    return success();
  }

  static std::string typeString(Type type) {
    std::string text;
    llvm::raw_string_ostream os(text);
    type.print(os);
    return text;
  }

  /// Makes the boundary between `call` and `fn` (declared by `decl`)
  /// callable from C under the bare-pointer convention, which needs static
  /// strides and offsets: a memref with a dynamic offset becomes its view at
  /// offset 0 plus the offset as an index, and a memref result, the argument
  /// written in place, is dropped for that argument. Describes the outcome
  /// in `params` and `results` (`in_place`: the original argument).
  LogicalResult callableAbi(func::FuncOp fn, func::FuncOp decl,
                            func::CallOp call, llvm::json::Array &params,
                            llvm::json::Array &results) {
    Block &entry = fn.getBody().front();
    auto ret = cast<func::ReturnOp>(fn.getBody().back().getTerminator());
    OpBuilder builder(call);
    OpBuilder inside = OpBuilder::atBlockBegin(&entry);
    Location loc = call.getLoc();
    IndexType indexType = builder.getIndexType();

    // Before the arguments are rewritten, which replaces their uses in the
    // return too.
    SmallVector<Value> kept;
    llvm::DenseMap<unsigned, unsigned> dropped; // result -> argument
    for (auto [ri, value] : llvm::enumerate(ret.getOperands())) {
      if (!isa<MemRefType>(value.getType())) {
        results.push_back(llvm::json::Object{
            {"kind", "scalar"}, {"type", typeString(value.getType())}});
        kept.push_back(value);
        continue;
      }
      auto written = dyn_cast<BlockArgument>(value);
      if (!written || written.getOwner() != &entry)
        return fn.emitError() << "result " << ri
                              << " is not one of the arguments; only a "
                                 "result written in place has a C form";
      dropped[ri] = written.getArgNumber();
      results.push_back(llvm::json::Object{
          {"kind", "memref"}, {"in_place", written.getArgNumber()}});
    }

    SmallVector<Value> operands;
    SmallVector<BlockArgument> args(entry.getArguments());
    for (auto [i, arg] : llvm::enumerate(args)) {
      Value operand = call.getOperand(i);
      auto memref = dyn_cast<MemRefType>(arg.getType());
      if (!memref) {
        params.push_back(
            llvm::json::Object{{"kind", "scalar"},
                               {"arg", i},
                               {"type", typeString(arg.getType())}});
        operands.push_back(operand);
        continue;
      }
      SmallVector<int64_t> strides;
      int64_t offset;
      if (!memref.hasStaticShape() ||
          failed(memref.getStridesAndOffset(strides, offset)) ||
          llvm::any_of(strides, ShapedType::isDynamic))
        return fn.emitError() << "argument " << i << " of type " << memref
                              << " has no C form: its shape and strides "
                                 "must be static";
      llvm::json::Object param{{"kind", "memref"},
                               {"arg", i},
                               {"dtype", typeString(memref.getElementType())},
                               {"shape", llvm::json::Array(memref.getShape())},
                               {"strides", llvm::json::Array(strides)}};
      if (!ShapedType::isDynamic(offset)) {
        param["offset"] = offset;
        params.push_back(std::move(param));
        operands.push_back(operand);
        continue;
      }
      auto atZero = MemRefType::get(
          memref.getShape(), memref.getElementType(),
          StridedLayoutAttr::get(memref.getContext(), 0, strides),
          memref.getMemorySpace());
      auto metadata =
          memref::ExtractStridedMetadataOp::create(builder, loc, operand);
      Value view = memref::ReinterpretCastOp::create(
          builder, loc, atZero, metadata.getBaseBuffer(), 0, memref.getShape(),
          strides);
      operands.push_back(view);
      operands.push_back(metadata.getOffset());
      param["offset"] = "next";
      params.push_back(std::move(param));
      params.push_back(llvm::json::Object{
          {"kind", "offset"}, {"arg", i}, {"type", "index"}});

      arg.setType(atZero);
      Value offsetArg =
          entry.insertArgument(arg.getArgNumber() + 1, indexType, arg.getLoc());
      auto rebuilt = memref::ReinterpretCastOp::create(
          inside, loc, memref, arg, OpFoldResult(offsetArg),
          getAsIndexOpFoldResult(fn.getContext(), memref.getShape()),
          getAsIndexOpFoldResult(fn.getContext(), strides));
      arg.replaceAllUsesExcept(rebuilt.getResult(), rebuilt);
    }

    // The call's operands follow the entry block's arguments one to one.
    auto newCall = func::CallOp::create(builder, loc, fn.getSymName(),
                                        ValueRange(kept).getTypes(), operands);
    unsigned next = 0;
    for (auto [ri, result] : llvm::enumerate(call.getResults())) {
      if (auto it = dropped.find(ri); it != dropped.end()) {
        result.replaceAllUsesWith(call.getOperand(it->second));
        continue;
      }
      result.replaceAllUsesWith(newCall.getResult(next++));
    }
    call.erase();
    OpBuilder at(ret);
    func::ReturnOp::create(at, ret.getLoc(), kept);
    ret.erase();

    auto type = FunctionType::get(fn.getContext(), entry.getArgumentTypes(),
                                  ValueRange(kept).getTypes());
    fn.setType(type);
    decl.setType(type);
    return success();
  }
};

} // namespace
} // namespace mlir::cinm

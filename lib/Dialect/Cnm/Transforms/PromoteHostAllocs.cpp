//===- PromoteHostAllocs.cpp - Reuse a function's scratch buffers --------===//
//
// Implements `--cnm-promote-host-allocs`; see the pass description in
// Passes.td for why a fresh allocation per call is slower than the transfer
// that fills it.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMPROMOTEHOSTALLOCSPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;

namespace {

/// Whether the buffer `v` names, or a view of it, can outlive the call or
/// reach code that could keep it: returned or yielded out of a region, passed
/// to a call, or turned into a raw pointer. Everything else -- loads, stores,
/// copies, transfers, compute ops reading or writing it -- uses it in place.
static bool escapes(Value v) {
  for (Operation *user : v.getUsers()) {
    if (user->hasTrait<OpTrait::IsTerminator>() || isa<CallOpInterface>(user) ||
        isa<memref::ExtractAlignedPointerAsIndexOp>(user))
      return true;
    if (auto view = dyn_cast<ViewLikeOpInterface>(user);
        view && view.getViewSource() == v) {
      for (Value result : user->getResults())
        if (escapes(result))
          return true;
      continue;
    }
    // An op that yields a memref it was handed may be returning a view of
    // it under another name; one it knows nothing about keeps it.
    if (llvm::any_of(user->getResultTypes(), llvm::IsaPred<BaseMemRefType>))
      return true;
  }
  return false;
}

struct CnmPromoteHostAllocsPass
    : public cnm::impl::CnmPromoteHostAllocsPassBase<CnmPromoteHostAllocsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<memref::AllocOp> allocs;
    module.walk([&](func::FuncOp func) {
      if (func.isExternal())
        return;
      // At the function's top level only: an allocation inside a loop may be
      // live in several iterations at once, and one slot cannot hold them.
      for (Block &block : func.getBody())
        for (auto alloc : block.getOps<memref::AllocOp>())
          if (alloc.getType().hasStaticShape() && !escapes(alloc))
            allocs.push_back(alloc);
    });

    OpBuilder b(&getContext());
    for (memref::AllocOp alloc : allocs) {
      MemRefType type = alloc.getType();
      b.setInsertionPointToStart(module.getBody());
      IntegerAttr alignment =
          alloc.getAlignment()
              ? b.getI64IntegerAttr(static_cast<int64_t>(*alloc.getAlignment()))
              : IntegerAttr{};
      auto global = memref::GlobalOp::create(
          b, alloc.getLoc(), getUniqueFunctionName(module, "__cnm_scratch_"),
          /*sym_visibility=*/b.getStringAttr("private"), type,
          /*initial_value=*/Attribute{},
          /*constant=*/false, alignment);
      b.setInsertionPoint(alloc);
      Value slot = memref::GetGlobalOp::create(b, alloc.getLoc(), type,
                                               global.getSymNameAttr());
      // A dealloc of the buffer would free the global.
      for (Operation *user : llvm::make_early_inc_range(alloc->getUsers()))
        if (isa<memref::DeallocOp>(user))
          user->erase();
      alloc.replaceAllUsesWith(slot);
      alloc.erase();
    }
  }
};

} // namespace

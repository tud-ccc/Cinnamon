//===- ScatterOptimizations.cpp - Shrink redundant cnm.scatter transfers -===//
//
// A cnm.scatter's host operand carries one region per workgroup element. When
// those regions provably hold the same bytes, the transfer moves more data
// than the workgroup can distinguish and can be narrowed. This file collects
// the rewrites that detect such cases; see the pass description in Passes.td.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMSCATTEROPTIMIZATIONSPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;

namespace {

/// Materialize a value of shape `shape` whose every element is `uniform`, of
/// the same flavour (tensor or memref) as `like`, right before `before`.
Value materializeUniformTile(RewriterBase &rewriter, Operation *before,
                             ShapedType like, ArrayRef<int64_t> shape,
                             TypedAttr uniform) {
  Location loc = before->getLoc();
  // A splat DenseElementsAttr stores one element whatever the shape, so this
  // stays cheap even before we know how the backend materializes it.
  auto contents = DenseElementsAttr::get(
      RankedTensorType::get(shape, uniform.getType()), uniform);

  if (isa<RankedTensorType>(like))
    return arith::ConstantOp::create(rewriter, loc, contents);

  // Bufferized: the tile has to be a constant global, which is what
  // bufferization would have produced for the constant above anyway.
  auto tileTy = MemRefType::get(shape, uniform.getType());
  auto module = before->getParentOfType<ModuleOp>();
  memref::GlobalOp global;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    global = memref::GlobalOp::create(
        rewriter, loc, "__cnm_scatter_tile",
        /*sym_visibility=*/rewriter.getStringAttr("private"), tileTy, contents,
        /*constant=*/true, /*alignment=*/IntegerAttr{});
    SymbolTable(module).insert(global);
  }
  return memref::GetGlobalOp::create(rewriter, loc, tileTy,
                                     global.getSymNameAttr());
}

/// Scattering a value whose elements are all the same constant sends every
/// workgroup element the same bytes, whatever the scatter map says. Replace
/// the host operand by a single buffer-shaped tile and drop the map.
struct BroadcastUniformScatter : OpRewritePattern<cnm::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cnm::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    ShapedType inputTy = op.getInput().getType();
    ArrayRef<int64_t> tileShape = op.getBuffer().getType().getShape();

    // Already one tile, so there is nothing to shrink -- and rewriting it
    // again would not terminate.
    if (inputTy.getShape() == tileShape)
      return failure();

    std::optional<TypedAttr> uniform = getUniformValue(op.getInput());
    if (!uniform || uniform->getType() != inputTy.getElementType())
      return failure();

    Value tile =
        materializeUniformTile(rewriter, op, inputTy, tileShape, *uniform);
    // Every leaf now takes the whole of `tile` as one block, so the map has
    // nothing left to name: no buffer dimension is retained, and the host
    // dimensions they cover are all of them.
    AffineMap broadcast =
        AffineMap::get(op.getBuffer().getType().getWorkgroupShape().size(), 0,
                       {}, getContext());
    rewriter.modifyOpInPlace(op, [&] {
      op.getInputMutable().assign(tile);
      op.setScatterMap(broadcast);
    });
    return success();
  }
};

/// The launch that consumes `buffer`, when replacing a scatter into it by an
/// initialization inside that launch's body is sound. Null otherwise.
///
/// Three things have to hold. The buffer must reach exactly one launch, or
/// there is no single body to initialize it in. That launch must be in the
/// same block as the scatter, so that it runs exactly as often -- a scatter
/// hoisted out of a loop around the launch seeds an accumulator once, and
/// re-seeding it on every launch would be a different program. And nothing
/// may read the buffer between the two, since after the rewrite it holds
/// nothing until the body fills it.
cnm::LaunchOp getSoleLaunchConsumer(cnm::ScatterOp scatter) {
  Value buffer = scatter.getBuffer();
  Block *block = scatter->getBlock();

  cnm::LaunchOp launch;
  for (Operation *user : buffer.getUsers()) {
    auto candidate = dyn_cast<cnm::LaunchOp>(user);
    if (!candidate)
      continue;
    if (launch)
      return {}; // more than one launch: no single body to fill in
    launch = candidate;
  }
  if (!launch || launch->getBlock() != block ||
      !scatter->isBeforeInBlock(launch))
    return {};

  for (Operation *user : buffer.getUsers())
    if (user != scatter && user != launch &&
        (user->getBlock() != block || user->isBeforeInBlock(launch)))
      return {};
  return launch;
}

/// Which of `launch`'s block arguments carries `buffer`.
std::optional<unsigned> getParamIndex(cnm::LaunchOp launch, Value buffer) {
  for (auto [index, param] : llvm::enumerate(launch.getParams()))
    if (param == buffer)
      return index;
  return std::nullopt;
}

/// Scattering a uniform value hands every leaf a constant it could just as
/// well write itself. Doing so is strictly better than transferring it: the
/// leaves write in parallel, no host bandwidth is spent, and -- once
/// `--upmem-tile-mram-buffers` has folded the fill into the staging buffer --
/// the leaves also stop loading a buffer they are about to overwrite.
///
/// The fill is emitted on the launch parameter, which still lives in the far
/// memory level, so this is correct on its own: a backend that does not fold
/// it pays an on-device round trip instead of a host transfer, which is the
/// cheaper of the two either way.
struct InitUniformBufferOnDevice : OpRewritePattern<cnm::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cnm::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<TypedAttr> uniform = getUniformValue(op.getInput());
    if (!uniform ||
        uniform->getType() != op.getBuffer().getType().getElementType())
      return failure();

    cnm::LaunchOp launch = getSoleLaunchConsumer(op);
    if (!launch)
      return failure();
    std::optional<unsigned> index = getParamIndex(launch, op.getBuffer());
    if (!index)
      return failure();

    // The body is IsolatedFromAbove, so the scalar has to be materialized
    // inside it rather than reused from around the scatter.
    Block &body = launch.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&body);
    Value scalar = arith::ConstantOp::create(rewriter, op.getLoc(), *uniform);
    linalg::FillOp::create(rewriter, op.getLoc(), ValueRange{scalar},
                           ValueRange{body.getArgument(*index)});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

struct CnmScatterOptimizationsPass
    : public cnm::impl::CnmScatterOptimizationsPassBase<
          CnmScatterOptimizationsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InitUniformBufferOnDevice, BroadcastUniformScatter>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

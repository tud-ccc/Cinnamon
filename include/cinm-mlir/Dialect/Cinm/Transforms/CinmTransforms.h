
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <mlir/Dialect/Transform/Interfaces/TransformInterfaces.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::cinm {

cinm::ComputeBlockOp isolateComputeBlock(cinm::ComputeOp, RewriterBase &);
cinm::ComputeOp deisolateComputeBlock(cinm::ComputeBlockOp, RewriterBase &);

void unwrapComputeBlockOp(cinm::ComputeBlockOp, RewriterBase &rewriter);

void unwrapComputeBlockOp(cinm::ComputeOp, RewriterBase &rewriter);
cinm::ComputeOp wrapOperationInCompute(Operation *op, RewriterBase &rewriter);

/// Whether `op` is a compute op (isolated or not) that only the host may run:
/// no accelerator, and a `cinm.available_platforms` of host platforms only,
/// which is how --cinm-complete-compute-graph marks the blocks it creates.
bool isHostComputeOp(Operation *op);

/// For a pass that takes compute ops in the isolated form only
/// (cinm.compute_block): an error on every cinm.compute under `root`, naming
/// `pass`, and failure if there is one. A pass that walks compute blocks would
/// otherwise skip such an op without a word.
LogicalResult requireIsolatedComputeOps(Operation *root, StringRef pass);

} // namespace mlir::cinm

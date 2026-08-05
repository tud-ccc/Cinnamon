
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <mlir/Dialect/Transform/Interfaces/TransformInterfaces.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::cinm {

cinm::ComputeBlockOp isolateComputeBlock(cinm::ComputeOp, RewriterBase &);
cinm::ComputeOp deisolateComputeBlock(cinm::ComputeBlockOp, RewriterBase &);

void unwrapComputeBlockOp(cinm::ComputeBlockOp, RewriterBase &rewriter);

void unwrapComputeBlockOp(cinm::ComputeOp, RewriterBase &rewriter);
cinm::ComputeOp wrapOperationInCompute(Operation *op, RewriterBase &rewriter);

} // namespace mlir::cinm

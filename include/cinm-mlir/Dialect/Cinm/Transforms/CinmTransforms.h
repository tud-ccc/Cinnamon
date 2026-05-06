
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <mlir/Dialect/Transform/Interfaces/TransformInterfaces.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::cinm {

cinm::ComputeOp isolateComputeBlock(cinm::FlexComputeOp, RewriterBase &);
cinm::FlexComputeOp deisolateComputeBlock(cinm::ComputeOp, RewriterBase &);

void unwrapComputeOp(cinm::ComputeOp, RewriterBase &rewriter);

void unwrapComputeOp(cinm::FlexComputeOp, RewriterBase &rewriter);

} // namespace mlir::cinm
//===- CinmOffloadModel.cpp - Reading an op, and what the host costs -----===//
//
// The device-independent half of an offload decision: what an op or a block
// computes and moves (measureOffloadFootprint), and what the host would take
// for it (hostRooflineSeconds). A backend supplies the other half.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmOffloadModel.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::cinm;

namespace {

/// Bytes a value occupies: a static shape's elements, or a scalar's own width
/// (a row maximum or sum handed to an elementwise body travels as one
/// number). Nothing when the shape is dynamic or the type has no width.
std::optional<double> bytesOf(Value v) {
  Type type = v.getType();
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped) {
    if (type.isIndex())
      return 8.0;
    if (type.isIntOrFloat())
      return static_cast<double>(
          llvm::divideCeil(type.getIntOrFloatBitWidth(), 8));
    return std::nullopt;
  }
  if (!shaped.hasStaticShape())
    return std::nullopt;
  Type elem = shaped.getElementType();
  if (!elem.isIntOrFloat())
    return std::nullopt;
  return static_cast<double>(shaped.getNumElements()) *
         llvm::divideCeil(elem.getIntOrFloatBitWidth(), 8);
}

/// Arithmetic ops in one iteration of a linalg body. A contraction's body is
/// a multiply and an add, an elementwise body whatever it spells out; casts
/// and yields are not arithmetic and are not counted.
double arithOpsPerIteration(linalg::LinalgOp op) {
  double n = 0.0;
  op.getBlock()->walk([&](Operation *inner) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
            arith::TruncFOp, arith::SIToFPOp, arith::FPToSIOp,
            arith::IndexCastOp, arith::BitcastOp, linalg::YieldOp>(inner))
      return;
    if (isa<arith::ArithDialect, math::MathDialect>(inner->getDialect()))
      n += 1.0;
  });
  return std::max(1.0, n);
}

/// The element type the multiplies happen in, and the one the accumulation
/// happens in. For a linalg contraction the inputs carry the first and the
/// init the second; they differ exactly when the op is mixed precision.
std::pair<Type, Type> arithTypes(linalg::LinalgOp op) {
  Type mulTy, accTy;
  if (op.getNumDpsInputs() > 0)
    if (auto shaped = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType()))
      mulTy = shaped.getElementType();
  if (op.getNumDpsInits() > 0)
    if (auto shaped = dyn_cast<ShapedType>(op.getDpsInits()[0].getType()))
      accTy = shaped.getElementType();
  if (!mulTy)
    mulTy = accTy;
  if (!accTy)
    accTy = mulTy;
  return {mulTy, accTy};
}

/// Fill in work and traffic for a linalg op. Fails when a shape or a loop
/// bound is dynamic, which the caller reports as "unknown" rather than as a
/// rejection.
LogicalResult measureLinalg(linalg::LinalgOp op, OffloadFootprint &f) {
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  double iterations = 1.0;
  for (int64_t r : ranges) {
    if (ShapedType::isDynamic(r))
      return failure();
    iterations *= static_cast<double>(r);
  }
  f.work = iterations * arithOpsPerIteration(op);

  for (OpOperand &operand : op->getOpOperands()) {
    std::optional<double> bytes = bytesOf(operand.get());
    if (!bytes)
      return failure();
    // An init operand is not traffic the device pays: a contraction's
    // accumulator is produced and consumed on the device, and its value
    // comes back as the result, counted below.
    if (op.isDpsInit(&operand))
      continue;
    if (cinm::isStaticValue(operand.get()))
      f.staticBytes += *bytes;
    else
      f.dynamicBytes += *bytes;
  }
  // Results are gathered back on every invocation.
  for (Value result : op->getResults()) {
    std::optional<double> bytes = bytesOf(result);
    if (!bytes)
      return failure();
    f.dynamicBytes += *bytes;
    f.dynamicOutBytes += *bytes;
  }

  std::tie(f.mulType, f.accType) = arithTypes(op);
  return success(f.mulType && f.accType);
}

/// Same for the cinm gemm-like ops, which the prim flow still has in hand
/// when platforms are assigned (the whole-program flow has converted to
/// linalg by then). Work is 2*M*N*K, the operands are what they say.
LogicalResult measureGemmlike(cinm::GemmlikeOpInterface op,
                              OffloadFootprint &f) {
  auto lhs = dyn_cast<ShapedType>(op.getLhs().getType());
  auto rhs = dyn_cast<ShapedType>(op.getRhs().getType());
  if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape())
    return failure();

  // Reduction extent is the lhs's last dimension in every variant; the
  // parallel extents are everything else the two operands span.
  double reduction = static_cast<double>(lhs.getDimSize(lhs.getRank() - 1));
  double parallel = static_cast<double>(lhs.getNumElements()) / reduction;
  double rhsParallel = static_cast<double>(rhs.getNumElements()) / reduction;
  // gemv's rhs is the vector, so it contributes no extra parallel extent.
  f.work = 2.0 * parallel *
           (rhs.getRank() > lhs.getRank() - 1 ? rhsParallel : 1.0) * reduction;

  for (Value operand : {op.getLhs(), op.getRhs()}) {
    std::optional<double> bytes = bytesOf(operand);
    if (!bytes)
      return failure();
    if (cinm::isStaticValue(operand))
      f.staticBytes += *bytes;
    else
      f.dynamicBytes += *bytes;
  }
  if (Value result = op.getGemmResult()) {
    std::optional<double> bytes = bytesOf(result);
    if (!bytes)
      return failure();
    f.dynamicBytes += *bytes;
    f.dynamicOutBytes += *bytes;
  }

  f.mulType = lhs.getElementType();
  f.accType = op.getAccumulatorElementType();
  return success();
}

/// Same for the cinm elementwise and reduce ops: one pass over the operands,
/// one arithmetic op per element. Both are pure traffic with no reuse, so
/// the roofline will reject them unless an operand is resident -- which is
/// the right answer and the reason `va` and `red` belong on the host.
LogicalResult measureCinmPointwise(Operation *op, OffloadFootprint &f) {
  double elements = 0.0;
  for (Value operand : op->getOperands()) {
    std::optional<double> bytes = bytesOf(operand);
    if (!bytes)
      return failure();
    auto shaped = cast<ShapedType>(operand.getType());
    elements = std::max(elements, static_cast<double>(shaped.getNumElements()));
    if (cinm::isStaticValue(operand))
      f.staticBytes += *bytes;
    else
      f.dynamicBytes += *bytes;
    if (!f.mulType)
      f.mulType = shaped.getElementType();
  }
  for (Value result : op->getResults()) {
    // A reduce to a scalar has nothing to gather back worth counting, but
    // the shaped case does.
    if (std::optional<double> bytes = bytesOf(result)) {
      f.dynamicBytes += *bytes;
      f.dynamicOutBytes += *bytes;
    } else if (isa<ShapedType>(result.getType()))
      return failure();
  }
  if (!f.mulType)
    return failure();
  f.accType = f.mulType;
  f.work = elements;
  return success();
}

} // namespace

namespace mlir::cinm {

OffloadFootprint measureOffloadFootprint(Operation *op) {
  OffloadFootprint f;
  LogicalResult measured = failure();
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    measured = measureLinalg(linalgOp, f);
  else if (auto gemmlike = dyn_cast<cinm::GemmlikeOpInterface>(op))
    measured = measureGemmlike(gemmlike, f);
  else if (isa<cinm::ElementwiseOp, cinm::ReduceOp>(op))
    measured = measureCinmPointwise(op, f);
  f.known = succeeded(measured);
  return f;
}

OffloadFootprint measureOffloadFootprint(ComputeBlockOp block) {
  OffloadFootprint f;
  // Work first: an op the reader does not know inside an otherwise readable
  // block would be priced at zero arithmetic, which is exactly the error
  // that makes a block look cheap on both sides.
  bool sawWork = false;
  double heaviest = 0.0;
  WalkResult walk = block.getBody().walk([&](Operation *op) {
    if (op == block.getOperation() || isa<cinm::YieldOp>(op))
      return WalkResult::advance();
    OffloadFootprint inner = measureOffloadFootprint(op);
    if (!inner.known)
      return WalkResult::advance();
    f.work += inner.work;
    // The types of whichever op does the most arithmetic, not of whichever
    // comes first: a contraction is usually preceded by the fill of its
    // accumulator, which has no operands to read a multiply's type from and
    // would hand the whole block the accumulator's. On this device that is
    // the difference between a multiply and a call -- an i8 multiply is one
    // instruction, an i32 one is __mulsi3 at twelve.
    if (!sawWork || inner.work > heaviest) {
      heaviest = inner.work;
      f.mulType = inner.mulType;
      f.accType = inner.accType;
    }
    sawWork = true;
    return WalkResult::advance();
  });
  if (walk.wasInterrupted() || !sawWork)
    return f;

  // Traffic is the block's own boundary: what it captures crosses the wire
  // once per invocation, what it yields comes back, and what its ops hand
  // each other never leaves the device.
  for (Value operand : block.getOperands()) {
    std::optional<double> bytes = bytesOf(operand);
    if (!bytes)
      return f;
    if (cinm::isStaticValue(operand))
      f.staticBytes += *bytes;
    else
      f.dynamicBytes += *bytes;
  }
  for (Value result : block.getResults()) {
    std::optional<double> bytes = bytesOf(result);
    if (!bytes)
      return f;
    f.dynamicBytes += *bytes;
    f.dynamicOutBytes += *bytes;
  }
  f.known = true;
  return f;
}

double hostRooflineSeconds(const OffloadFootprint &f, const HostModel &host) {
  if (!f.known)
    return 0.0;
  return std::max(f.work / host.opsPerSecond,
                  (f.staticBytes + f.dynamicBytes) / host.dramBytesPerSecond);
}

} // namespace mlir::cinm

/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFloat.h"
#include <mlir/IR/BuiltinTypes.h>

#define DEBUG_TYPE "cinm-ops"

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmEnums.cpp.inc"

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"
      >();
}

namespace mlir {
namespace cinm {

::mlir::ParseResult MinOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  return MaxOp::parse(parser, result);
}

::mlir::ParseResult MaxOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType;
  if (parser.parseOperand(input) || parser.parseColonType(inputType))
    return failure();

  SmallVector<Value> typedInput;
  if (parser.resolveOperand(input, inputType, typedInput))
    return failure();

  result.addOperands(typedInput);

  if (auto shaped = dyn_cast<ShapedType>(inputType)) {
    result.addTypes({shaped.getElementType()});
    return success();
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected a tensor type as operand");
  }
}

void printMinMaxOp(Value v, ::mlir::OpAsmPrinter &p) {
  p.printOperand(v);
  p << ": ";
  p.printType(v.getType());
}
void MaxOp::print(::mlir::OpAsmPrinter &p) { printMinMaxOp(getOperand(), p); }

void MinOp::print(::mlir::OpAsmPrinter &p) { printMinMaxOp(getOperand(), p); }

Value GemmOp::tile(OpBuilder builder, ArrayRef<int64_t> tileSizes) {
	MLIRContext *ctx = getContext();
    const Value lhs = getOperand(0);
    const Value rhs = getOperand(1);
    const ArrayRef<int64_t> lhsShape = lhs.getType().dyn_cast<RankedTensorType>().getShape();
    const RankedTensorType resultType = getResult().getType().cast<RankedTensorType>();
    const ArrayRef<int64_t> resultShape = resultType.getShape();

	return builder.create<tensor::GenerateOp>(getLoc(), resultType, ValueRange{}, [&](OpBuilder &builder, Location loc, ValueRange indices) {
        const SmallVector<int64_t> lhsOffsets{ShapedType::kDynamic, 0};
        const SmallVector<int64_t> rhsOffsets{0, ShapedType::kDynamic};
        const SmallVector<int64_t> lhsSizes{1, lhsShape[1]};
        const SmallVector<int64_t> rhsSizes{lhsShape[1], 1};
        const SmallVector<int64_t> strides{1, 1};

        SmallVector<Value> lhsDynamicOffsets{indices[0]};
        RankedTensorType lhsResultType = tensor::ExtractSliceOp::inferResultType(lhs.getType().cast<RankedTensorType>(), lhsOffsets, lhsSizes, strides);
        Value lhsSlice = builder.create<tensor::ExtractSliceOp>(loc, lhsResultType, lhs, lhsDynamicOffsets, ValueRange{}, ValueRange{}, lhsOffsets, lhsSizes, strides);

        SmallVector<Value> rhsDynamicOffsets{indices[1]};
        Type rhsResultType = tensor::ExtractSliceOp::inferResultType(rhs.getType().cast<RankedTensorType>(), rhsOffsets, rhsSizes, strides);
        Value rhsSlice = builder.create<tensor::ExtractSliceOp>(loc, rhsResultType, rhs, rhsDynamicOffsets, ValueRange{}, ValueRange{}, rhsOffsets, rhsSizes, strides);
        rhsSlice = builder.create<linalg::TransposeOp>(loc, rhsSlice,
            builder.create<tensor::EmptyOp>(loc, lhsResultType.getShape(), lhsResultType.getElementType()),
            SmallVector<int64_t>{1, 0}
        ).getResult()[0];

        Value prod = builder.create<arith::MulIOp>(loc, lhsSlice, rhsSlice);

        int64_t clusterSize = 16;
        int64_t clusterCount = lhsShape[1] / clusterSize;

        RankedTensorType intermediateResultType = RankedTensorType::get(SmallVector<int64_t>{clusterCount}, resultType.getElementType());
        Type reduceResultType = RankedTensorType::get({}, resultType.getElementType());
        Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(resultType.getElementType()));
        Value init = builder.create<tensor::FromElementsOp>(loc, reduceResultType, ValueRange{zero});

        Value sum = builder.create<tensor::GenerateOp>(loc, intermediateResultType, ValueRange{}, [&](OpBuilder &builder, Location loc, ValueRange indices) {
            SmallVector<int64_t> offsets{0, ShapedType::kDynamic};
            SmallVector<int64_t> sizes{1, clusterSize};
            SmallVector<int64_t> strides{1, clusterSize};
            Type sliceType = tensor::ExtractSliceOp::inferResultType(prod.getType().cast<RankedTensorType>(), offsets, sizes, strides);
            Value slice = builder.create<tensor::ExtractSliceOp>(loc, sliceType, prod, indices, ValueRange{}, ValueRange{}, offsets, sizes, strides);

            linalg::ReduceOp reduce = builder.create<linalg::ReduceOp>(loc, ValueRange{slice}, ValueRange{init}, SmallVector<int64_t>{0, 1}, [&](OpBuilder &builder, Location loc, ValueRange args) {
                Value result = builder.create<arith::AddIOp>(loc, args[0], args[1]);
                builder.create<linalg::YieldOp>(loc, result);
            });

            builder.create<tensor::YieldOp>(loc,
                builder.create<tensor::ExtractOp>(loc, reduce.getResult(0), ValueRange{})
            );
        });

        linalg::ReduceOp reduce = builder.create<linalg::ReduceOp>(loc, ValueRange{sum}, ValueRange{init}, SmallVector<int64_t>{0}, [&](OpBuilder &builder, Location loc, ValueRange args) {
            Value result = builder.create<arith::AddIOp>(loc, args[0], args[1]);
            builder.create<linalg::YieldOp>(loc, result);
        });

        builder.create<tensor::YieldOp>(loc,
            builder.create<tensor::ExtractOp>(loc, reduce.getResult(0), ValueRange{})
        );
    });
}

::mlir::LogicalResult SimSearchOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getLeft().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}

::mlir::LogicalResult TopKOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}


// Copied from the TOSA codebase.
::mlir::LogicalResult TransposeOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  ShapeAdaptor permsShape(adaptor.getPerms().getType());

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank() || !permsShape.hasRank() ||
      permsShape.isDynamicDim(0)) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of the
  // input which is illegal.
  if (permsShape.getDimSize(0) != inputShape.getRank()) {
    return failure();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(permsShape.getDimSize(0), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Rank-0 means no permutations matter.
  if (inputShape.getRank() == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputShape.getRank(), inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
  // If the permuations are a constant we can directly determine the output
  // shape.
  DenseIntElementsAttr attr;
  if (matchPattern(adaptor.getPerms(), m_Constant(&attr)) &&
      attr.getType().getRank() == 1) {
    ShapeAdaptor permShape = attr;
    outputShape.reserve(inputShape.getRank());
    for (int i = 0, s = inputShape.getRank(); i < s; i++) {
      outputShape[i] = inputShape.getDimSize(permShape.getDimSize(i));
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}
} // namespace cinm
} // namespace mlir

// parsers/printers

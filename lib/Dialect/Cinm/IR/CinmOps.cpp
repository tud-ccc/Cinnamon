/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
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
#include <cstdint>
#include <llvm/ADT/APInt.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LogicalResult.h>

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
  const ArrayRef<int64_t> lhsShape =
      lhs.getType().dyn_cast<RankedTensorType>().getShape();
  const RankedTensorType resultType =
      getResult().getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> resultShape = resultType.getShape();

  return builder.create<tensor::GenerateOp>(
      getLoc(), resultType, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange indices) {
        const SmallVector<int64_t> lhsOffsets{ShapedType::kDynamic, 0};
        const SmallVector<int64_t> rhsOffsets{0, ShapedType::kDynamic};
        const SmallVector<int64_t> lhsSizes{1, lhsShape[1]};
        const SmallVector<int64_t> rhsSizes{lhsShape[1], 1};
        const SmallVector<int64_t> strides{1, 1};

        SmallVector<Value> lhsDynamicOffsets{indices[0]};
        RankedTensorType lhsResultType =
            tensor::ExtractSliceOp::inferResultType(
                lhs.getType().cast<RankedTensorType>(), lhsOffsets, lhsSizes,
                strides);
        Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, lhsResultType, lhs, lhsDynamicOffsets, ValueRange{},
            ValueRange{}, lhsOffsets, lhsSizes, strides);

        SmallVector<Value> rhsDynamicOffsets{indices[1]};
        Type rhsResultType = tensor::ExtractSliceOp::inferResultType(
            rhs.getType().cast<RankedTensorType>(), rhsOffsets, rhsSizes,
            strides);
        Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, rhsResultType, rhs, rhsDynamicOffsets, ValueRange{},
            ValueRange{}, rhsOffsets, rhsSizes, strides);
        rhsSlice = builder
                       .create<linalg::TransposeOp>(
                           loc, rhsSlice,
                           builder.create<tensor::EmptyOp>(
                               loc, lhsResultType.getShape(),
                               lhsResultType.getElementType()),
                           SmallVector<int64_t>{1, 0})
                       .getResult()[0];

        Value prod = builder.create<arith::MulIOp>(loc, lhsSlice, rhsSlice);

        int64_t clusterSize = 16;
        int64_t clusterCount = lhsShape[1] / clusterSize;

        RankedTensorType intermediateResultType = RankedTensorType::get(
            SmallVector<int64_t>{clusterCount}, resultType.getElementType());
        Type reduceResultType =
            RankedTensorType::get({}, resultType.getElementType());
        Value zero = builder.create<arith::ConstantOp>(
            loc, builder.getZeroAttr(resultType.getElementType()));
        Value init = builder.create<tensor::FromElementsOp>(
            loc, reduceResultType, ValueRange{zero});

        Value sum = builder.create<tensor::GenerateOp>(
            loc, intermediateResultType, ValueRange{},
            [&](OpBuilder &builder, Location loc, ValueRange indices) {
              SmallVector<int64_t> offsets{0, ShapedType::kDynamic};
              SmallVector<int64_t> sizes{1, clusterSize};
              SmallVector<int64_t> strides{1, clusterSize};
              Type sliceType = tensor::ExtractSliceOp::inferResultType(
                  prod.getType().cast<RankedTensorType>(), offsets, sizes,
                  strides);
              Value slice = builder.create<tensor::ExtractSliceOp>(
                  loc, sliceType, prod, indices, ValueRange{}, ValueRange{},
                  offsets, sizes, strides);

              linalg::ReduceOp reduce = builder.create<linalg::ReduceOp>(
                  loc, ValueRange{slice}, ValueRange{init},
                  SmallVector<int64_t>{0, 1},
                  [&](OpBuilder &builder, Location loc, ValueRange args) {
                    Value result =
                        builder.create<arith::AddIOp>(loc, args[0], args[1]);
                    builder.create<linalg::YieldOp>(loc, result);
                  });

              builder.create<tensor::YieldOp>(
                  loc, builder.create<tensor::ExtractOp>(
                           loc, reduce.getResult(0), ValueRange{}));
            });

        linalg::ReduceOp reduce = builder.create<linalg::ReduceOp>(
            loc, ValueRange{sum}, ValueRange{init}, SmallVector<int64_t>{0},
            [&](OpBuilder &builder, Location loc, ValueRange args) {
              Value result =
                  builder.create<arith::AddIOp>(loc, args[0], args[1]);
              builder.create<linalg::YieldOp>(loc, result);
            });

        builder.create<tensor::YieldOp>(
            loc, builder.create<tensor::ExtractOp>(loc, reduce.getResult(0),
                                                   ValueRange{}));
      });
}

::mlir::LogicalResult GemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLeft().getType());
  ShapeAdaptor rhsShape(adaptor.getRight().getType());

  SmallVector<int64_t, 2> outShape;
  outShape.push_back(lhsShape.getDimSize(0));
  outShape.push_back(rhsShape.getDimSize(1));

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success()
}

::mlir::LogicalResult GemvOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  auto result = ShapedTypeComponents(adaptor.getRight().geadaptor.getRight().getShape());
  inferredReturnShapes.emplace_back(std::move(result));
  return success()
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

ParseResult
parseCaptureArgs(OpAsmParser &parser,
                 SmallVectorImpl<OpAsmParser::Argument> &lhs,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &rhs,
                 SmallVectorImpl<Type> &types) {
  auto parseElt = [&]() -> ParseResult {
    if (parser.parseArgument(lhs.emplace_back()) || parser.parseEqual() ||
        parser.parseOperand(rhs.emplace_back()) ||
        parser.parseColonType<Type>(types.emplace_back()))
      return failure();
    return success();
  };
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt);
}

/*

    %rbuf = cinm.compute(%arg0 = %inpt : tensor<...>) -> tensor<...> {
      %flt = arith.constant <"...">: tensor<...>
      %conv = cinm.op.gemm %arg0, %flt: tensor<...>, tensor<...>
      cinm.yield %conv : tensor<...>
    }
*/
ParseResult parseComputeOp(OpAsmParser &parser, OperationState &result) {

  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;

  SmallVector<OpAsmParser::Argument, 4> capturedParams;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> capturedArgs;
  SmallVector<Type, 4> capturedArgsTypes;

  if (parseCaptureArgs(parser, capturedParams, capturedArgs, capturedArgsTypes))
    return failure();

  SmallVector<Value, 4> resolvedCaptArgs;
  if (parser.resolveOperands(std::move(capturedArgs), capturedArgsTypes,
                             parser.getCurrentLocation(), resolvedCaptArgs))
    return failure();

  result.addOperands(std::move(resolvedCaptArgs));

  SmallVector<Type, 4> resultTys;
  if (parser.parseArrowTypeList(resultTys))
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected -> return type");
  result.addTypes(resultTys);

  std::string errorMessage;
  SmallVector<OpAsmParser::Argument> regionParams;
  regionParams.reserve(capturedParams.size());
  for (auto [arg, argT] : llvm::zip(capturedParams, capturedArgsTypes)) {
    arg.type = argT;
    regionParams.push_back(arg);
  }

  // Parse the body.
  auto *body = result.addRegion();
  SMLoc loc = parser.getCurrentLocation();
  OptionalParseResult parseResult =
      parser.parseRegion(*body, regionParams,
                         /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty cinm.compute body");
  }
  return success();
}

ParseResult ComputeOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseComputeOp(parser, result);
}

void ComputeOp::print(OpAsmPrinter &p) {
  // Print the operation and the function name.
  p << ' ';

  Region &body = getBody();
  ValueRange captured = getOperands();

  unsigned numCapt = captured.size();
  p << '(';
  for (unsigned i = 0; i < numCapt; ++i) {
    if (i > 0)
      p << ", ";

    p.printRegionArgument(body.getArgument(i), {}, /*omitType=*/true);
    p << " = ";
    p.printOperand(captured[i]);
    p << " : ";
    p.printType(captured[i].getType());
  }
  p << ')';

  auto resultTy = getResult().getType();
  p.getStream() << " -> ";
  p.printType(resultTy);

  // Print the body if this is not an external function.
  p << ' ';
  p.printRegion(body,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

ParseResult SimSearchOp::parse(::mlir::OpAsmParser &parser,
                               ::mlir::OperationState &result) {

  //  let assemblyFormat= "$metric `,` $k `(` $left `,` $right `)` attr-dict `:`
  //  type($left)";
  std::string opname;
  if (parser.parseKeywordOrString(&opname))
    return failure();

  SimilarityMetric metric;
  if (opname == "cos") {
    metric = SimilarityMetric::COS;
  } else if (opname == "dot") {
    metric = SimilarityMetric::DOT;
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected a string \"cos\" or \"dot\"");
  }

  result.addAttribute(getMetricAttrName(result.name),
                      SimilarityMetricAttr::get(result.getContext(), metric));

  int64_t numK;
  if (parser.parseInteger(numK))
    return failure();

  auto i64Ty = IntegerType::get(result.getContext(), 64);
  result.addAttribute(getKAttrName(result.name), IntegerAttr::get(i64Ty, numK));

  OpAsmParser::UnresolvedOperand left, right;
  NamedAttrList attrDict;
  Type opTy;

  if (parser.parseLParen() || parser.parseOperand(left, false) ||
      parser.parseComma() || parser.parseOperand(right, false) ||
      parser.parseRParen() || parser.parseColonType(opTy))
    return failure();

  result.addAttributes(attrDict);
  if (parser.resolveOperand(left, opTy, result.operands) ||
      parser.resolveOperand(right, opTy, result.operands))
    return failure();

  // finally add result types

  auto eltTy = opTy.cast<RankedTensorType>().getElementType();
  auto resultValuesTy = RankedTensorType::get({ShapedType::kDynamic}, eltTy);
  auto resultIndicesTy = RankedTensorType::get(
      {ShapedType::kDynamic}, IndexType::get(result.getContext()));
  result.addTypes({resultValuesTy, resultIndicesTy});

  return success();
}

void SimSearchOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printKeywordOrString(stringifySimilarityMetric(getMetric()));
  p << ' ' << getK();
  p << " (";
  p.printOperand(getLeft());
  p << ", ";
  p.printOperand(getRight());
  p << ") : ";
  p.printType(getLeft().getType());
}

} // namespace cinm
} // namespace mlir

// parsers/printers

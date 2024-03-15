/// Implements the Cnm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/APFloat.h"

#define DEBUG_TYPE "cnm-ops"

using namespace mlir;
using namespace mlir::cnm;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CnmDialect
//===----------------------------------------------------------------------===//

void CnmDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.cpp.inc"
        >();
}

// parsers/printers

ParseResult mlir::cnm::WorkgroupOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    if (parser.parseLSquare().failed()) {
        return ParseResult::failure();
    }

    int64_t current = 0;
    llvm::SmallVector<int64_t, 2> shape;

    OptionalParseResult dimension = parser.parseOptionalInteger(current);
    while (dimension.has_value()) {
        if (dimension.value().failed()) {
            return ParseResult::failure();
        }

        shape.push_back(current);
        dimension = parser.parseOptionalInteger(current);
    }

    if (parser.parseRSquare().failed()) {
        return ParseResult::failure();
    }

    result.addTypes(WorkgroupType::get(result.getContext(), shape));

    NamedAttrList attributes;
    if (parser.parseOptionalAttrDict(attributes).failed()) {
        return ParseResult::failure();
    }
    result.addAttributes(attributes);

    return ParseResult::success();
}

void mlir::cnm::WorkgroupOp::print(mlir::OpAsmPrinter &printer) {
    printer<<" [";
    const auto shape = getType().getShape();
    for (uint64_t i = 0; i < shape.size(); i++) {
        if (i > 0) {
            printer<<" ";
        }
        printer<<shape[i];
    }
    printer<<"]";
    printer.printOptionalAttrDict(this->getOperation()->getAttrs());
}

ParseResult mlir::cnm::AllocOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    if (parser.parseLParen().failed()) {
        return ParseResult::failure();
    }

    // TODO: maybe parameters?, not covered by the example cnm file

    if (parser.parseRParen().failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("for").failed()) {
        return ParseResult::failure();
    }

    OpAsmParser::UnresolvedOperand wg;
    if (parser.parseOperand(wg).failed()) {
        return ParseResult::failure();
    }

    NamedAttrList attributes;
    if (parser.parseOptionalAttrDict(attributes).failed()) {
        return ParseResult::failure();
    }
    result.addAttributes(attributes);

    Type bufferType;
    if (parser.parseColonType(bufferType).failed()) {
        return ParseResult::failure();
    }
    result.addTypes(bufferType);

    if (parser.parseKeyword("for").failed()) {
        return ParseResult::failure();
    }

    Type workgroupType;
    if (parser.parseType(workgroupType).failed()) {
        return ParseResult::failure();
    }

    llvm::SmallVector<Value, 1> operands;
    if (parser.resolveOperand(wg, workgroupType, operands).failed()) {
        return ParseResult::failure();
    }
    result.addOperands(operands);

    return ParseResult::success();
}

void mlir::cnm::AllocOp::print(mlir::OpAsmPrinter &printer) {
    printer<<" () for "<<getOperand();
    printer.printOptionalAttrDict(this->getOperation()->getAttrs());
    printer<<" : "<<getType()<<" for "<<getOperand().getType();
}

ParseResult mlir::cnm::SetZeroOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    OpAsmParser::UnresolvedOperand buffer;
    if (parser.parseOperand(buffer).failed()) {
        return ParseResult::failure();
    }

    Type bufferType;
    if (parser.parseColonType(bufferType).failed()) {
        return ParseResult::failure();
    }
    result.addTypes(bufferType);

    llvm::SmallVector<Value, 1> operands;
    if (parser.resolveOperand(buffer, bufferType, operands).failed()) {
        return ParseResult::failure();
    }
    result.addOperands(operands);

    return ParseResult::success();
}

void mlir::cnm::SetZeroOp::print(mlir::OpAsmPrinter &printer) {
    printer<<" "<<getOperand()<<" : "<<getOperand().getType();
}

ParseResult mlir::cnm::ScatterOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    SmallVector<OpAsmParser::UnresolvedOperand, 3> unresolvedOperands;
    if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("into").failed()) {
        return ParseResult::failure();
    }

    if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseLSquare().failed()) {
        return ParseResult::failure();
    }

    AffineMapAttr mapAttr;
    if (parser.parseAttribute(mapAttr, "map", result.attributes)) {
        return ParseResult::failure();
    }

    if (parser.parseRSquare().failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("of").failed()) {
        return ParseResult::failure();
    }

    if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    SmallVector<Type, 3> operandTypes;
    if (parser.parseColonType(operandTypes.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("into").failed()) {
        return ParseResult::failure();
    }

    if (parser.parseType(operandTypes.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("of").failed()) {
        return ParseResult::failure();
    }

    if (parser.parseType(operandTypes.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    llvm::SmallVector<Value, 3> operands;
    for (size_t i = 0; i < unresolvedOperands.size(); i++) {
        if (parser.resolveOperand(unresolvedOperands[i], operandTypes[i], operands).failed()) {
            return ParseResult::failure();
        }
    }
    result.addOperands(operands);

    result.addTypes(ScatterTokenType::get(parser.getContext()));

    return ParseResult::success();
}

void mlir::cnm::ScatterOp::print(mlir::OpAsmPrinter &printer) {
    printer<<" "<<getOperand(0)<<" into "<<getOperand(1)<<"["<<getOperation()->getAttr("map")<<"]"<<" of "<<getOperand(2);
    printer<<" : "<<getOperand(0).getType()<<" into "<<getOperand(1).getType()<<" of "<<getOperand(2).getType();
}

ParseResult mlir::cnm::GatherOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    SmallVector<OpAsmParser::UnresolvedOperand, 2> unresolvedOperands;
    if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseLSquare().failed()) {
        return ParseResult::failure();
    }

    AffineMapAttr mapAttr;
    if (parser.parseAttribute(mapAttr, "map", result.attributes)) {
        return failure();
    }

    if (parser.parseRSquare().failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("of").failed()) {
        return ParseResult::failure();
    }

    if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    SmallVector<Type, 2> operandTypes;
    if (parser.parseColonType(operandTypes.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("of").failed()) {
        return ParseResult::failure();
    }

    if (parser.parseType(operandTypes.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseKeyword("into").failed()) {
        return ParseResult::failure();
    }

    Type result_type;
    if (parser.parseType(result_type).failed()) {
        return ParseResult::failure();
    }

    llvm::SmallVector<Value, 3> operands;
    for (size_t i = 0; i < unresolvedOperands.size(); i++) {
        if (parser.resolveOperand(unresolvedOperands[i], operandTypes[i], operands).failed()) {
            return ParseResult::failure();
        }
    }
    result.addOperands(operands);

    result.addTypes(result_type);
    result.addTypes(GatherTokenType::get(parser.getContext()));

    return ParseResult::success();
}

void mlir::cnm::GatherOp::print(mlir::OpAsmPrinter &printer) {
    printer<<" "<<getOperand(0)<<"["<<getOperation()->getAttr("map")<<"]"<<" of "<<getOperand(1);
    printer<<" : "<<getOperand(0).getType()<<" of "<<getOperand(1).getType()<<" into "<<getResultTypes()[0];
}

ParseResult mlir::cnm::LaunchOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    result.addTypes(LaunchTokenType::get(parser.getContext()));

    SmallVector<OpAsmParser::UnresolvedOperand> unresolvedOperands;
    SmallVector<Type> operandTypes;
    operandTypes.push_back(Type()); // workgroup type

    if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
        return ParseResult::failure();
    }

    if (parser.parseLParen().failed()) {
        return ParseResult::failure();
    }

    if (parser.parseOptionalRParen().failed()) { // 1 or more parameters
        if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
            return ParseResult::failure();
        }

        while (parser.parseOptionalColon().failed()) {
            if (parser.parseComma().failed()
            || parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
                return ParseResult::failure();
            }
        }

        if (parser.parseType(operandTypes.emplace_back()).failed()) {
            return ParseResult::failure();
        }

        while (parser.parseOptionalRParen().failed()) {
            if (parser.parseComma().failed()
            || parser.parseType(operandTypes.emplace_back()).failed()) {
                return ParseResult::failure();
            }
        }
    }

    if (parser.parseRegion(*result.addRegion(), {}, false).failed()) {
        return ParseResult::failure();
    }

    OpBuilder builder(result.getContext());
    builder.setInsertionPointToEnd(&result.regions.back()->back());
    builder.create<cnm::TerminatorOp>(result.location);

    if (parser.parseColonType(operandTypes[0]).failed()) {
        return ParseResult::failure();
    }

    llvm::SmallVector<Value> operands;
    for (size_t i = 0; i < unresolvedOperands.size(); i++) {
        if (parser.resolveOperand(unresolvedOperands[i], operandTypes[i], operands).failed()) {
            return ParseResult::failure();
        }
    }
    result.addOperands(operands);

    return ParseResult::success();
}

void mlir::cnm::LaunchOp::print(mlir::OpAsmPrinter &printer) {
    printer<<" "<<getOperand(0);
    printer<<" (";
    for (uint64_t i = 1; i < getNumOperands(); i++) {
        if (i > 1) {
            printer<<", ";
        }
        printer<<getOperand(i);
    }
    printer<<": ";
    for (uint64_t i = 1; i < getNumOperands(); i++) {
        if (i > 1) {
            printer<<", ";
        }
        printer<<getOperand(i).getType();
    }
    printer<<") ";
    printer.printRegion(getRegion());
    printer<<" : "<<getOperand(0).getType();
}

ParseResult mlir::cnm::TerminatorOp::parse(mlir::OpAsmParser&, mlir::OperationState&) {
    return ParseResult::success();
}

void mlir::cnm::TerminatorOp::print(mlir::OpAsmPrinter&) {}

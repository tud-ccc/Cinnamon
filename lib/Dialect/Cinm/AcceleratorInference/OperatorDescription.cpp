//===- OperatorDescription.cpp - A compute block as data ------------------===//
//
// describeComputeBlock: the computation of a compute block, read off its
// linalg form, in a shape a tool outside the compiler can rebuild it from.
// See OperatorDescription.h for the format.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/OperatorDescription.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>

namespace mlir::cinm {

namespace {

namespace json = llvm::json;

template <class T> std::string printed(const T &thing) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << thing;
  return s;
}

/// The element type in the spelling TVM (and numpy) use.
std::string dtypeOf(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
    return (integer.isUnsigned() ? "uint" : "int") +
           std::to_string(integer.getWidth());
  }
  if (isa<IndexType>(type))
    return "int64";
  if (type.isBF16())
    return "bfloat16";
  if (auto real = dyn_cast<FloatType>(type))
    return "float" + std::to_string(real.getWidth());
  return printed(type);
}

json::Value shapeOf(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank())
    return nullptr;
  json::Array dims;
  for (int64_t dim : shaped.getShape())
    dims.push_back(ShapedType::isDynamic(dim) ? json::Value(nullptr)
                                              : json::Value(dim));
  return dims;
}

json::Object typeOf(Type type) {
  return json::Object{{"shape", shapeOf(type)}, {"dtype", dtypeOf(type)}};
}

/// Which iteration dimension indexes each operand dimension, or the map
/// itself when that is not a well-defined question.
json::Value mapOf(AffineMap map) {
  if (!map.isProjectedPermutation())
    return json::Object{{"affine_map", printed(map)}};
  json::Array dims;
  for (AffineExpr result : map.getResults())
    dims.push_back(
        static_cast<int64_t>(cast<AffineDimExpr>(result).getPosition()));
  return dims;
}

/// A scalar attribute as a number, if it is one.
std::optional<json::Value> numberOf(Attribute attr) {
  if (auto splat = dyn_cast<SplatElementsAttr>(attr))
    attr = splat.getSplatValue<Attribute>();
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    if (integer.getType().isInteger(1))
      return json::Value(integer.getValue().getBoolValue());
    return json::Value(integer.getValue().getSExtValue());
  }
  if (auto real = dyn_cast<FloatAttr>(attr))
    return json::Value(real.getValueAsDouble());
  return std::nullopt;
}

/// The value a constant fill writes, when `op` is one: a `linalg.fill`, or
/// the generic it generalizes to -- one input, broadcast into the init with
/// every iterator parallel and nothing but the input yielded -- where the
/// input is a constant splat.
std::optional<json::Value> fillValueOf(Operation *op) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInputs() != 1 ||
      linalgOp.getNumDpsInits() != 1)
    return std::nullopt;
  if (!isa<linalg::FillOp>(op)) {
    if (linalgOp.getNumReductionLoops() != 0)
      return std::nullopt;
    Block *body = linalgOp.getBlock();
    auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != body->getArgument(0) ||
        &body->front() != yield.getOperation())
      return std::nullopt;
  }
  Value input = linalgOp.getDpsInputOperand(0)->get();
  Attribute constant;
  if (!matchPattern(input, m_Constant(&constant)))
    return std::nullopt;
  return numberOf(constant);
}

/// Whether `attr` is an arith flag set with no flag in it -- the default,
/// which says nothing.
bool isDefaultFlags(Attribute attr) {
  if (auto overflow = dyn_cast<arith::IntegerOverflowFlagsAttr>(attr))
    return overflow.getValue() == arith::IntegerOverflowFlags::none;
  if (auto fastmath = dyn_cast<arith::FastMathFlagsAttr>(attr))
    return fastmath.getValue() == arith::FastMathFlags::none;
  return false;
}

/// Short spelling of a body op for the readable expression: `arith.muli`
/// becomes `muli`, and a comparison carries its predicate.
std::string shortName(Operation *op) {
  StringRef name = op->getName().getStringRef();
  std::string out = name.substr(name.find('.') + 1).str();
  Attribute predicate = op->getAttr("predicate");
  if (auto i = dyn_cast_or_null<arith::CmpIPredicateAttr>(predicate))
    out += "_" + arith::stringifyCmpIPredicate(i.getValue()).str();
  else if (auto f = dyn_cast_or_null<arith::CmpFPredicateAttr>(predicate))
    out += "_" + arith::stringifyCmpFPredicate(f.getValue()).str();
  return out;
}

class Describer {
public:
  explicit Describer(ComputeBlockOp block) : block(block) {}

  json::Value run() {
    json::Array args;
    for (auto [i, arg] : llvm::enumerate(block.getBodyArguments())) {
      std::string name = "arg" + std::to_string(i);
      names[arg] = name;
      json::Object entry = typeOf(arg.getType());
      entry["name"] = name;
      entry["static"] = isStaticValue(block->getOperand(i));
      args.push_back(std::move(entry));
    }

    collectFolded();

    json::Array ops;
    for (Operation &op : block.getBody().front()) {
      if (isa<cinm::YieldOp>(op) || folded.contains(&op))
        continue;
      nameResults(&op);
      if (auto generic = dyn_cast<linalg::GenericOp>(op))
        ops.push_back(describeGeneric(generic));
      else
        ops.push_back(describeOther(&op));
    }

    json::Array results;
    for (Value value : block.getBody().front().getTerminator()->getOperands())
      results.push_back(nameOf(value));

    OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(16);
    std::string mlir;
    llvm::raw_string_ostream os(mlir);
    block->print(os, flags);

    return json::Object{{"args", std::move(args)},
                        {"ops", std::move(ops)},
                        {"results", std::move(results)},
                        {"mlir", std::move(mlir)}};
  }

private:
  ComputeBlockOp block;
  DenseMap<Value, std::string> names;
  /// Ops that are not listed because every use of them is folded into the
  /// description of a generic: fills and empties feeding inits, and the
  /// constants those fills broadcast.
  DenseSet<Operation *> folded;
  unsigned nextResult = 0, nextConstant = 0;

  std::string nameOf(Value value) const {
    auto it = names.find(value);
    return it == names.end() ? "?" : it->second;
  }

  void nameResults(Operation *op) {
    if (op->getNumResults() == 0)
      return;
    const bool constant = op->hasTrait<OpTrait::ConstantLike>();
    std::string base = constant ? "c" + std::to_string(nextConstant++)
                                : "t" + std::to_string(nextResult++);
    for (auto [k, result] : llvm::enumerate(op->getResults()))
      names[result] = k == 0 ? base : base + "." + std::to_string(k);
  }

  /// Whether every use of `op`'s results is as the init of a generic.
  static bool onlyFeedsInits(Operation *op) {
    for (OpOperand &use : op->getUses()) {
      auto consumer = dyn_cast<linalg::GenericOp>(use.getOwner());
      if (!consumer || !consumer.isDpsInit(&use))
        return false;
    }
    return !op->use_empty();
  }

  void collectFolded() {
    for (Operation &op : block.getBody().front()) {
      if (isa<tensor::EmptyOp>(op) && onlyFeedsInits(&op))
        folded.insert(&op);
      else if (fillValueOf(&op) && onlyFeedsInits(&op))
        folded.insert(&op);
    }
    // A constant goes with the fills it feeds, and with the empties those
    // fills write into.
    for (Operation &op : block.getBody().front()) {
      if (!op.hasTrait<OpTrait::ConstantLike>() || op.use_empty())
        continue;
      if (llvm::all_of(op.getUsers(),
                       [&](Operation *user) { return folded.contains(user); }))
        folded.insert(&op);
    }
  }

  json::Object describeInit(OpOperand &init, AffineMap map) {
    json::Object entry{{"map", mapOf(map)}};
    Operation *producer = init.get().getDefiningOp();
    if (producer && folded.contains(producer)) {
      if (isa<tensor::EmptyOp>(producer))
        entry["empty"] = true;
      else
        entry["fill"] = *fillValueOf(producer);
      return entry;
    }
    entry["value"] = nameOf(init.get());
    return entry;
  }

  json::Object describeGeneric(linalg::GenericOp op) {
    json::Object entry;
    entry["name"] = nameOf(op->getResult(0));
    auto tag = op->getAttrOfType<StringAttr>(CinmDialect::DEBUG_TAG_NAME);
    entry["kind"] = tag ? tag.getValue().str() : "linalg.generic";

    json::Array domain;
    for (int64_t extent : op.getStaticLoopRanges())
      domain.push_back(ShapedType::isDynamic(extent) ? json::Value(nullptr)
                                                     : json::Value(extent));
    entry["domain"] = std::move(domain);
    json::Array iterators;
    for (utils::IteratorType it : op.getIteratorTypesArray())
      iterators.push_back(utils::stringifyIteratorType(it).str());
    entry["iterators"] = std::move(iterators);

    // Body arguments are the inputs then the inits, in operand order.
    Block &body = op.getRegion().front();
    json::Array inputs, inits;
    for (OpOperand *input : op.getDpsInputOperands()) {
      inputs.push_back(
          json::Object{{"value", nameOf(input->get())},
                       {"map", mapOf(op.getMatchingIndexingMap(input))}});
      names[body.getArgument(input->getOperandNumber())] =
          "in" + std::to_string(input->getOperandNumber());
    }
    for (auto [k, init] : llvm::enumerate(op.getDpsInitsMutable())) {
      inits.push_back(describeInit(init, op.getMatchingIndexingMap(&init)));
      names[op.getMatchingBlockArgument(&init)] = "out" + std::to_string(k);
    }
    entry["inputs"] = std::move(inputs);
    entry["inits"] = std::move(inits);

    json::Array results;
    for (Value result : op->getResults())
      results.push_back(typeOf(result.getType()));
    entry["results"] = std::move(results);

    json::Array steps;
    unsigned nextValue = 0;
    for (Operation &inner : body.without_terminator()) {
      json::Object step;
      for (Value result : inner.getResults())
        names[result] = "v" + std::to_string(nextValue++);
      if (inner.getNumResults() == 1) {
        step["name"] = nameOf(inner.getResult(0));
        step["dtype"] = dtypeOf(inner.getResult(0).getType());
      }
      step["op"] = inner.getName().getStringRef().str();
      json::Array operands;
      for (Value operand : inner.getOperands())
        operands.push_back(nameOf(operand));
      step["args"] = std::move(operands);
      json::Object attrs;
      for (NamedAttribute attr : inner.getAttrs()) {
        if (isDefaultFlags(attr.getValue()))
          continue;
        if (std::optional<json::Value> number = numberOf(attr.getValue()))
          attrs[attr.getName().getValue()] = std::move(*number);
        else
          attrs[attr.getName().getValue()] = printed(attr.getValue());
      }
      if (!attrs.empty())
        step["attrs"] = std::move(attrs);
      steps.push_back(std::move(step));
    }
    entry["body"] = std::move(steps);

    json::Array yields, exprs;
    for (Value value : body.getTerminator()->getOperands()) {
      yields.push_back(nameOf(value));
      exprs.push_back(exprOf(value));
    }
    entry["yield"] = std::move(yields);
    entry["expr"] = std::move(exprs);
    return entry;
  }

  /// The readable form of a body value: its definition inlined all the way
  /// down to the body arguments. Repeats shared subexpressions; `body` is the
  /// exact form.
  std::string exprOf(Value value) const {
    Operation *def = value.getDefiningOp();
    if (!def)
      return nameOf(value);
    if (Attribute constant; matchPattern(value, m_Constant(&constant)))
      if (std::optional<json::Value> number = numberOf(constant))
        return printed(*number);
    std::string out = shortName(def) + "(";
    llvm::raw_string_ostream os(out);
    llvm::interleave(
        def->getOperands(), os, [&](Value operand) { os << exprOf(operand); },
        ", ");
    os << ")";
    return out;
  }

  json::Object describeOther(Operation *op) {
    json::Object entry;
    if (op->getNumResults() > 0)
      entry["name"] = nameOf(op->getResult(0));
    entry["kind"] = "other";
    entry["op"] = op->getName().getStringRef().str();
    json::Array operands;
    for (Value operand : op->getOperands())
      operands.push_back(nameOf(operand));
    entry["operands"] = std::move(operands);
    json::Array results;
    for (Value result : op->getResults())
      results.push_back(typeOf(result.getType()));
    entry["results"] = std::move(results);
    if (op->hasTrait<OpTrait::ConstantLike>() && op->getNumResults() == 1) {
      Attribute constant;
      if (matchPattern(op->getResult(0), m_Constant(&constant)))
        if (std::optional<json::Value> number = numberOf(constant))
          entry["splat"] = std::move(*number);
    }
    return entry;
  }
};

} // namespace

llvm::json::Value describeComputeBlock(ComputeBlockOp reference) {
  return Describer(reference).run();
}

std::string describeElementType(Type type) { return dtypeOf(type); }

} // namespace mlir::cinm

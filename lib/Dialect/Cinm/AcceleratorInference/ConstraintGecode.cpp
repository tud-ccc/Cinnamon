#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintGecode.h"

#include <algorithm>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <memory>

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

namespace mlir::cinm::constraints {
namespace {

using Kind = ConstraintNode::Kind;

// ===----------------------------------------------------------------------===//
// Domains
// ===----------------------------------------------------------------------===//

/// A parameter's domain as a Gecode value set.
///
/// A contiguous unit-step range keeps its interval representation, which is
/// what lets the bounds propagators do their work without materialising the
/// values; everything else is an explicit set, because a divisor list is not
/// an interval and pretending otherwise would offer values the parameter does
/// not have.
Gecode::IntSet domainOf(const SearchParam &param) {
  if (const auto *range = std::get_if<IntRange>(&param.domain)) {
    if (range->step == 1)
      return Gecode::IntSet(range->lo, range->hi);
  }
  std::vector<int> values;
  values.reserve(param.cardinality());
  for (size_t i = 0, e = param.cardinality(); i < e; ++i)
    values.push_back(static_cast<int>(param.valueAt(i)));
  // IntSet wants no particular order, but duplicates would silently make the
  // domain smaller than the parameter claims, so this is worth being sure of.
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return Gecode::IntSet(values.data(), static_cast<int>(values.size()));
}

/// Number of Configuration entries `params` describes, which is not
/// `params.size()`: see the note above ConfigSpace.
size_t numDimensions(llvm::ArrayRef<SearchParam> params) {
  size_t n = 0;
  for (const SearchParam &param : params)
    n += param.arity();
  return n;
}

// ===----------------------------------------------------------------------===//
// Translation
// ===----------------------------------------------------------------------===//

/// Translates the constraint IR into MiniModel expressions over a fixed
/// variable array. Stateless apart from the array; it exists as a struct only
/// so the two mutually recursive halves can share it.
struct Translator {
  /// Needed to reify a BoolAsInt: turning a truth value into a number means
  /// posting a variable that stands for it, which is a change to the model
  /// rather than a change to an expression.
  Gecode::Home home;
  const Gecode::IntVarArgs &vars;

  Gecode::LinIntExpr toInt(const ConstraintNode &node) const {
    switch (node.kind) {
    case Kind::Const:
      return Gecode::LinIntExpr(static_cast<int>(node.constValue()));
    case Kind::Var:
      return Gecode::LinIntExpr(vars[static_cast<int>(node.varIdx())]);
    case Kind::Add: {
      Gecode::LinIntExpr acc(0);
      for (const ConstraintNodePtr &op : node.operands())
        acc = acc + toInt(*op);
      return acc;
    }
    case Kind::Mul: {
      Gecode::LinIntExpr acc(1);
      for (const ConstraintNodePtr &op : node.operands())
        // LinIntExpr * LinIntExpr is the nonlinear multiplication propagator,
        // not a linear term; a product of two parameters is exactly what the
        // identity constraints are made of.
        acc = acc * toInt(*op);
      return acc;
    }
    case Kind::BoolAsInt:
      // A reified boolean: the variable is 1 exactly when the operand holds.
      // Its own divisions are discharged inside toBool, which is why
      // divisibilityOf stops here.
      return Gecode::expr(home, toBool(*node.operands()[0]));
    case Kind::Div:
      // Truncating quotient. What makes it the *exact* quotient is the side
      // condition divisibilityOf() collects for whichever comparison encloses
      // this node.
      return toInt(*node.operands()[0]) / toInt(*node.operands()[1]);
    default:
      llvm_unreachable("boolean node translated as arithmetic");
    }
  }

  /// `b divides a` for every Div `a / b` at or below `node`, conjoined; absent
  /// when there is no division below it.
  ///
  /// This mirrors the `exact` mask evalNodeVec threads through its recursion.
  std::optional<Gecode::BoolExpr>
  divisibilityOf(const ConstraintNode &node) const {
    std::optional<Gecode::BoolExpr> acc;
    auto conjoin = [&acc](Gecode::BoolExpr e) {
      acc = acc ? Gecode::BoolExpr(*acc && e) : e;
    };

    // A division under a BoolAsInt falsifies that boolean, making the node 0;
    // hoisting it here would falsify the enclosing comparison instead, which
    // is a different (and wrong) claim.
    if (node.kind == Kind::BoolAsInt)
      return acc;

    if (node.kind == Kind::Div)
      conjoin(toInt(*node.operands()[0]) % toInt(*node.operands()[1]) == 0);
    for (const ConstraintNodePtr &child : node.operands())
      if (std::optional<Gecode::BoolExpr> sub = divisibilityOf(*child))
        conjoin(*sub);
    return acc;
  }

  /// `e`, conjoined with the divisibility side conditions of every division
  /// below `node`.
  Gecode::BoolExpr withDivisibility(Gecode::BoolExpr e,
                                    const ConstraintNode &node) const {
    if (std::optional<Gecode::BoolExpr> exact = divisibilityOf(node))
      return e && *exact;
    return e;
  }

  Gecode::BoolExpr toBool(const ConstraintNode &node) const {
    if (node.kind == Kind::Implies)
      // Each side carries its own divisibility conditions, so nothing about
      // this node's operands escapes to the top level -- which is what the
      // DSL means by a division under an `implies` not being reified.
      return toBool(*node.operands()[0]) >> toBool(*node.operands()[1]);

    if (node.kind == Kind::Divides) {
      // `operands[0]` divides `operands[1]`. A test, so it reports being
      // inexact rather than being falsified by it -- but a `/` *inside* an
      // operand still has to come out exact for the operand to mean anything.
      Gecode::BoolExpr test =
          toInt(*node.operands()[1]) % toInt(*node.operands()[0]) == 0;
      return withDivisibility(test, node);
    }

    assert(ConstraintNode::isBoolKind(node.kind) && "expected a boolean node");
    Gecode::LinIntExpr lhs = toInt(*node.operands()[0]);
    Gecode::LinIntExpr rhs = toInt(*node.operands()[1]);
    Gecode::BoolExpr cmp = [&]() -> Gecode::BoolExpr {
      switch (node.kind) {
      case Kind::Le:
        return lhs <= rhs;
      case Kind::Ge:
        return lhs >= rhs;
      case Kind::Lt:
        return lhs < rhs;
      case Kind::Gt:
        return lhs > rhs;
      case Kind::Eq:
        return lhs == rhs;
      case Kind::Ne:
        return lhs != rhs;
      default:
        llvm_unreachable("unknown ConstraintNode::Kind");
      }
    }();
    return withDivisibility(cmp, node);
  }
};

// ===----------------------------------------------------------------------===//
// The model
// ===----------------------------------------------------------------------===//

class SpaceModel : public Gecode::Space {
public:
  Gecode::IntVarArray x;

  SpaceModel(llvm::ArrayRef<SearchParam> params,
             llvm::ArrayRef<ConstraintNodePtr> constraints)
      : x(*this, static_cast<int>(numDimensions(params))) {
    Gecode::IntVarArgs vars;
    for (const SearchParam &param : params) {
      // One variable per *dimension*, all sharing the parameter's domain: a
      // parameter of arity n is n entries of a Configuration, and every index
      // reaching this file -- a Var node's, a divisibility relation's -- is
      // already an index into those entries rather than into `params`.
      Gecode::IntVarArgs own;
      for (size_t k = 0, e = param.arity(); k < e; ++k) {
        Gecode::IntVar v(*this, domainOf(param));
        x[vars.size()] = v;
        vars << v;
        own << v;
      }
      // What makes n places an ordering. Posted here rather than written by
      // whoever declares the parameter: it follows from the *kind*, holds for
      // every permutation parameter there will ever be, and stating it in the
      // DSL would mean a node whose only purpose is to be translated back into
      // this call.
      if (param.kind() == ParamKind::Permutation && own.size() > 1)
        Gecode::distinct(*this, own);
    }

    Translator translator{*this, vars};
    for (const ConstraintNodePtr &node : constraints)
      Gecode::rel(*this, translator.toBool(*node));

    // First-fail: assign the most constrained variable first. The enumeration
    // order this produces is not the order solutions come back in -- they are
    // sorted -- so the heuristic is free to be chosen for search size alone.
    Gecode::branch(*this, x, Gecode::INT_VAR_SIZE_MIN(), Gecode::INT_VAL_MIN());
  }

  SpaceModel(SpaceModel &s) : Gecode::Space(s) { x.update(*this, s.x); }
  Gecode::Space *copy(void) override { return new SpaceModel(*this); }
};

} // namespace

SolveResult solveSpace(llvm::ArrayRef<SearchParam> params,
                       llvm::ArrayRef<ConstraintNodePtr> constraints,
                       const SolveOptions &opts) {
  SolveResult result;
  if (params.empty())
    return result;

  std::unique_ptr<SpaceModel> model;
  try {
    model = std::make_unique<SpaceModel>(params, constraints);
  } catch (const Gecode::Exception &e) {
    // Overflow is the case that reaches here: the solver has to represent the
    // *bounds* of every subexpression, and a product of several parameters can
    // exceed what an int variable holds even when no feasible configuration
    // does. The vectorized evaluator wraps around silently instead, so a space
    // that appeared to work can start reporting this.
    result.error =
        std::string("could not build the constraint model: ") + e.what();
    return result;
  }

  Gecode::Search::Options options;
  options.threads = static_cast<double>(opts.threads);
  std::unique_ptr<Gecode::Search::Stop> stop;
  if (opts.nodeLimit) {
    stop.reset(new Gecode::Search::NodeStop(opts.nodeLimit));
    options.stop = stop.get();
  }

  try {
    Gecode::DFS<SpaceModel> engine(model.get(), options);
    const size_t numDims = numDimensions(params);
    Configuration conf(numDims);
    while (SpaceModel *solution = engine.next()) {
      for (size_t d = 0; d < numDims; ++d)
        conf[d] =
            static_cast<ParmValue>(solution->x[static_cast<int>(d)].val());
      result.solutions.push_back(conf);
      delete solution;
      if (opts.solutionLimit && result.solutions.size() >= opts.solutionLimit) {
        result.complete = false;
        break;
      }
    }
    if (engine.stopped())
      result.complete = false;
    Gecode::Search::Statistics stats = engine.statistics();
    result.nodes = stats.node;
    result.failures = stats.fail;
  } catch (const Gecode::Exception &e) {
    result.error = std::string("constraint solving failed: ") + e.what();
    return result;
  }

  // See SolveResult::solutions: the index has to belong to the space, not to
  // the search that produced it.
  llvm::sort(result.solutions);
  return result;
}

} // namespace mlir::cinm::constraints

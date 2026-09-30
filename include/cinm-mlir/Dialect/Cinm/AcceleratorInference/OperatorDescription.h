#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <llvm/Support/JSON.h>

namespace mlir::cinm {

/// What a compute block computes, as data: the description of it that a tool
/// outside the compiler (a TVM schedule, say) can rebuild the computation
/// from without parsing MLIR. `reference` is the block in the form the search
/// space is read off (prepareReferenceModule), i.e. a sequence of
/// `linalg.generic`s over the block's arguments:
///
///   { "args":    [{"name": "arg0", "shape": [8, 256], "dtype": "int8",
///                  "static": false}, ...],
///     "ops":     [{"name": "t0", "kind": "cinm.op.gemm",
///                  "domain": [8, 512, 256],
///                  "iterators": ["parallel", "parallel", "reduction"],
///                  "inputs": [{"value": "arg0", "map": [0, 2]}, ...],
///                  "inits":  [{"fill": 0, "map": [0, 1]}],
///                  "results": [{"shape": [8, 512], "dtype": "int32"}],
///                  "body": [{"name": "v0", "op": "arith.extsi",
///                            "args": ["in0"], "dtype": "int32"}, ...],
///                  "yield": ["v3"],
///                  "expr": ["addi(out0, muli(extsi(in0), extsi(in1)))"]},
///                 ...],
///     "results": ["t1"],
///     "mlir":    "<the block, printed with large constants elided>" }
///
/// Values are named: block arguments `argN`, op results `tN` (`tN.K` past
/// the first), body arguments `inK` and `outK`, body values `vN`. A map is
/// the list of iteration dimensions each operand dimension is indexed by, or
/// `{"affine_map": "..."}` when the indexing is not a projected permutation.
/// An init that a constant fill produces is `{"fill": <value>}` in place of
/// the fill op. Ops that are not `linalg.generic` (a reshape, a constant)
/// are listed with their operands and result types only, under
/// `"kind": "other"`; `mlir` then says what they are.
llvm::json::Value describeComputeBlock(ComputeBlockOp reference);

/// An element type (or a shaped type's element type) in the spelling the
/// description uses: TVM's and numpy's, `int8`, `float32`, ...
std::string describeElementType(Type type);

} // namespace mlir::cinm

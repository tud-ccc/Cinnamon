
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>
namespace mlir::cnm {

/// Reshape the workgroup by turning a dimension D at index `dim`
/// into two dimensions, of size `factor` and `D/factor`. Fails if
/// D is not divisible by factor.
/// ExpandWorkgroupDim(dim=0, factor=2) turns
/// ```
/// cnm.launch
///  ins(%a[(i)->(i)]: memref<1024xi32>)
///  outs(%o[(i)->(i)]: memref<1024xi32>)
///  on hierarchy<1024>
///  do (%a1: memref<i32>, %o1: memref<i32>)  {
///       %x = memref.load %a1[]
///       %t2 = arith.muli %x, 2
///       memref.store %t2, %o1[]
///   }
/// ```
/// into
/// ```
/// cnm.launch
///  ins(%a[(i,j)->(i*512+j)]: memref<1024xi32>)
///  outs(%o[(i,j)->(i*512+j)]: memref<1024xi32>)
///  on hierarchy<2x512>
///  do (%a1: memref<i32>, %o1: memref<i32>)  {
///       %x = memref.load %a1[]
///       %t2 = arith.muli %x, 2
///       memref.store %t2, %o1[]
///   }
/// ```
LogicalResult expandWorkshoupDim(cnm::ComputeOp compute, uint64_t dim,
                                 int64_t factor);

/// Turn the leftmost dimension of the workgroup into an outer parallel loop.
/// This transformation might delete the op if the workgroup has only a single
/// dimension.
/// ```
/// cnm.launch
///  ins(%a[(i,j)->(i*512+j)]: memref<1024xi32>)
///  outs(%o[(i,j)->(i*512+j)]: memref<1024xi32>)
///  on hierarchy<2x512>
///  do (%a1: memref<i32>, %o1: memref<i32>)  {
///       %x = memref.load %a1[]
///       %t2 = arith.muli %x, 2
///       memref.store %t2, %o1[]
///   }
/// ```
/// into
/// ```
/// affine.parallel (%i) = (0) to (2) {
///  %as = memref.subview %a[%i][512][1]
///  %os = memref.subview %o[%i][512][1]
///  cnm.launch
///      ins(%as[(j)->(j)]: memref<512xi32>)
///      outs(%os[(j)->(j)]: memref<512xi32>)
///      on hierarchy<512>
///      do (%a1: memref<i32>, %o1: memref<i32>) {
///           %x = memref.load %a1[]
///           %t2 = arith.muli %x, 2
///           memref.store %t2, %o1[]
///      }
/// }
/// ```
FailureOr<std::optional<cnm::ComputeOp>> peelLeft(OpBuilder &builder,
                                                  cnm::ComputeOp compute);

/// Turn the rightmost dimension of the workgroup into a
/// parallel loop within the kernel.
/// This transformation might delete the op if the workgroup
/// has only a single dimension.
/// ```
/// cnm.launch
///  ins(%a[(i,j)->(i*512+j)]: memref<1024xi32>)
///  outs(%o[(i,j)->(i*512+j)]: memref<1024xi32>)
///  on hierarchy<2x512>
///  do (%a1: memref<i32>, %o1: memref<i32>)  {
///       %x = memref.load %a1[]
///       %t2 = arith.muli %x, 2
///       memref.store %t2, %o1[]
///   }
/// ```
/// into
/// ```
/// %as = memref.reshape %a: (mr<1024xi32>) ...
/// %os = memref.reshape %o: (mr<1024xi32>) ...
/// cnm.launch
///   ins(%as[(i)->(i)]: memref<2x512xi32>)
///   outs(%os[(i)->(i)]: memref<2x512xi32>)
///   on hierarchy<2>
///   do (%a1: memref<512xi32>,
///       %o1: memref<512xi32>)  {
///     affine.parallel (%i) = (0) to (512) {
///       %x = memref.load %a1[%i]
///       %t2 = arith.muli %x, 2
///       memref.store %t2, %o1[%i]
///     }
///   }
/// ```
FailureOr<std::optional<cnm::ComputeOp>> peelRight(OpBuilder &builder,
                                                   cnm::ComputeOp compute);
} // namespace mlir::cnm
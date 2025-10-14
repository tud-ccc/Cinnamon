//===- Ops.td - Memristor dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Memristor dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MEMRISTOR_OPS
#define MEMRISTOR_OPS


include "cinm-mlir/Dialect/Memristor/IR/MemristorBase.td"
include "cinm-mlir/Dialect/Memristor/IR/MemristorTypes.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/IR/CommonAttrConstraints.td"


def WriteToCrossbarOp : Memristor_Op<"write_to_crossbar"> {
  let summary = "Write a matrix into a Memristor crossbar";
  let description = [{
    The "write_to_crossbar" operation writes the provided matrix into
    the crossbar specified by its tile ID.
    The shapes of the matrix is expected to fit into the crossbar.
    The operation expects contiguous memref.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      memristor.write_to_crossbar(%tileId, %B) : i32, memref<?x?xi32>
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      std::string name = "memristor_write_to_crossbar";

      llvm::raw_string_ostream ss(name);
      appendOperandPrecision(ss, getB().getType());

      return ss.str();
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId,
                       AnyStridedMemRefOfRank<2>:$B);

  let assemblyFormat = "$tileId `,` $B attr-dict `:` qualified(type($tileId)) `,` type($B)";
}

def GemmOp : Memristor_Op<"gemm"> {
  let summary = "Perform general matrix-matrix multiplication on Memristor";
  let description = [{
    The "gemm" operation performs matrix-matrix multiplication between
    the first matrix and the one stored on a crossbar. The result is
    stored in the second matrix.
    The tile ID specifies which crossbar executes the operation.
    The shapes of the matricies are expected to be valid.
    The operation expects contiguous memrefs.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      memristor.write_to_crossbar(%tileId, %B) : i32, memref<?x?xi32>
      memristor.gemm(%tileId, %A, %C) : i32, memref<?x?xi32>, memref<?x?xi32>
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      std::string name = "memristor_gemm";

      llvm::raw_string_ostream ss(name);
      appendOperandPrecision(ss, getC().getType());

      return ss.str();
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId,
                       AnyStridedMemRefOfRank<2>:$A,
                       AnyStridedMemRefOfRank<2>:$C);

  let assemblyFormat = "$tileId `,` $A `,` $C attr-dict `:` qualified(type($tileId)) `,` type($A) `,` type($C)";
}

def GevmOp : Memristor_Op<"gevm"> {
  let summary = "Perform general vector-matrix multiplication on Memristor";
  let description = [{
    The "gevm" operation performs vector-matrix multiplication between
    the first vector and the matrix stored on a crossbar. The result is
    stored in the second vector.
    The tile ID specifies which crossbar executes the operation.
    The shapes of the vector and matricies are expected to be valid.
    The operation expects contiguous memrefs.

    Example:

    ```mlir
      // C(n) = A(k) * B(k, n)
      memristor.write_to_crossbar(%tileId, %B) : i32, memref<?x?xi32>
      memristor.gevm(%tileId, %A, %C) : i32, memref<?xi32>, memref<?xi32>
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      std::string name = "memristor_gevm";

      llvm::raw_string_ostream ss(name);
      appendOperandPrecision(ss, getC().getType());

      return ss.str();
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId,
                       AnyStridedMemRefOfRank<1>:$A,
                       AnyStridedMemRefOfRank<1>:$C);

  let assemblyFormat = "$tileId `,` $A `,` $C attr-dict `:` qualified(type($tileId)) `,` type($A) `,` type($C)";
}

def BarrierOp : Memristor_Op<"barrier"> {
  let summary = "Wait until the specified Memristor tile finishes computation";
  let description = [{
    The "barrier" operation blocks program execution until computation on
    the specified tile is finished.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      memristor.write_to_crossbar(%tileId, %B) : i32, memref<1x?xi32>
      memristor.gemm(%tileId, %A, %C) : i32, memref<?x?xi32>, memref<?x?xi32>

      // Wait until computation is finished
      memristor.barrier(%tileId) : i32
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      return "memristor_barrier";
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId);

  let assemblyFormat = "$tileId attr-dict `:` qualified(type($tileId))";
}

#endif

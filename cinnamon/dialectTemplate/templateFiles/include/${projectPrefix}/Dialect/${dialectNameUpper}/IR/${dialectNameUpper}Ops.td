//===- Ops.td - ${dialectNameUpper} dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the ${dialectNameUpper} dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef ${dialectNameAllCaps}_OPS
#define ${dialectNameAllCaps}_OPS


include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Base.td"
include "${projectPrefix}/Dialect/${dialectNameUpper}/IR/${dialectNameUpper}Types.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"



#endif

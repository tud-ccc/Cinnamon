//===- Types.td - Cnm dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_TYPES
#define CNM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "CnmBase.td"

def WorkgroupType : Cnm_Type<"Workgroup"> {
  let mnemonic = "workgroup";
  let summary = "";

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);

  let hasCustomAssemblyFormat = 1;
}

def BufferType : Cnm_Type<"Buffer"> {
  let mnemonic = "buffer";
  let summary = "";

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "Type":$element_type, "int64_t":$level);

  let hasCustomAssemblyFormat = 1;
}

def ScatterTokenType : Cnm_Type<"ScatterToken"> {
  let mnemonic = "scatter_token";
  let summary = "";
}

def GatherTokenType : Cnm_Type<"GatherToken"> {
  let mnemonic = "gather_token";
  let summary = "";
}

def LaunchTokenType : Cnm_Type<"LaunchToken"> {
  let mnemonic = "launch_token";
  let summary = "";
}

#endif

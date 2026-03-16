//===- Types.td - Cinm dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_TYPES
#define CINM_TYPES

include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"


def AcceleratorRefType : Cinm_Type<"AcceleratorRef"> {
  let mnemonic = "accelerator";
  let summary = "A type to represent an accelerator as a value";

  let extraClassDeclaration = [{
    
  }];
}

#endif

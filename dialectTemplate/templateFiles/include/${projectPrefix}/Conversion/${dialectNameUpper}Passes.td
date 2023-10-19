//===- Passes.td - ${dialectNameUpper} dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the ${dialectNameUpper} dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef ${dialectNameAllCaps}_CONVERSION_PASSES
#define ${dialectNameAllCaps}_CONVERSION_PASSES

include "PassUtil.td"

def Convert${dialectNameUpper}ToLLVM : ConvertToLlvm<"${dialectNameUpper}", "${dialectNs}">;


#endif // ${dialectNameAllCaps}_CONVERSION_PASSES
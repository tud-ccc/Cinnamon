//===- Passes.td - Cnm dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_TRANSFORM_PASSES
#define CNM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

// def CnmApplyTransformScriptPass : Pass<"cnm-apply-transform", "ModuleOp"> {
//   let summary = "Apply a transform dialect script to cnm compute ops";
//   let description = [{}];
// //  let constructor = "cnm::createApplyTransformScriptPass()";
//   let dependentDialects = ["transform::TransformDialect"];
//    let options = [
//     Option<"debugPayloadRootTag", "debug-payload-root-tag", "std::string",
//            /*default=*/[{""}],
//            "Select the operation with 'transform.target_tag' attribute having "
//            "the given value as payload IR root. If empty select the pass "
//            "anchor operation as the payload IR root.">,
//     Option<"disableExpensiveChecks", "disable-expensive-checks", "bool",
//            "false",
//            "Disable expensive checks in the interpreter for a faster run.">,
//     Option<"transformFileName", "transform-file-name", "std::string", "", 
//            "File name of the transform script">,
//     Option<"entryPoint", "entry-point", "std::string",
//            /*default=*/[{transform::TransformDialect::kTransformEntryPointSymbolName.str()}],
//            "Entry point of the pass pipeline.">,
//     Option<"debugTransformRootTag", "debug-transform-root-tag", "std::string",
//            /*default=*/[{""}],
//           "Select the operation with 'transform.target_tag' attribute having "
//           "the given value as container IR for top-level transform ops. This "
//           "allows user control on what transformation to apply. If empty, "
//           "select the container of the top-level transform op."
//        >,
//  ListOption<"transformLibraryPaths", "transform-library-paths", "std::string",
//       "Optional paths to files with modules that should be "
//                      "merged into the transform module to provide the "
//                      "definitions of external named sequences.">
//   ];
// }

#endif // CNM_TRANSFORM_PASSES

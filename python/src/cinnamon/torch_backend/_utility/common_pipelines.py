class CommonPipelines:
    LOWER_TORCH_TO_LINALG_UNCHECKED_PIPELINE = [
        "func.func(torch-restructure-non-constant-axes)",
        "func.func(torch-fuse-quantized-ops)",
        "func.func(torch-scalarize-shapes)",
        "func.func(convert-torch-to-tmtensor)",
        "func.func(canonicalize)",
        "func.func(convert-torch-to-linalg)",
        "func.func(canonicalize)",
        "func.func(convert-torch-to-scf)",
        "func.func(convert-torch-to-arith)",
        "func.func(convert-torch-to-tensor)",
        "convert-torch-conversion-to-mlprogram",
        "memref-expand",
        "canonicalize",
        "resolve-shaped-type-result-dims",
        "cse",
        "torch-func-backend-type-conversion",
        "canonicalize",
        "func.func(torch-finalizing-backend-type-conversion)",
    ]

    LOWER_TORCH_TO_LINALG_CHECKED_PIPELINE = (
        LOWER_TORCH_TO_LINALG_UNCHECKED_PIPELINE
        + ["torch-verify-linalg-on-tensors-backend-contract"]
    )

    LOWER_LINALG_TO_LLVM_PIPELINE = [
        "func.func(llvm-request-c-wrappers)",
        "convert-tensor-to-linalg",
        "linalg-generalize-named-ops",
        "one-shot-bufferize{copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        "refback-mlprogram-bufferize",
        "canonicalize",
        "convert-bufferization-to-memref",
        "convert-linalg-to-loops",
        "convert-scf-to-cf",
        "convert-arith-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "convert-index-to-llvm",
        "expand-strided-metadata",
        "memref-expand",
        "finalize-memref-to-llvm",
        "canonicalize",
        "convert-to-llvm",
        "reconcile-unrealized-casts",
        "llvm-legalize-for-export",
    ]

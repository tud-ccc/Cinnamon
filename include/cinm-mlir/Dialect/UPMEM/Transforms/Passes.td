//===- Passes.td - UPMEM dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_TRANSFORM_PASSES
#define UPMEM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def UPMEMDedupKernelsPass: Pass<"upmem-dedup-kernels", "ModuleOp"> {
  let summary = "Deduplicate the DPU kernels within an upmem module";
  let description = [{
  }];
  let dependentDialects = [
    "::mlir::upmem::UPMEMDialect"
  ];
}

def simulatorClValues {
  string values = [{
  ::llvm::cl::values(
    clEnumValN(UpmemSimulatorId::CYCLE_ACCURATE, "cycle-accurate", "Cycle accurate simulator"),
    clEnumValN(UpmemSimulatorId::FAST, "fast", "Fast simulator based on the cycle-accurate one, may be imprecise"),
    clEnumValN(UpmemSimulatorId::HYBRID, "hybrid", "Cycle accurate, falling back on the hybrid model on timeout"),
    clEnumValN(UpmemSimulatorId::OPCOUNT, "op-count", "Operation counting, broken because doesn't report time in milliseconds")
    )}];
}

def UpmemInferAcceleratorPass : Pass<"upmem-infer-accelerator", "ModuleOp"> {
  let summary = "Infer UPMEM accelerator parameters for cinm.compute ops";
  let description = [{
    For each cinm.compute op that declares a UPMEM platform via
    cinm.available_platforms, runs Bayesian optimization to find the best
    combination of ranks, dpus_per_rank, tasklets_per_dpu, and tiling factors.
    Annotates the op with the resulting accelerator attribute and sets
    cinm.tile_sizes on each tileable op in the compute body.
  }];
  let dependentDialects = [
    "::mlir::cinm::CinmDialect",
    "::mlir::cnm::CnmDialect",
    "::mlir::upmem::UPMEMDialect",
    "::mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::affine::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect"
  ];
  let options = [
    // InferenceOptions fields
    Option<"maxEvals", "max-evals", "int", "50",
           "Total number of valid BO evaluations (LHS init + surrogate-guided)">,
    Option<"nInit", "n-init", "int", "10",
           "Number of LHS evaluations before the surrogate model takes over">,
    Option<"rngSeed", "rng-seed", "int", "42",
           "RNG seed for reproducibility">,
    Option<"nSeeds", "n-seeds", "int", "1",
           "Number of independent BO seeds to run concurrently in one process, "
           "sharing the config space / valid-config scan / validation set. Each "
           "seed dumps to a seed_<value>/ subdirectory. 1 = single-seed">,
    Option<"kappa", "kappa", "double", "2.0",
           "UCB exploration weight for the BANANAS surrogate">,
    Option<"epochs", "epochs", "int", "1800",
           "Training epochs per MLP ensemble member">,
    Option<"nEnsemble", "n-ensemble", "int", "5",
           "Number of MLP ensemble members">,
    Option<"hidden", "hidden-width", "int", "32",
           "Hidden layer width of each ensemble MLP">,
    Option<"depth", "hidden-depth", "int", "4",
           "Number of hidden layers in each ensemble MLP">,
    Option<"neighborDepth", "neighbor-depth", "unsigned", "2",
           "Number of discrete steps away from observed points to include as "
           "candidates (1 = immediate neighbors only)">,
    Option<"neighborFrontierOnly", "neighbor-frontier-only", "bool", "false",
           "When true, only add candidates at exactly neighbor-depth steps away; "
           "otherwise add all points within that distance">,
    // UPMEM-specific options
    Option<"annotateOpCosts", "annotate-op-costs", "bool", "false",
           "Annotate each op in the committed region with its simulated cost "
           "via the 'upmem.sim_cost' attribute">,
    Option<"dumpDir", "dump-dir", "std::string", "\"\"",
           "If non-empty, write a pool.csv file to this directory containing "
           "all candidate configurations, their observed costs, and "
           "surrogate model statistics (mu, sigma, acq) after inference">,
    Option<"exhaustiveSearch", "exhaustive-search", "bool", "false",
           "Evaluate every valid configuration in the search space exhaustively "
           "instead of running Bayesian optimisation">,
    Option<"sampleN", "sample-n", "unsigned", "0u",
           "If > 0, evaluate a random sample of this many valid configurations "
           "instead of the full exhaustive scan or Bayesian optimisation -- a "
           "cheap alternative to exhaustive-search when only a small "
           "ground-truth sample is needed. Takes priority over "
           "exhaustive-search if both are set">,
    Option<"sampleMaxCostMs", "sample-max-cost-ms", "double", "2000.0",
           "When sample-n > 0, reject (and resample past) candidates predicted "
           "to cost more than this many ms, so the sample isn't dominated by "
           "configs that would take minutes on real hardware">,
    Option<"nValidation", "n-validation", "int", "0",
           "Number of held-out validation points sampled before BO begins; "
           "0 disables surrogate validation">,
    Option<"validationInterval", "validation-interval", "int", "1",
           "Record surrogate predictions on the validation set every N BO "
           "iterations">,
    Option<"objectiveScale", "objective-scale", "std::string", "\"log10\"",
           "Cost transform before surrogate training: linear, log2, log10, ln, sqrt, cbrt">,
    Option<"useMRAMTiling", "use-mram-tiling", "bool", "true",
           "Enable MRAM tiling in the UPMEM lowering">,
    Option<"simulator", "simulator", "UpmemSimulatorId", "UpmemSimulatorId::CYCLE_ACCURATE",
           "Simulator backend",
           simulatorClValues.values>,
    Option<"evalTimeoutMs", "eval-timeout-ms", "int", "0",
           "Wall-clock timeout in milliseconds for a single simulation; "
           "0 disables the timeout (default: 0)">,
    Option<"dumpFullPool", "dump-full-pool", "bool", "false",
           "Dump the full pool (all valid configurations) into pool.csv, otherwise only the visited configurations are dumped.">,
    Option<"numWorkers", "n-workers", "unsigned", "0u",
           "Worker threads for exhaustive search (0 = hardware_concurrency)">,
    Option<"fixedDpus", "fixed-dpus", "int64_t", "-1",
           "Pin the number of DPUs to this value instead of searching over it "
           "(-1 = search normally). Remaining dimensions are still searched, "
           "so this fixes the working group while letting the tiling be chosen "
           "for it -- useful for comparing codegen quality at a fixed group size">,
    Option<"fixedTasklets", "fixed-tasklets", "int64_t", "-1",
           "Pin the number of tasklets to this value instead of searching over "
           "it (-1 = search normally). See fixed-dpus.">,
    ListOption<"evalSolution", "eval-solution", "int64_t",
               "Evaluate this single configuration (one int per space parameter) "
               "and commit it, bypassing search entirely">
  ];
}

def UpmemAnnotateCostsPass
    : Pass<"upmem-annotate-costs"> {
  let summary = "Annotate UPMEM ops with estimated simulation costs";
  let description = [{
    Runs the selected cost estimator on each cinm.compute_block body and
    annotates each op with its estimated cost via the 'upmem.sim_cost'
    float attribute.  Use 'opcount' for a fast weighted heuristic or
    'cycleaccurate' for the cycle-accurate upmem_simulator Python backend.
  }];
  let dependentDialects = [
    "::mlir::cinm::CinmDialect",
    "::mlir::upmem::UPMEMDialect"
  ];
  let options = [
    Option<"simulator", "simulator", "UpmemSimulatorId", "UpmemSimulatorId::CYCLE_ACCURATE",
           "Simulator backend",
           simulatorClValues.values>,
    Option<"evalTimeoutMs", "eval-timeout-ms", "int", "0",
           "Wall-clock timeout in milliseconds for a single simulation; "
           "0 disables the timeout (default: 0)">,
    Option<"costsCsv", "costs-csv", "std::string", "\"\"",
           "If non-empty, write a detailed cost breakdown (one row per "
           "compute_block x cost category x sub-label, e.g. transfer/scatter) "
           "to this CSV path">,
  ];
}

def UpmemLegalizeForTranslationPass
    : Pass<"upmem-legalize-for-translation", "ModuleOp"> {
  let summary = "Strip all ops except upmem.dpu_program for DPU translation";
  let description = [{
    Erases every top-level op in the module that is not a upmem.dpu_program,
    leaving a module that contains only DPU kernels and is ready for
    mlir-to-upmem-cpp translation.
  }];
  let dependentDialects = ["::mlir::upmem::UPMEMDialect"];
}

#endif // UPMEM_TRANSFORM_PASSES

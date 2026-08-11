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

def UpmemTileMRAMBuffersPass: Pass<"upmem-tile-mram-buffers"> {
  let summary = "Stage a launch body's non-leaf buffers down to the leaf memory level";
  let description = [{
    Inside a `cnm.launch`, operands may live in a memory level the compute
    elements cannot address directly -- MRAM, for UPMEM. This pass tiles each
    such op down to a leaf-sized tile and stages the tile through the leaf
    level, wrapping it in `cnm.local_transfer`s.

    It applies to any `linalg` op with pure buffer semantics whose operands are
    in a non-leaf level, rather than to a fixed set of ops: the tiling is
    `scf::tileUsingSCF` and the staging is `linalg::promoteSubViews`, with the
    memory space, allocation and copy op supplied by this pass. Which levels
    exist, which one is the leaf, and what memory space attribute denotes them
    all come from the accelerator, so no backend names appear in the logic.

    Tile sizes come from an `upmem.leaf_tile_sizes` attribute on the op when
    present (which is how the inference plugin passes a resolved configuration
    in), otherwise from the `tile-sizes` option. With neither, the op is staged
    whole -- correct when the buffer already fits the leaf level. The attribute
    is not `cinm.tile_sizes`: the Cinm dialect reserves that one for ops
    implementing CinmTilingInterface, and these are linalg ops.
  }];
  let dependentDialects = [
    "::mlir::cnm::CnmDialect",
    "::mlir::upmem::UPMEMDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for ops that carry no upmem.leaf_tile_sizes "
               "attribute. One per iteration-space dimension; 0 means do not "
               "tile.">,
    Option<"hoistOutputTransfers", "hoist-output-transfers", "bool", "true",
           "Stage the output tile outside the reduction loops rather than "
           "around the op. An output is indexed only by parallel dimensions, "
           "so its tile is the same on every trip of a reduction loop; staging "
           "it inside would transfer it back and forth once per trip.">
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
    Option<"debugPipeline", "debug-pipeline", "bool", "",
           "Print debug info about the internal pipelines">,
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
    Option<"graphAllocation", "graph-allocation", "bool", "false",
           "Run the graph-level two-level solve (profile each program class "
           "over the DPU rank menu, allocate the grid exactly across the "
           "graph's classes, stamp each group's winning configuration onto "
           "its members; docs/GraphOptimizationDesign.md) instead of "
           "searching every compute block independently with the whole "
           "device to itself. Incompatible with fixed-dpus, which pins the "
           "resource the profiling needs to sweep">,
    Option<"programReloadMs", "program-reload-ms", "double", "40.0",
           "Graph allocation only: cost of switching a DPU set to a "
           "different program, per op per inference -- what a timeshared "
           "(non-pinned) placement pays and pinning avoids">,
    Option<"fusionEdges", "fusion-edges", "bool", "true",
           "Declare a `fuse.<producer>-><consumer>` parameter for each "
           "producer/consumer pair of distributed ops in the block, coupling "
           "their tile sizes so that the configurations whose launches can be "
           "fused are dense in the space. No pass reads the parameter; turning "
           "this off only makes those configurations rarer, and shrinks the "
           "space">,
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
    Option<"nSolveWorkers", "n-solve-workers", "unsigned", "1u",
           "Worker threads used to solve the constraint system (0 = hardware_concurrency)">,
    Option<"fixedDpus", "fixed-dpus", "int64_t", "-1",
           "Pin the number of DPUs to this value instead of searching over it "
           "(-1 = search normally). Remaining dimensions are still searched, "
           "so this fixes the working group while letting the tiling be chosen "
           "for it -- useful for comparing codegen quality at a fixed group size">,
    Option<"fixedTasklets", "fixed-tasklets", "int64_t", "-1",
           "Pin the number of tasklets to this value instead of searching over "
           "it (-1 = search normally). See fixed-dpus.">,
    ListOption<"evalSolution", "eval-solution", "std::string",
               "Evaluate this single configuration and commit it, bypassing "
               "search entirely. Given as `name=value` pairs, e.g. "
               "`eval-solution=blockM=8,blockK=128`. Every parameter the space "
               "declares must be named; unknown names are an error. Named "
               "rather than positional so that changing the space cannot "
               "silently reinterpret a stored configuration.">
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

def UpmemCheckOccupancyPass
    : Pass<"upmem-check-occupancy", "ModuleOp"> {
  let summary = "Fail if a DPU program's buffers do not fit the device";
  let description = [{
    Sums what each `upmem.dpu_program` actually allocates -- static MRAM and
    WRAM buffers once per DPU, private WRAM once per tasklet plus the runtime's
    stack reserve -- and fails the pass if either exceeds the capacity the
    accelerator declares for that level.

    This is a *measurement*, not a model, and that is the point. How much
    memory a configuration ends up needing is not a property of the
    configuration: it moves with whether the tasklets of a DPU can share a
    buffer or must replicate it, with how promotion sizes its staging buffers,
    with buffer hoisting, kernel deduplication, and even with how well the
    affine simplifier folds a delinearized index. So it can only be known
    after every memory optimization has run, which is where this pass belongs
    -- last. See docs/CnmMemoryLevelsDesign.md §H3.

    The capacities come from the accelerator of the `cinm.compute_block` whose
    `upmem.alloc_dpus` loads the program, unless `mram-size`/`wram-size`
    override them. A program that no `upmem.alloc_dpus` refers to is dead and
    is not checked.
  }];
  let dependentDialects = [
    "::mlir::cinm::CinmDialect",
    "::mlir::upmem::UPMEMDialect"
  ];
  let options = [
    Option<"mramSize", "mram-size", "int64_t", "0",
           "MRAM capacity per DPU in bytes; 0 takes it from the accelerator">,
    Option<"wramSize", "wram-size", "int64_t", "0",
           "WRAM capacity per DPU in bytes; 0 takes it from the accelerator">,
  ];
}

def UpmemSpecializeTransfersPass
    : Pass<"upmem-specialize-transfers", "ModuleOp"> {
  let summary = "Narrow each transfer op to the cheapest form its map allows";
  let description = [{
    The transfer ops are one capability at three levels of specificity:
    `scatter_blocks`/`gather_blocks` give each DPU several blocks fetched
    independently (`dpu_push_sg_xfer`), `scatter_on_array`/`gather_from_array`
    give it one contiguous block (a flat memcpy), and `broadcast` gives every
    DPU the same bytes (one `dpu_broadcast_to`). Emitters produce the general
    form; this pass reads each op's map and narrows it where that is legal, so
    the rule lives in one place instead of in every emitter.

    Three rewrites, and they chain:

    - **blocks to one block.** When a DPU's blocks turn out to be adjacent and
      in order in the host buffer, they are one run, so one flat transfer
      moves them.
    - **the same bytes everywhere.** A map that does not depend on rank, DPU
      or block sends every DPU the same region. If that region also *is* the
      whole host buffer, `broadcast` delivers it.
    - **a uniform value, widened.** A DPU's blocks land back to back in MRAM,
      so when the transferred value is one repeated constant it does not
      matter where each block came from: a constant of the target buffer's own
      shape, broadcast once, puts the same bytes there. This is the only
      rewrite that grows the host side, and it only ever grows a constant.

    Gathers are never narrowed to a broadcast: two DPUs writing one host
    region is a race, not a broadcast.
  }];
  let options = [
    Option<"useSgXferCodegen", "use-sg-xfer-codegen", "bool", "true",
           "Allow the scatter/gather transfer API. When false, an op that "
           "cannot be narrowed out of the block form is an error: the "
           "packing that would have made it flat was supposed to happen "
           "upstream.">,
    Option<"useBcXferCodegen", "use-bc-xfer-codegen", "bool", "true",
           "Allow narrowing to upmem.broadcast.">,
  ];
  let dependentDialects = [
    "::mlir::upmem::UPMEMDialect",
    "::mlir::memref::MemRefDialect"
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

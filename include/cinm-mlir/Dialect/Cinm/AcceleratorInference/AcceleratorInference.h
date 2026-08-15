#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::cinm {
class SpaceBuilder;

// ===----------------------------------------------------------------------===//
// Accelerator inference
// ===----------------------------------------------------------------------===//
//
// The search itself: what a target has to supply to be searched over
// (InferencePlugin), what one trial looks like, and the entry point. The space
// being searched is ConfigSpace.h.

/// Per-trial context owned by the framework and passed to plugin callbacks.
/// Before evaluate() runs the pipeline, computeBlock is a live clone inside
/// module; after the pipeline lowers it away, computeBlock is invalid.
struct TrialInfo {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  cinm::ComputeBlockOp computeBlock;
  Configuration config;
  const ConfigSpace *space = nullptr;
  /// The evaluated cost of `config` (SimCost::total()). NaN until the trial
  /// has been evaluated; the framework fills it in when it records a best
  /// trial, so a search's result carries its own objective value.
  double cost = std::numeric_limits<double>::quiet_NaN();

  ConfWrapper conf() const { return ConfWrapper(*space, config); }
};

/// Footprint of one configuration in one memory level of the platform,
/// split by operand staticness (see isStaticValue). Stated per *instance* of
/// the level -- per DPU for a DPU-private memory, per tasklet for a
/// tasklet-private one -- the same unit the level's declared capacity uses.
struct LevelResidency {
  /// The level's name, matching a CinmLevelDefAttr of the platform.
  std::string level;
  /// Bytes that stay resident across inferences (pinned weights). When
  /// several ops share a device set, these all occupy the level at once and
  /// sum. Scratch levels that hold nothing between kernels report 0, which
  /// makes them never bind the packing.
  int64_t staticBytes = 0;
  /// Bytes of per-inference working set. Co-resident ops run sequentially,
  /// reuse one region, and take the max.
  int64_t dynBytes = 0;
};

/// What the device holds under a given configuration, per memory level: the
/// currency of the graph-level co-residency packing. `weightScatterMs` is
/// what scattering the static operands once costs -- the price a
/// *non*-pinned (timeshared) placement pays per inference, and what pinning
/// amortizes.
struct ResidencyInfo {
  SmallVector<LevelResidency> levels;
  double weightScatterMs = 0;

  /// The entry for `level`, or null if none was measured.
  const LevelResidency *find(llvm::StringRef level) const {
    for (const LevelResidency &entry : levels)
      if (entry.level == level)
        return &entry;
    return nullptr;
  }
};

// ===----------------------------------------------------------------------===//
// Plugin interface
// ===----------------------------------------------------------------------===//

/// Abstract plugin, one implementation per target.
/// Responsible for populating the config space and evaluating configurations.
/// The core framework calls these methods; target-specific logic lives here.
struct InferencePlugin {
  virtual ~InferencePlugin() = default;

  /// Populate the configuration space from the reference clone.
  /// The plugin decides what to add and how to explore the IR — it may walk
  /// the compute body, inspect op shapes, attach attributes to nodes, etc.
  /// Annotations left on the clone are inherited by every per-evaluation clone.
  ///
  /// The reference may also be *rewritten* here, and every trial then starts
  /// from the rewritten form: a plugin whose space is stated over some lowered
  /// form of the block can lower it once here instead of once per trial. The
  /// framework re-finds the compute block afterwards, so `refClone` itself
  /// need not survive.
  virtual void initializeSpace(cinm::ComputeBlockOp refClone,
                               cinm::SpaceBuilder &space) = 0;

  /// Evaluate a configuration. Lower total cost is better.
  /// `trial.computeBlock` is a fresh clone inside a minimal trial module
  /// (`module { func @host { clone } }`). The plugin annotates computeBlock,
  /// runs passes on `trial.module`, then returns a cost breakdown. The
  /// framework owns `trial`; the plugin must not retain references after
  /// returning.
  virtual utils::Maybe<utils::SimCost> evaluate(TrialInfo &trial) = 0;

  /// Called once after the best configuration has been found.
  /// `bestTrial.module` is the fully-lowered module from the winning evaluation
  /// (computeBlock is gone by this point). The plugin should splice the lowered
  /// code into the original module and replace `original` with it.
  virtual DiagnosedSilenceableFailure
  commitBestCandidate(cinm::ComputeBlockOp original, TrialInfo bestTrial);

  /// Return a fresh independent copy of this plugin, safe to use from a
  /// different thread. Called by the framework before parallel exhaustive
  /// search; `initializeSpace` has already run on `this` so any indices or
  /// space-derived state should be copied to the new instance.
  virtual std::unique_ptr<InferencePlugin> clone() const = 0;

  /// Optional hook called on each clone (on the main thread) before parallel
  /// evaluation begins. Use it to eagerly build pipelines or other state that
  /// is cheaper to construct single-threaded.
  virtual void warmUp(mlir::MLIRContext *) {}

  /// Whether this plugin is safe to evaluate concurrently from multiple
  /// threads. If false, exhaustive search will run single-threaded.
  virtual bool supportsMultithreading() const { return true; }

  /// Emit debug statistics (e.g. cache hit rate). Called after exhaustive
  /// search completes. Default is a no-op.
  virtual void printStats() const {}

  /// The name of the search parameter that counts the shared device resource
  /// the graph level allocates between compute blocks -- the DPU count for
  /// UPMEM. Cost profiles (profileComputeBlock) are indexed by this
  /// parameter's value. Empty when the target has no notion of graph-level
  /// allocation.
  virtual llvm::StringRef sharedResourceParam() const { return {}; }

  /// The resource values worth profiling `block` at. The plugin derives them
  /// from the block itself -- for UPMEM, divisors of the iteration-space
  /// size, since the workgroup must be filled exactly, quantized by
  /// InferenceOptions::allocationGranularity -- so different blocks get
  /// different menus. The values are candidates, not promises: a menu value
  /// the pinned search finds infeasible (capacity, say) becomes a hole in
  /// the profile. Only meaningful when sharedResourceParam() is non-empty.
  virtual SmallVector<int64_t>
  sharedResourceMenu(cinm::ComputeBlockOp block) const {
    (void)block;
    return {};
  }

  /// The most of the shared resource the device has at all: the total DPU
  /// count for UPMEM. This is the budget the graph-level allocation divides
  /// between device sets, and an upper bound on every menu value. 0 when the
  /// target has no notion of graph-level allocation.
  virtual int64_t sharedResourceMax() const { return 0; }

  /// Measure what the device holds under `trial`'s configuration, per
  /// memory level, for the co-residency packing and timeshare pricing of the
  /// graph-level allocation. Level names must match the platform's level
  /// declarations, whose capacities bound the packing. `trial` is a fresh,
  /// *unlowered* clone annotated with the configuration to measure; operand
  /// staticness is readable through isStaticValue (the framework forwards
  /// the original operands' staticness onto the trial module's function
  /// arguments). The default reports no footprints: a target without a
  /// residency model pins nothing, and the capacity check never rejects a
  /// packing.
  virtual ResidencyInfo measureResidency(TrialInfo &trial) {
    (void)trial;
    return {};
  }

  /// Materialize the device set a pinned group of compute blocks owns, at
  /// the builder's insertion point -- the graph level calls this once per
  /// group, at the top of the group's container function -- and return a
  /// handle whose type implements cinm::WorkgroupTypeInterface. `config`
  /// is the group's winning configuration, in evalSingleSolution currency;
  /// the plugin reads whatever parameters determine the set's shape (dpus
  /// and tasklets, for UPMEM). The framework forwards the handle into every
  /// member block as an operand, and the member's lowering uses it instead
  /// of allocating its own set (the forwarding contract; see CnmToUPMEM's
  /// findForwardedWorkgroup). Return null when the target has no hoisted
  /// allocation: members then allocate per block, as without graph
  /// allocation. The default is exactly that.
  virtual Value
  materializeWorkgroupAlloc(OpBuilder &builder, Location loc,
                            const llvm::StringMap<ParmValue> &config) {
    (void)builder;
    (void)loc;
    (void)config;
    return Value();
  }

  /// Release a device set materialized by materializeWorkgroupAlloc, at the
  /// builder's insertion point (the graph level calls this once per exit of
  /// the container function).
  virtual void materializeWorkgroupFree(OpBuilder &builder, Location loc,
                                        Value workgroup) {
    (void)builder;
    (void)loc;
    (void)workgroup;
  }
};

// ===----------------------------------------------------------------------===//
// Core framework API
// ===----------------------------------------------------------------------===//

struct InferenceOptions {
  /// Total number of valid evaluations (LHS init + surrogate-guided).
  int maxEvals = 100;
  /// Number of configurations evaluated in the LHS initialisation phase
  /// before the surrogate model takes over.  Must be ≤ maxEvals.
  int nInit = 20;

  int rngSeed = 42;

  /// If true, dumping stats will dump the entire valid space into the pool.csv.
  /// Otherwise pool.csv only contains the visited points, not the whole space.
  bool dumpFullPool = true;

  /// Number of independent BO seeds to run in one process. When > 1, the
  /// ConfigSpace, the valid-config scan, and the validation set are built once
  /// and shared; each seed then runs concurrently on its own thread (single-
  /// threaded per seed), capped by `numWorkers`. Seed values are derived from
  /// `rngSeed`. Each seed dumps to a `seed_<value>/` subdirectory. Has no
  /// effect in exhaustive or single-solution modes.
  int nSeeds = 1;

  /// How a round turns the surrogate's predictions into the candidates it
  /// evaluates.
  enum class Acquisition {
    /// Rank by mu - kappa*sigma and take the best. Deterministic given the
    /// surrogate, so the best q under it are the q points nearest one
    /// optimum -- fine for a single pick, redundant as a batch.
    LCB,
    /// One posterior draw per batch slot: score each candidate as
    /// mu + sigma*z with fresh z, and take that draw's minimum. Slots
    /// disagree wherever the ensemble does, so the batch spreads exactly as
    /// far as the surrogate is unsure and no further -- no diversity knob to
    /// tune, and identical to LCB in expectation for a single pick.
    Thompson,
  };
  Acquisition acquisition = Acquisition::LCB;

  /// Candidates selected and evaluated per surrogate fit. One round costs one
  /// fit whatever this is, so raising it amortises the fit and, when the
  /// evaluator has workers to spare, overlaps the evaluations. Values above 1
  /// want Thompson: LCB's top q are near-duplicates of each other.
  size_t boBatchSize = 1;

  // Surrogate model (BANANAS) hyperparameters.
  double kappa = 2.0; ///< UCB exploration weight
  int epochs = 5000;  ///< Training epochs per ensemble member
  int nEnsemble = 7;  ///< Number of MLP ensemble members
  int hidden = 64;    ///< Hidden layer width
  int depth = 2;      ///< Number of hidden layers

  bool sampleOnlyValid = true;
  /// Max number of candidate configs passed to the surrogate for ranking
  /// each round (neighbors of observed points + random draws).
  size_t nCandidates = 500;
  /// How many discrete steps away from observed points to include as
  /// candidates. 1 = immediate neighbors only; 2 = neighbors-of-neighbors, etc.
  unsigned neighborDepth = 1;
  /// When true, only the outermost frontier (exactly `neighborDepth` steps
  /// away) is added. When false, all points within `neighborDepth` steps are
  /// added.
  bool neighborFrontierOnly = false;
  /// Batch sizes at which per-round search diagnostics are computed: for each
  /// q, the top-q candidates under the acquisition are measured for clustering
  /// and for how much of the ranking the mean alone explains. The batch is
  /// hypothetical -- only the accepted candidates are evaluated -- so a run
  /// that selects one point per round still reports what a q-wide batch would
  /// have covered. Only computed when `dumpDir` is set (see rounds.csv and
  /// batchdiag.csv).
  std::vector<size_t> diagBatchSizes = {2, 4, 8, 16, 32, 64};
  /// If non-empty, dump the full candidate pool to a CSV file in this
  /// directory at the end of inference. Columns: one per search param,
  /// then observed cost (empty if not evaluated), then mu / sigma / acq
  /// from a final ensemble fit (omitted when fewer than 2 observations).
  std::string dumpDir;

  /// Build each block's config space, write its space.json into dumpDir, and
  /// stop: no evaluation, no search, no commit, and (in graph mode) no
  /// allocation. The IR is left untouched. This is how the space is made
  /// inspectable -- parameter docs, permutation encodings -- without paying
  /// for a run. Requires a non-empty dumpDir to be useful.
  bool dumpSpaceOnly = false;

  /// When true, evaluate every valid configuration in the search space
  /// instead of running Bayesian optimisation. Useful for collecting ground-
  /// truth cost data and comparing against BO solutions. The pool is dumped
  /// in the same CSV format as the BO run (surrogate columns are omitted
  /// since no model is trained).
  bool exhaustiveSearch = false;

  /// When > 0, evaluate a random sample of this many configurations (drawn via
  /// Latin Hypercube Sampling, see CandidatePool::sampleInitialSet) instead of
  /// the whole space or running Bayesian optimisation. A cheap alternative to
  /// exhaustiveSearch when only a small ground-truth sample is needed --
  /// exhaustive search's cost is entirely its one simulator call per
  /// configuration, so sampling down to sampleN evaluations makes this
  /// proportionally faster. Takes priority over exhaustiveSearch if both are
  /// set. Dumped the same way (dumpFullPool controls whether pool.csv includes
  /// unvisited configs too). See also sampleMaxCostMs.
  unsigned sampleN = 0;

  /// When sampleN > 0, a candidate predicted to cost more than this many ms
  /// is rejected (not counted towards sampleN, and never dumped) and
  /// resampled past -- without this, a uniform-random sample over the valid
  /// space routinely includes configs whose predicted (and, worse, actual
  /// on-hardware) cost is orders of magnitude above the rest of the sample,
  /// which is wasteful once every sampled config gets compiled and run on
  /// real hardware downstream.
  double sampleMaxCostMs = 0;

  /// Number of held-out validation points sampled (via LHS) before BO begins.
  /// These are evaluated once for their true cost and never used as BO training
  /// data. At each snapshot the surrogate's mu/sigma are recorded for them.
  /// Zero disables validation entirely.
  int nValidation = 0;
  /// Record a surrogate snapshot on the validation set every N BO iterations
  /// (Phase 2 iterations only). Has no effect when nValidation == 0.
  int validationInterval = 5;

  /// Transform applied to costs before surrogate training.
  /// Supported: "linear", "log2", "log10", "ln", "sqrt", "cbrt".
  std::string objectiveScale = "log10";

  /// Number of worker threads used for exhaustive search.
  /// 0 (default) means use std::thread::hardware_concurrency().
  unsigned numWorkers = 0;

  /// Number of workers used for parallel solving of the constraint system.
  /// A >1 value may yield slowdowns.
  unsigned nSolveWorkers = 1;

  /// When set, skip search entirely and evaluate only this single
  /// configuration. The values are in the same order as the ConfigSpace params
  /// populated by the plugin's initializeSpace(). Acts as a third mode
  /// alongside exhaustiveSearch and Bayesian optimisation.
  /// Evaluate exactly this configuration and commit it, bypassing search.
  /// Keyed by parameter name rather than by position: the space's variable
  /// order is an implementation detail of the handlers, and a positional
  /// encoding silently reinterprets every stored configuration when it
  /// changes. Resolved against the space once it has been built.
  std::optional<llvm::StringMap<ParmValue>> evalSingleSolution;

  /// With evalSingleSolution: skip the feasible-set membership check and
  /// attempt the lowering anyway. Membership is normally the whole check
  /// (the space holds exactly the feasible configurations), so forcing past
  /// it measures the constraint system itself: a rejected configuration
  /// that lowers and runs fine is a false negative of the space, which is
  /// exactly what the rejected-region experiment counts. Completeness and
  /// unknown-name checks still apply -- a value for every dimension is
  /// structural, not a feasibility judgement.
  bool evalSolutionForce = false;

  /// Parameters pinned to a single value when the space is built: each named
  /// parameter is constrained to equal the given value, on top of whatever
  /// the plugin declares. This is how a decision taken above the search is
  /// imposed on it -- the graph level fixing the device size for a profiling
  /// run, or re-searching under an allotted budget. A name the space does
  /// not declare is an error, not a no-op: a search that ignores a pin
  /// measures something other than what was asked.
  llvm::StringMap<ParmValue> pinnedParams;

  /// Run the graph-level two-level solve -- profile each program-identity
  /// class over the resource menu, allocate the device exactly across the
  /// graph, stamp each group's winning configuration onto its members --
  /// instead of searching every block independently with the whole device to
  /// itself. Requires a plugin that declares a shared resource; incompatible
  /// with externally pinning that resource (fixed-dpus), since the profiling
  /// pins it per menu value itself.
  bool graphAllocation = false;

  /// Graph allocation only: cost of switching a device set to a different
  /// program, per op per inference -- what a timeshared (non-pinned)
  /// placement pays and pinning avoids. Assumed constant (40 ms) until
  /// measured on hardware.
  double programReloadMs = 40.0;

  /// Graph allocation only: optimize single-inference latency (the makespan
  /// of the dependency graph) instead of steady-state throughput (the
  /// busiest set's per-inference work). The latency solve is a critical-path
  /// greedy and makes no claim to optimality; the throughput solve is exact.
  bool latencyObjective = false;

  /// Graph allocation only: the quantum of a device-set size, in resource
  /// units (DPUs). The menu prefers multiples of it -- rank-sized (or
  /// half-rank) allocations keep host<->device transfers rank-parallel --
  /// but it is a preference, not a constraint: when the problem size admits
  /// no such multiple, the menu falls back to what divides the problem.
  int64_t allocationGranularity = 64;

  /// Render terminal progress bars for this search. Progress is already
  /// self-suppressing when stdout is not a terminal; this turns it off even
  /// on one -- what a caller running many searches concurrently does, since
  /// interleaved bars from independent searches shred each other's renders.
  bool showProgress = true;
};

// ===----------------------------------------------------------------------===//
// Cost profiles
// ===----------------------------------------------------------------------===//

/// One measured point of a compute block's cost profile: the best cost a
/// search found with the shared resource pinned to `resource`, and the
/// configuration that achieved it. The profile is the complete summary of
/// the block that the graph-level allocation consumes: the only variables a
/// cross-block constraint ever mentions are the resource and the capacity
/// footprints, so conditioning on them separates the joint problem exactly.
struct ProfilePoint {
  /// The shared-resource value (sharedResourceParam) this point measured.
  int64_t resource;
  /// Total cost of the best configuration found at this resource, in ms.
  double costMs;
  /// The argmin configuration, keyed by space dimension name in the same
  /// currency as InferenceOptions::evalSingleSolution, so it can be replayed
  /// through a later search or evaluation without reinterpretation.
  llvm::StringMap<ParmValue> config;
  /// The argmin's residency summary (plugin-measured); consumed by the
  /// graph-level co-residency packing and timeshare pricing.
  ResidencyInfo residency;
};

/// Measure `computeOp`'s cost profile over the plugin's shared-resource menu
/// by running one search per menu value with the resource pinned. The menu
/// points are independent and run concurrently, each on its own clone of
/// `plugin`, when the plugin tolerates concurrent evaluation and the context
/// is multithreaded; `opts.numWorkers` caps the total across both levels
/// (the sweep and any parallelism inside a single search). `opts` otherwise
/// applies to each per-point search (maxEvals is per point); dumps, when
/// enabled, go to a `<param>_<value>/` subdirectory per point. Menu values
/// for which the pinned space has no valid configuration (divisibility,
/// capacity) yield no point rather than an error; the profile is measured
/// pointwise and the allocation copes with holes. Fails only when every menu
/// value is infeasible or a search fails definitively.
utils::Maybe<SmallVector<ProfilePoint>>
profileComputeBlock(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                    const InferenceOptions &opts);

/// Entry point for Bayesian inference.
DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts = {});

} // namespace mlir::cinm

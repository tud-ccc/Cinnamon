#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <condition_variable>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>
#include <mutex>
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

/// One resource value's device roofline, as the menu screen reads it: the
/// price, and the two terms it is the greater of.
struct DeviceRoofline {
  /// max(arithmetic, traffic), in ms -- what the screen compares to the host.
  double ms = 0.0;
  /// The traffic term alone, in ms: the scatter in and the gather out at
  /// this resource value, whose fixed cost grows with it.
  double transferMs = 0.0;
  /// The arithmetic term's rate, in ops/s across the whole resource value,
  /// for this block's element types. The arithmetic term is work / this.
  double opsPerSecond = 0.0;
  /// The traffic term the same block would pay with nothing held resident,
  /// in ms: every static operand sent on every call, as the host reads them
  /// from DRAM on every call. Nothing decides anything by it -- residency is
  /// the offload argument, and this is what the argument is worth. Recorded
  /// because it cannot be recovered downstream: it prices a byte count that
  /// no row of the dump has, and the transfer model's fixed cost is large
  /// enough at these sizes that extrapolating to it is out by several times.
  double transferMsIfNothingResident = 0.0;
};

/// Abstract plugin, one implementation per target.
/// Responsible for populating the config space and evaluating configurations.
/// The core framework calls these methods; target-specific logic lives here.
struct InferencePlugin {
  virtual ~InferencePlugin() = default;

  /// Rewrite a reference module (buildReferenceModule) into the form the
  /// search space is stated over: a plugin whose space is read off some
  /// lowered form of the block lowers it here, once per reference, instead
  /// of once per trial. Every trial is a clone of what this leaves, and so is
  /// every block sharedResourceMenu is asked about. The rewrite may replace
  /// the compute block op itself; the framework re-finds it afterwards. The
  /// result must not depend on the configuration. The default leaves the
  /// reference as it is.
  virtual LogicalResult prepareReference(ModuleOp reference) {
    (void)reference;
    return success();
  }

  /// Populate the configuration space from the reference clone, which
  /// prepareReference has already rewritten -- or, under
  /// InferenceOptions::stampConfigs, the original block itself, which the
  /// pass has put in that form up front.
  /// The plugin decides what to add and how to explore the IR — it may walk
  /// the compute body, inspect op shapes, attach attributes to nodes, etc.
  /// Annotations left on the clone are inherited by every per-evaluation clone.
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

  /// Commit under InferenceOptions::stampConfigs: write the winning
  /// configuration onto `original` as attributes (the accelerator, and the
  /// per-op parameters the lowering passes read), without splicing any
  /// lowered code. `original` carries the parameter-name annotations the
  /// space build left on it -- in stamp mode the space is built on the
  /// original itself and the reference is cloned from it -- so the plugin
  /// only has to resolve each name against `bestTrial`'s configuration.
  virtual DiagnosedSilenceableFailure
  stampBestCandidate(cinm::ComputeBlockOp original, TrialInfo &bestTrial) {
    (void)bestTrial;
    return emitDefiniteFailure(
        original.getLoc(),
        "this plugin does not implement configuration stamping");
  }

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

  /// The resource values worth profiling `reference` at: a compute block
  /// inside a reference module that prepareReference has rewritten, i.e. in
  /// the form the search space is read off. The plugin derives them
  /// from the block itself -- for UPMEM, divisors of the iteration-space
  /// size, since the workgroup must be filled exactly, quantized by
  /// InferenceOptions::allocationGranularity -- so different blocks get
  /// different menus. The values are candidates, not promises: a menu value
  /// the pinned search finds infeasible (capacity, say) becomes a hole in
  /// the profile. Only meaningful when sharedResourceParam() is non-empty.
  virtual SmallVector<int64_t>
  sharedResourceMenu(cinm::ComputeBlockOp reference) const {
    (void)reference;
    return {};
  }

  /// What this target's cost model says one invocation of `block` would take
  /// with the shared resource pinned to `resource`: a roofline, the greater
  /// of the device's arithmetic and its traffic. Nothing when the target has
  /// no such model, which turns the menu screen off (profileComputeBlock).
  ///
  /// It is a lower bound, so the screen drops a resource value only when an
  /// idealized device at that value loses to the host (cinm::hostSeconds).
  /// What makes that worth doing is that both terms move with `resource` --
  /// arithmetic down, the transfer's fixed cost up -- so the screen answers
  /// "at which sizes could this ever pay", which no single-valued gate can.
  ///
  /// The screen itself only needs `ms`. The terms it was taken from come
  /// back with it because the two of them are the roof this resource value
  /// was judged against -- its ceiling and its slope -- and a report of the
  /// screen that carries only their maximum cannot be read back as one
  /// (MenuPointTrace::verdict).
  virtual std::optional<DeviceRoofline>
  deviceRoofline(cinm::ComputeBlockOp block, int64_t resource) {
    (void)block;
    (void)resource;
    return std::nullopt;
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

/// A compute block on its own, in a module of its own:
/// `module { func @host(args) -> results { %r = <clone>; return %r } }`,
/// with the original operands' staticness forwarded onto the function's
/// arguments, so that isStaticValue resolves inside it exactly as it does on
/// the original. Every search clones its trials from one of these.
struct ReferenceModule {
  OwningOpRef<ModuleOp> module;
  cinm::ComputeBlockOp block;
};

/// Wrap a clone of `original` into a reference module, as it is before the
/// plugin has rewritten anything.
ReferenceModule buildReferenceModule(cinm::ComputeBlockOp original);

/// The reference module of `original` in the form the plugin states its
/// space over (InferencePlugin::prepareReference). Fails, with an error
/// emitted, when the plugin cannot rewrite it.
FailureOr<ReferenceModule> prepareReferenceModule(cinm::ComputeBlockOp original,
                                                  InferencePlugin &plugin);

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

  /// Graph profiling only: how many independent searches to run per menu
  /// point, for measuring how much of a profile's shape is search noise
  /// rather than scaling. Purely diagnostic -- the point handed to the
  /// allocator is always the first seed's, exactly what a single-seed run
  /// would have produced, so turning this up cannot move an allocation. The
  /// extra seeds are reported through profileComputeBlock's `samples`
  /// out-parameter and cost a full search each.
  int profileSeeds = 1;

  /// Graph profiling only: keep each class's profile to its lower envelope.
  /// A menu point whose pinned search measured no better than a smaller
  /// point is dropped, leaving a hole like an infeasible menu value:
  /// allocating R devices can always run the best configuration found at
  /// any R' < R and idle the rest, so such a point offers nothing the
  /// smaller one does not, and an allocation that picked it would pin
  /// devices for a program that runs on fewer. What remains is strictly
  /// decreasing in the resource, which is what the allocator's greedy
  /// optimality argument reads it as -- the achievable envelope, whether the
  /// measured cliff was a stalled search seed or a real scaling limit.
  bool profileRepair = true;

  /// Graph profiling only: when > 0, a class whose *best* profile point
  /// spends at least this share of its per-inference cost on transfers
  /// (ProfilePoint::transferShare) is kept on the host instead of entering
  /// the allocation -- transfer-bound work gains little from the device and
  /// occupies budget the compute-bound classes could use. Heuristic, not a
  /// comparison: there is no host cost model yet, so 0 (off, the default)
  /// only surfaces the shares in the allocation report and leaves the
  /// decision to the reader.
  double hostTransferBoundShare = 0;

  /// Which algorithm drives the post-init search phase. Every strategy shares
  /// the init sample (Phase 1) and the evaluation/bookkeeping machinery
  /// (CandidatePool); the strategy is only the policy that decides what to
  /// evaluate next each round.
  enum class SearchStrategyKind {
    /// BANANAS-style BO: MLP-ensemble surrogate ranked by an acquisition
    /// function over a neighbour+random candidate set.
    Bananas,
    /// Uniform random search over the unvisited configs: the floor every
    /// learned strategy has to beat at equal budget.
    Random,
    /// Random-restart neighbourhood descent: batched steepest descent over
    /// grid neighbours, restarting from a random config at local optima.
    Descent,
    /// Steady-state GA: tournament selection, per-parameter crossover,
    /// neighbour-step mutation, population seeded from the init sample.
    Ga,
  };
  SearchStrategyKind searchStrategy = SearchStrategyKind::Bananas;

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
  /// When > 0, each BANANAS round draws this many random candidates *in
  /// addition to* the neighbour set. When 0, random draws only top the
  /// candidate set up to nCandidates -- which, on spaces whose neighbour set
  /// alone exceeds nCandidates, means no random candidates at all: the search
  /// degenerates to a local hill climb from the init sample.
  size_t nRandCandidates = 16384;
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

  /// Commit by stamping configurations instead of splicing lowered code: the
  /// winning configuration is written onto the original block as attributes
  /// (see InferencePlugin::stampBestCandidate) and the block's body stays in
  /// the form the search read it in. A separate finalization pipeline then
  /// lowers the whole module in one go, bufferizing across block boundaries.
  /// Requires the module to already be in the plugin's converted (linalg)
  /// form: the space is built directly on the original blocks, so the pass
  /// runs the conversion once, up front, instead of once per reference.
  bool stampConfigs = false;

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

  /// When > 0, evaluate a random sample of this many configurations (drawn as
  /// samplingMode says, see CandidatePool::sampleInitialSet) instead of the
  /// whole space or running Bayesian optimisation. A cheap alternative to
  /// exhaustiveSearch when only a small ground-truth sample is needed --
  /// exhaustive search's cost is entirely its one simulator call per
  /// configuration, so sampling down to sampleN evaluations makes this
  /// proportionally faster. Takes priority over exhaustiveSearch if both are
  /// set. Dumped the same way (dumpFullPool controls whether pool.csv includes
  /// unvisited configs too). See also sampleMaxCostMs.
  unsigned sampleN = 0;

  /// How the sampleN draw picks the configurations it evaluates.
  enum class SamplingMode {
    /// Latin Hypercube Sampling over the normalised parameter encoding, each
    /// target snapped to its nearest unused config. Spreads the draw evenly
    /// over the space, which is what makes it a good design to fit a model on
    /// -- and exactly what makes it the wrong draw to compute a statistic
    /// from: the picks are stratified, so they are not independent and no
    /// config's inclusion probability is the plain 1/|space|.
    LHS,
    /// Independent uniform draws over the configs not yet picked. The only
    /// mode under which the sample is an unbiased estimator of the space, so
    /// the one to use when the sample feeds a percentile, a rank correlation
    /// or a binomial confidence bound rather than a surrogate fit.
    Uniform,
  };
  /// Only the sampleN draw reads this. BO's Phase-1 initial design and the
  /// nValidation held-out set always use LHS: they want the space-filling
  /// property, and nothing downstream of them is a population statistic.
  SamplingMode samplingMode = SamplingMode::LHS;

  /// When sampleN > 0, a candidate predicted to cost more than this many ms
  /// is rejected (not counted towards sampleN, and never dumped) and
  /// resampled past -- without this, a uniform-random sample over the valid
  /// space routinely includes configs whose predicted (and, worse, actual
  /// on-hardware) cost is orders of magnitude above the rest of the sample,
  /// which is wasteful once every sampled config gets compiled and run on
  /// real hardware downstream.
  ///
  /// Note that this conditions the sample on the accepted region, so a draw
  /// that has to stay a uniform sample of the whole space needs it at 0.
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

  /// Graph profiling only: drop a menu value whose device roofline
  /// (InferencePlugin::deviceRoofline) is no better than the host's for
  /// the same block, before profiling it. The search then only sweeps
  /// resource values that could pay, and a block whose every value is
  /// dropped never enters a space at all -- which is the same decision the
  /// per-op offload gate makes, taken where the feasible resource values
  /// are known instead of at the whole array.
  ///
  /// `gateDryRun` reports what it decides, per candidate value, without
  /// running a search or changing a program.
  bool screenMenuAgainstHost = true;

  /// The share of its roofline the host is taken to achieve. Every host cost
  /// the graph allocation uses -- the menu screen, the host profile point and
  /// the screen after profiling -- is the roofline divided by it
  /// (cinm::hostSeconds).
  double hostAchievedFraction = 0.64;

  /// Graph allocation only, latency objective only: offer every class the
  /// option of staying on the host, as a profile point costed by the host
  /// roofline, and let the allocation choose it. Placement then falls out of
  /// the same solve that sizes the device sets -- a block stays on the host
  /// when the devices it would take are worth more to another block, which
  /// no screen in front of the solve can know. The screens remain: they
  /// decide which device sizes are worth profiling, not where a block runs.
  bool allowHostPlacement = true;

  /// Graph profiling only: the most menu values to profile per block, after
  /// the screen above. 0 leaves the menu as the plugin (and the screen) left
  /// it; a positive value thins what remains geometrically, which bounds the
  /// sweep when the screen keeps most of a large menu.
  int64_t maxMenuPoints = 16;

  /// Graph allocation only: a directory to write each graph's allocation
  /// report to, as `<dir>/<graph>.json` -- everything the allocation did,
  /// from the menu screen's verdicts to the groups it carved out. The report
  /// is written as `<dumpDir>/<graph>/allocation.json` anyway when there is a
  /// dumpDir; this is for wanting the report without the per-search dumps
  /// dumpDir also turns on.
  std::string allocationReportDir;

  /// Graph allocation only: a directory holding the allocation to commit for
  /// each graph, as `<dir>/<graph>.json` (the layout allocationReportDir
  /// writes), in place of profiling and solving. Per class, its groups:
  /// `members` (member indices), and either `"on_host": true` or the
  /// `resource` of the point the group runs, the `config` to run there (in
  /// evalSingleSolution currency) and optionally `"timeshared": true` and
  /// `cost_ms`. Every class of the graph must be listed, and every member
  /// placed exactly once.
  std::string allocationIn;

  /// Graph profiling only: stop after the screen, profiling nothing. The
  /// surviving menu of each block is the sweep that would have run, so the
  /// allocation report of such a run says what the screen decides and what
  /// it saves, and the program that comes out of it is all host.
  bool gateDryRun = false;

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
  /// Total cost of the best configuration this point offers, in ms. Within
  /// a profile these strictly decrease as the resource grows: a point that
  /// does no better than a smaller one is dropped
  /// (InferenceOptions::profileRepair).
  double costMs;
  /// The argmin configuration, keyed by space dimension name in the same
  /// currency as InferenceOptions::evalSingleSolution, so it can be replayed
  /// through a later search or evaluation without reinterpretation.
  llvm::StringMap<ParmValue> config;
  /// The argmin's residency summary (plugin-measured); consumed by the
  /// graph-level co-residency packing and timeshare pricing.
  ResidencyInfo residency;
  /// Share of the incumbent's per-inference cost spent moving data
  /// (Transfer + TransferBack over the non-excluded total, so amortized
  /// weight scatters do not count). A class whose best point is mostly
  /// transfer gains little from the device; see
  /// InferenceOptions::hostTransferBoundShare. Negative when not measured.
  double transferShare = -1;
  /// Whether this point is the block staying where it is: `costMs` is then
  /// what the host would take for it (cinm::hostSeconds), it holds
  /// no device and pins nothing. A profile carries at most one, first, so
  /// that the allocation starts from everything on the host and spends the
  /// device where it buys the most -- which is what makes placement part of
  /// the allocation rather than a screen in front of it. Latency only: the
  /// throughput objective takes the busiest device set's load, and host work
  /// loads no set (see allocateGraph).
  bool onHost = false;
};

/// What the menu screen made of one resource value: the device roofline it
/// was priced at, and whether that beat the host (see
/// InferencePlugin::deviceRoofline).
struct MenuVerdict {
  int64_t resource = 0;
  double deviceMs = 0.0;
  /// The two terms deviceMs is the greater of, kept so that a dump of the
  /// screen describes the roof and not only its height, plus the traffic the
  /// same block would pay with nothing resident (DeviceRoofline).
  double transferMs = 0.0;
  double deviceOpsPerSecond = 0.0;
  double transferMsIfNothingResident = 0.0;
  bool kept = false;
};

/// The screen's reading of a whole menu. `hostMs` is 0 when the screen could
/// not run -- no device model, or a block the footprint reader cannot
/// measure -- and the menu is then left alone, since a screen that cannot
/// see is not evidence of unprofitability.
struct MenuScreen {
  double hostMs = 0.0;
  SmallVector<MenuVerdict> verdicts;
};

/// Price every value of `menu` against the host cost of `block`
/// (cinm::hostSeconds at `hostAchievedFraction`) and erase, in place, the
/// ones whose device roofline cannot beat it.
MenuScreen screenMenu(cinm::ComputeBlockOp block, InferencePlugin &plugin,
                      SmallVectorImpl<int64_t> &menu,
                      double hostAchievedFraction = 1.0);

/// Keep at most `maxPoints` values of `menu`, geometrically spaced, both
/// endpoints included. A no-op when `maxPoints` is 0 or the menu is already
/// short enough. Sorted divisor menus are distributed roughly
/// geometrically, so index spacing approximates log spacing.
void thinMenu(SmallVectorImpl<int64_t> &menu, int64_t maxPoints);

/// Caps how many profiling searches run at once across all the sweeps sharing
/// it. Several classes of one graph are profiled concurrently, and their costs
/// are wildly uneven -- on a transformer one class is half the work -- so
/// dividing the machine between them up front starves whichever class is the
/// critical path. Letting every sweep offer all its points and gating the
/// total instead means the threads go where the work still is: when the cheap
/// classes are done, the expensive one has the machine to itself.
class ProfileGate {
public:
  explicit ProfileGate(unsigned permits)
      : available(permits), total(std::max(1u, permits)) {}

  /// Permits held when idle -- what a sweep sizes its fan-out by.
  unsigned capacity() const { return total; }

  void acquire() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return available > 0; });
    --available;
  }
  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++available;
    }
    condition.notify_one();
  }

private:
  std::mutex mutex;
  std::condition_variable condition;
  unsigned available;
  unsigned total;
};

/// One search's outcome at one menu point. Several of these share a resource
/// when `InferenceOptions::profileSeeds` asks for repeats; the spread between
/// them is what separates a profile's real shape from its search noise.
struct SearchOutcome {
  /// Index of the repeat, 0 being the seed whose result the profile keeps.
  unsigned repeat = 0;
  /// The rngSeed the search ran with.
  int rngSeed = 0;
  /// The best cost it found; none when the pinned space had no feasible
  /// configuration, and `failure` then says why.
  std::optional<double> costMs;
  std::string failure;
};

/// What became of one value of a block's resource menu, from the screen to
/// the profile.
struct MenuPointTrace {
  int64_t resource = 0;
  /// The menu screen's reading of it; none when the screen did not run
  /// (InferenceOptions::screenMenuAgainstHost off, or a block it cannot
  /// price).
  std::optional<MenuVerdict> verdict;
  /// Whether a search ran at it: kept by the screen and by thinMenu.
  bool profiled = false;
  /// One per repeat, in repeat order; empty when it was not profiled.
  SmallVector<SearchOutcome> searches;
  /// Repeat 0's result, when it found one.
  std::optional<ProfilePoint> point;
  /// The smaller resource value whose point was no worse, which drops this
  /// one from the profile (InferenceOptions::profileRepair); 0 when nothing
  /// dominates it.
  int64_t dominatedBy = 0;
  /// Where repeat 0's search dumped its data (InferenceOptions::dumpDir);
  /// empty when it dumped nothing.
  std::string dumpDir;
};

/// Everything profileComputeBlock did to one block, including the menu values
/// that never reached the profile and why.
struct ProfileTrace {
  /// The host time the menu screen compared against (cinm::hostSeconds); 0
  /// when the screen did not run.
  double screenHostMs = 0.0;
  /// Every value sharedResourceMenu offered, in menu order.
  std::vector<MenuPointTrace> menu;

  MenuPointTrace *find(int64_t resource) {
    for (MenuPointTrace &point : menu)
      if (point.resource == resource)
        return &point;
    return nullptr;
  }
};

/// The part of profiling that runs no search: `reference`'s menu
/// (sharedResourceMenu), the screen's verdict on every value of it, and which
/// values survive the screen and thinMenu to be profiled. `computeOp` is the
/// original block, which the screen prices. The returned trace is what
/// profileComputeBlock starts from, so a caller that stops here
/// (InferenceOptions::gateDryRun) sees the same menu a run would have swept.
ProfileTrace planProfile(cinm::ComputeBlockOp computeOp,
                         cinm::ComputeBlockOp reference,
                         InferencePlugin &plugin, const InferenceOptions &opts);

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
///
/// `trace`, when given, records what became of every menu value, including
/// every search this ran with the `opts.profileSeeds` repeats per menu point.
/// The returned profile is unaffected by the repeats -- it is always seed
/// 0's -- so they are a measurement of the search, not an input to anything.
/// It is filled even when profiling fails silenceably, so that a caller can
/// say why.
///
/// `gate`, when given, is a concurrency budget shared with the other sweeps
/// running alongside this one: the sweep then offers every point it has and
/// each search runs single-threaded, so the permits land wherever work
/// remains instead of being divided up front. Without it the sweep sizes its
/// own fan-out out of `opts.numWorkers` and owns the machine.
///
/// `reference`, when given, is `computeOp`'s prepared reference module
/// (prepareReferenceModule): the menu is read off it, and every search
/// clones its trials from it instead of preparing a reference of its own.
/// Without it the sweep prepares one itself.
utils::Maybe<SmallVector<ProfilePoint>>
profileComputeBlock(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                    const InferenceOptions &opts, ProfileTrace *trace = nullptr,
                    ProfileGate *gate = nullptr,
                    const ReferenceModule *reference = nullptr);

/// Entry point for Bayesian inference.
DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts = {});

} // namespace mlir::cinm

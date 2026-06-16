# Main TODOs

## Cost model (UPMEM)

The current cost model in `UpmemSimulator.cpp` uses rough estimates. It needs to be replaced with real data and a proper simulator.

- [ ] scatter/gather cost: keep the existing C++ logic for `upmem.scatter`/`upmem.gather` but replace the current latency estimates with numbers derived from real hardware measurements.
- [x] DPU kernel estimator (`upmem.wait_for`): integrate Hamid's cycle-accurate DPU simulator. Options in order of preference:
  - [x] bring it in as an external C++ or Rust library linked into cinm-mlir
  - ~~bind via its Python interface if that is easier to maintain~~
- [ ] Remove `CostModel.cpp` and its associated pass — this is superseded by the Bayesian approach
- [ ] Accuracy benchmarks: once the new model is in place, benchmark predicted vs. measured latency on real UPMEM hardware 

## Bayesian optimization

- [ ] Extend the BO objective beyond latency. Candidates:
  - [ ] Compute intensity: `compute_time / (compute_time + transfer_time)`
  - [ ] Energy efficiency: maybe approximate this with number of DPUs running? or is compute intensity anyway a better target?

- [x] Maybe add an exhaustive search option to compare bayesian solutions against the true best solutions for experiments
- [ ] Wire the InferencePlugin into the main CinmPlatformAttrInterface
  - This needs care to avoid adding pulling all the dependencies of Bayesian into CinmIR - maybe we need a separate interface that we publish in the core AcceleratorInference lib but implement in eg UpmemTransforms

- [x] Add some evaluation of the surrogate and make sure its learning parameters are adapted for the task
  - Add an evaluation metric, like recall for the best kernels

- [x] Bayesian is working ok but doesn't outperform random search significantly on small spaces. We need to make the space larger. 

## CINM 

- [ ] Better tiling for dynamic dimensions: for now we use tiling factors that assume the dynamic dimensions are divisible by the tiling factors. This needs support in CNM at least, see below

## CNM pipeline

- [ ] Support padding of the kernel, to support imperfect tiling efficiently. Taking padding into account transforms the Bayesion design space significantly as it has fewer constraints, so more potentially valid points. But
  - Maybe a padded solution gives us better compute intensity for an insignificant overhead (which may be absorbed by the transfers anyway)
  - Padding allows us to tile dynamic dimensions and static dimensions that don't have many divisors
- [ ] Experiment with MRAM tiling

## CIM pipeline

- The current CIM lowering relies entirely on memrefs - is this the right call?
- Fix any regressions introduced by refactorings that broke the CIM path.

## Cleanups

- Do we still need the LinalgCleanup pass? I improved the bufferization pipeline for CINM->CNM a lot and do not need it at least for the benchmarks in ./testbench
- 

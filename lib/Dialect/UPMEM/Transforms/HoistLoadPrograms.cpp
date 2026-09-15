/// Hoist upmem.load_program ops next to their set's allocation when the set
/// only ever holds one program.
///
/// @file

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMHOISTLOADPROGRAMSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

} // namespace mlir::upmem

using namespace mlir;

namespace {

struct UpmemHoistLoadProgramsPass
    : upmem::impl::UpmemHoistLoadProgramsPassBase<UpmemHoistLoadProgramsPass> {
  void runOnOperation() override {
    // Group the loads by the set they target. MapVector: the iteration
    // order below must not depend on pointer values.
    llvm::MapVector<Value, llvm::SmallVector<upmem::LoadProgramOp>> loadsOn;
    getOperation()->walk([&](upmem::LoadProgramOp load) {
      loadsOn[load.getHierarchy()].push_back(load);
    });

    for (auto &[set, loads] : loadsOn) {
      Operation *alloc = set.getDefiningOp();
      if (!alloc)
        continue; // Block argument: the allocation is not visible here.

      SymbolRefAttr program = loads.front().getDpuProgramRefAttr();
      if (!llvm::all_of(loads, [&](upmem::LoadProgramOp load) {
            return load.getDpuProgramRefAttr() == program;
          }))
        continue; // Timeshared set: the load placement is the schedule.

      // One program for the set's whole lifetime: load it right where the
      // set is allocated and nowhere else. The set's uses are all dominated
      // by the allocation, so the single load dominates every launch, and
      // being the unique load on the value it is what program resolution
      // finds from anywhere (see programLoadedOn).
      loads.front()->moveAfter(alloc);
      for (upmem::LoadProgramOp redundant : llvm::drop_begin(loads))
        redundant.erase();
    }
  }
};

} // namespace

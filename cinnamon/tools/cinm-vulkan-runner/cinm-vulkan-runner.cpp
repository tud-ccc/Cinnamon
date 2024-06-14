#include "cinm-mlir/Conversion/CnmToGPU/CnmToGPU.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h"

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/ExecutionEngine/JitRunner.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

using namespace mlir;

static LogicalResult runMLIRPasses(Operation *op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module) {
    return op->emitOpError("expected a 'builtin.module' op");
  }

  PassManager passManager(module.getContext());
  if (failed(applyPassManagerCLOptions(passManager))) {
    return failure();
  }

  passManager.addPass(cnm::createConvertCnmToGPUPass());

  passManager.addPass(createGpuKernelOutliningPass());

  passManager.addPass(createLowerAffinePass()); // affine.load -> memref.load
  passManager.addPass(
      memref::createFoldMemRefAliasOpsPass());  // memref.load -> affine.apply +
                                                // memref.load
  passManager.addPass(createLowerAffinePass()); // affine.apply -> arith ops

  passManager.addPass(createCnmSPIRVAttachTargetAttributePass(
      CnmSPIRVAttachTargetAttributePassOptions{
          .spirvCapabilities = {"Shader"},
          .spirvExtensions = {"SPV_KHR_storage_buffer_storage_class"},
      }));

  OpPassManager &gpuModulePM = passManager.nest<gpu::GPUModuleOp>();
  gpuModulePM.addPass(createConvertMemRefToSPIRVPass());
  gpuModulePM.addPass(createConvertControlFlowToSPIRVPass());
  gpuModulePM.addPass(createCnmSPIRVAttachKernelEntryPointAttributePass());

  passManager.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

  OpPassManager &spirvModulePM = passManager.nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spirvModulePM.addPass(spirv::createSPIRVUpdateVCEPass());

  passManager.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.addPass(createConvertVectorToLLVMPass());

  OpPassManager &functionPM = passManager.nest<func::FuncOp>();
  functionPM.addPass(LLVM::createRequestCWrappersPass());

  ConvertFuncToLLVMPassOptions funcToLLVMOptions{};
  funcToLLVMOptions.indexBitwidth =
      DataLayout(module).getTypeSizeInBits(IndexType::get(module.getContext()));
  passManager.addPass(createConvertFuncToLLVMPass(funcToLLVMOptions));
  // passManager.addPass(createReconcileUnrealizedCastsPass());
  passManager.addPass(createConvertVulkanLaunchFuncToVulkanCallsPass());

  return passManager.run(module);
}

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = [](Operation *op, JitRunnerOptions &) {
    return runMLIRPasses(op);
  };

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<cnm::CnmDialect>();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}

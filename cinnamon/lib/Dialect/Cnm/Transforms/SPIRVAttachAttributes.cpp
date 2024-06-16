#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h"

#include <llvm/Support/Regex.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVAttributes.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Target/SPIRV/Target.h>

namespace mlir::cnm {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_CNMSPIRVATTACHKERNELENTRYPOINTATTRIBUTEPASS
#define GEN_PASS_DEF_CNMSPIRVATTACHTARGETATTRIBUTEPASS
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct CnmSPIRVAttachKernelEntryPointAttributePass : public impl::CnmSPIRVAttachKernelEntryPointAttributePassBase<CnmSPIRVAttachKernelEntryPointAttributePass> {
    using Base::Base;

    void runOnOperation() final;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<spirv::SPIRVDialect>();
    }
};

struct CnmSPIRVAttachTargetAttributePass : public impl::CnmSPIRVAttachTargetAttributePassBase<CnmSPIRVAttachTargetAttributePass> {
    using Base::Base;

    void runOnOperation() final;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<spirv::SPIRVDialect>();
    }
};

void CnmSPIRVAttachKernelEntryPointAttributePass::runOnOperation() {
    const llvm::Regex matcher(kernelMatcher);
    getOperation()->walk([&](gpu::GPUFuncOp gpuFunc) {
        if (!kernelMatcher.empty() && !matcher.match(gpuFunc.getName())) {
            return;
        }

        // todo: calculate based on block/grid size
        const DenseI32ArrayAttr workgroup_size = DenseI32ArrayAttr::get(&getContext(), {1, 1, 1});
        const std::optional<int> subgroup_size;

        gpuFunc->setAttr("spirv.entry_point_abi", spirv::EntryPointABIAttr::get(&getContext(), workgroup_size, subgroup_size));
    });
}

void CnmSPIRVAttachTargetAttributePass::runOnOperation() {
    const auto versionSymbol = spirv::symbolizeVersion(spirvVersion);
    if (!versionSymbol) {
        return signalPassFailure();
    }

    const auto apiSymbol = spirv::symbolizeClientAPI(clientApi);
    if (!apiSymbol) {
        return signalPassFailure();
    }

    const auto vendorSymbol = spirv::symbolizeVendor(deviceVendor);
    if (!vendorSymbol) {
        return signalPassFailure();
    }

    const auto deviceTypeSymbol = spirv::symbolizeDeviceType(deviceType);
    if (!deviceTypeSymbol) {
        return signalPassFailure();
    }

    // Set the default device ID if none was given
    if (!deviceId.hasValue()) {
        deviceId = mlir::spirv::TargetEnvAttr::kUnknownDeviceID;
    }

    const spirv::Version version = versionSymbol.value();

    SmallVector<spirv::Capability, 4> capabilities;
    for (const auto &cap : spirvCapabilities) {
        if (const auto capSymbol = spirv::symbolizeCapability(cap)) {
            capabilities.push_back(capSymbol.value());
        }
    }

    SmallVector<spirv::Extension, 8> extensions;
    for (const auto &ext : spirvExtensions) {
        if (const auto extSymbol = spirv::symbolizeExtension(ext)) {
            extensions.push_back(extSymbol.value());
        }
    }

    const spirv::VerCapExtAttr vce = spirv::VerCapExtAttr::get(version, capabilities, extensions, &getContext());
    const auto target = spirv::TargetEnvAttr::get(
        vce, spirv::getDefaultResourceLimits(&getContext()),
        apiSymbol.value(), vendorSymbol.value(),
        deviceTypeSymbol.value(), deviceId
    );

    const llvm::Regex matcher(moduleMatcher);
    getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
        if (moduleMatcher.empty() || matcher.match(gpuModule.getName())) {
            gpuModule->setAttr("spirv.target_env", target);
        }
    });
}

} // namespace mlir

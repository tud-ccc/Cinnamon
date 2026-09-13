from ._utility.compiler_invoker import CompilerInvoker
from .model_invoker import ModelInvoker
from .compiled_model import CompiledModel
from ._utility.common_pipelines import CommonPipelines
from ._utility.resource_paths import ResourcePaths
from ._utility.signature_extractor import SignatureExtractor


class CinmBackend:
    CINM_PIPELINE = [
        # Convert supported torch ops to cinm dialect
        "func.func(convert-torch-to-cinm)",
        "func.func(convert-cinm-to-cim)",
        "func.func(cim-schedule-asap)",
        "func.func(convert-cim-to-memristor)",
        "func.func(convert-memristor-to-func)",
        "convert-func-to-llvm",
    ]
    LINALG_PIPELINE = (
        CommonPipelines.LOWER_TORCH_TO_LINALG_UNCHECKED_PIPELINE
        + CommonPipelines.LOWER_LINALG_TO_LLVM_PIPELINE
    )

    def compile(
        self, model, input_tensor, *, dump_dir: str = None, step_by_step: bool = False
    ) -> CompiledModel:
        invoker = CompilerInvoker(dump_dir, step_by_step)

        mlir = invoker.compile_torch_module(model, input_tensor)
        mlir = invoker.cinm_opt(CinmBackend.CINM_PIPELINE, mlir)
        llvm_mlir = invoker.cinm_opt(CinmBackend.LINALG_PIPELINE, mlir)
        ll_ir = invoker.mlir_translate(llvm_mlir)
        shared_object = invoker.clang(ll_ir)

        signatures = SignatureExtractor.extract(mlir)

        return CompiledModel(signatures, shared_object)

    def load(self, model: CompiledModel) -> ModelInvoker:
        runtimes = [ResourcePaths.memristor_runtime()]
        return ModelInvoker(model, runtimes)

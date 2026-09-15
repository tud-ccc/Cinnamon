import os
import subprocess
import torch
import tempfile

from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

from .resource_paths import ResourcePaths
from ..exceptions import BackendException


class CompilerInvoker:
    def __init__(self, dump_dir: str = None, step_by_step: bool = False):
        self._dump_dir = dump_dir
        self._step_by_step = step_by_step
        self._step_counter = 0

        if self._dump_dir:
            for f in os.listdir(self._dump_dir):
                os.remove(os.path.join(self._dump_dir, f))

    def _dump(self, filename: str, data: bytes):
        if self._dump_dir:
            os.makedirs(self._dump_dir, exist_ok=True)

            filename = f"{self._step_counter:02d}-{filename}"
            self._step_counter += 1
            with open(os.path.join(self._dump_dir, filename), "wb") as f:
                f.write(data)

    def _invoke(self, *args, **kwargs) -> bytes:
        try:
            return subprocess.check_output(*args, **kwargs)
        except subprocess.CalledProcessError as e:
            raise BackendException(f"Compiler invocation failed: {args}") from e

    def _pass_pipeline(self, pipeline: list[str]) -> str:
        return f"--pass-pipeline=builtin.module({','.join(pipeline)})"

    def _invoke_opt(self, binary: str, pipeline: list[str], input_mlir: bytes) -> bytes:
        opt_name = os.path.basename(binary)

        if self._step_by_step:
            for i, step in enumerate(pipeline):
                output_mlir = self._invoke(
                    [binary, self._pass_pipeline([pipeline[i]])],
                    input=input_mlir,
                )

                self._dump(
                    f"model.lowered.{opt_name}.{i}-{step.strip('-')}.mlir", output_mlir
                )

                input_mlir = output_mlir
        else:
            output_mlir = self._invoke(
                [binary, self._pass_pipeline(pipeline)],
                input=input_mlir,
            )

        self._dump(f"model.lowered.{opt_name}.mlir", output_mlir)

        return output_mlir

    def compile_torch_module(self, module: torch.nn.Module, inp: torch.Tensor) -> bytes:
        compiled_module = fx.export_and_import(
            module, inp, output_type=OutputType.TORCH
        )
        module_mlir = str(compiled_module).encode("utf-8")

        self._dump("model.mlir", module_mlir)

        return module_mlir

    def torch_mlir_opt(self, pipeline: list[str], input_mlir: bytes) -> bytes:
        return self._invoke_opt(ResourcePaths.torch_mlir_opt(), pipeline, input_mlir)

    def cinm_opt(self, pipeline: list[str], input_mlir: bytes) -> bytes:
        return self._invoke_opt(ResourcePaths.cinm_opt(), pipeline, input_mlir)

    def mlir_translate(self, mlir: bytes) -> bytes:
        llvm_mlir = self._invoke(
            [ResourcePaths.mlir_translate(), "--mlir-to-llvmir"],
            input=mlir,
        )

        self._dump("model.ll", llvm_mlir)

        return llvm_mlir

    def llc(self, llvm_ir: bytes) -> bytes:
        # llc pairs with mlir-translate from the same LLVM: the IR they exchange
        # is that LLVM's, and no other version is guaranteed to read it.
        obj = self._invoke(
            [ResourcePaths.llc(), "-filetype=obj", "-relocation-model=pic", "-o", "-"],
            input=llvm_ir,
        )
        self._dump("model.o", obj)
        return obj

    def link(self, obj: bytes) -> bytes:
        # The linker's driver rather than ld/mold directly, so that libc and its
        # crt bits come from whatever sysroot this clang was built for.
        with (
            tempfile.NamedTemporaryFile(
                prefix="cinnamon_compiled_model.", suffix=".o"
            ) as obj_file,
            tempfile.NamedTemporaryFile(
                prefix="cinnamon_compiled_model.", suffix=".so"
            ) as so_file,
        ):
            obj_file.write(obj)
            obj_file.flush()
            self._invoke(
                [ResourcePaths.clang(), "-shared", "-o", so_file.name, obj_file.name]
            )
            so_file.seek(0)
            shared_object = so_file.read()

        self._dump("model.so", shared_object)

        return shared_object

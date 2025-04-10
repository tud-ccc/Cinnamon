import ctypes
import tempfile

from .compiled_model import CompiledModel, FunctionSignature
from ._utility.ciface_type_wrappers import get_ciface_wrapper


class ModelInvoker:

    def __init__(self, compiled_model: CompiledModel, runtimes: list[str] = None):
        self._shared_object = tempfile.NamedTemporaryFile(
            prefix="cinnamon_compiled_model", suffix=".so"
        )

        with open(self._shared_object.name, "wb") as f:
            f.write(compiled_model.data())

        self._runtimes = {}

        if runtimes is None:
            runtimes = []

        for runtime in runtimes:
            self._runtimes[runtime] = ctypes.CDLL(runtime, mode=ctypes.RTLD_GLOBAL)

        self._ffi = ctypes.CDLL(f.name)

        for signature in compiled_model.signatures().values():
            setattr(
                self,
                signature.name,
                lambda *args, sig=signature: self._function_wrapper(
                    *args, signature=sig
                ),
            )

    def _function_wrapper(
        self,
        *args,
        signature: FunctionSignature,
    ):
        assert len(signature.argument_types) == len(args)

        result_wrapper = get_ciface_wrapper(signature.return_type)
        args_wrappers = [
            get_ciface_wrapper(ty, arg)
            for ty, arg in zip(signature.argument_types, args)
        ]

        result_ffi = result_wrapper.as_ffi_arg()
        args_ffi = [arg.as_ffi_arg() for arg in args_wrappers]

        all_args = [result_ffi] + args_ffi

        function = self._ffi[f"_mlir_ciface_{signature.name}"]
        function.restype = None
        function.argtypes = [type(arg) for arg in all_args]
        function(*all_args)

        result = result_wrapper.retrieve()
        return result

import ctypes
import numpy
import torch
import math

_MLIR_TO_TORCH_TYPE_MAPPING = {
    "f32": torch.float32,
    "f64": torch.float64,
    "si8": torch.int8,
    "si16": torch.int16,
    "si32": torch.int32,
    "si64": torch.int64,
}

_MLIR_TO_CTYPES_TYPE_MAPPING = {
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "si8": ctypes.c_int8,
    "si16": ctypes.c_int16,
    "si32": ctypes.c_int32,
    "si64": ctypes.c_int64,
}


def _torch_type_from_mlir(mlir_type: str) -> torch.dtype:
    if mlir_type in _MLIR_TO_TORCH_TYPE_MAPPING:
        return _MLIR_TO_TORCH_TYPE_MAPPING[mlir_type]

    raise ValueError(f"Unsupported MLIR type: {mlir_type}")


def _ctypes_type_from_mlir(mlir_type: str) -> any:
    if mlir_type in _MLIR_TO_CTYPES_TYPE_MAPPING:
        return _MLIR_TO_CTYPES_TYPE_MAPPING[mlir_type]

    raise ValueError(f"Unsupported MLIR type: {mlir_type}")


class CIfaceWrapper:
    def __init__(self, typename: str, data: any = None):
        self._typename = typename
        self._data = data

    def as_ffi_arg(self) -> any:
        raise NotImplementedError

    def retrieve(self) -> any:
        raise NotImplementedError


class TensorCIfaceWrapper(CIfaceWrapper):
    def _shape(self):
        return tuple(map(int, self._typename.split("[")[1].split("]")[0].split(",")))

    def _dtype(self):
        return self._typename.split(",")[-1].split(">")[0]

    def _descriptor(self):
        dimensions = len(self._shape())

        class MemRefDescriptor(ctypes.Structure):
            _fields_ = [
                ("basePtr", ctypes.POINTER(ctypes.c_float)),
                ("dataPtr", ctypes.POINTER(ctypes.c_float)),
                ("offset", ctypes.c_int64),
                ("sizes", ctypes.c_int64 * dimensions),
                ("strides", ctypes.c_int64 * dimensions),
            ]

        return MemRefDescriptor()

    def as_ffi_arg(self) -> ctypes.c_void_p:
        dtype = _torch_type_from_mlir(self._dtype())

        if self._data is None:
            self._data = torch.zeros(self._shape(), dtype=dtype)

        data = self._data.contiguous()
        data_ptr = data.data_ptr()

        descriptor = self._descriptor()
        descriptor.basePtr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_float))
        descriptor.dataPtr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_float))
        descriptor.offset = 0
        for i in range(len(descriptor.sizes)):
            descriptor.sizes[i] = data.size(i)
            descriptor.strides[i] = data.stride(i)

        self._memref = descriptor

        return ctypes.cast(ctypes.pointer(descriptor), ctypes.c_void_p)

    def retrieve(self):
        assert self._memref

        data_ptr = ctypes.addressof(self._memref.dataPtr.contents)
        count = math.prod(self._shape())
        dtype = _ctypes_type_from_mlir(self._dtype())

        data_array = numpy.ctypeslib.as_array(
            (dtype * count).from_address(data_ptr)
        ).reshape(self._shape())

        return torch.from_numpy(data_array)


def get_ciface_wrapper(typename: str, data: any = None) -> CIfaceWrapper:
    if typename.startswith("!torch.vtensor"):
        return TensorCIfaceWrapper(typename, data)

    raise ValueError(f"Unsupported interface type: {typename}")

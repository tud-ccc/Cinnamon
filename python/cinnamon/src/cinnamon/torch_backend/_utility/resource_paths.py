import os


class ResourcePaths:
    def _resource_dir():
        file_location = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(file_location, "..", "..", "_resources")

    def torch_mlir_opt():
        resource_dir = ResourcePaths._resource_dir()
        return os.path.join(resource_dir, "torch-mlir-opt")

    def cinm_opt():
        resource_dir = ResourcePaths._resource_dir()
        return os.path.join(resource_dir, "cinm-opt")

    def mlir_translate():
        resource_dir = ResourcePaths._resource_dir()
        return os.path.join(resource_dir, "mlir-translate")

    def clang():
        resource_dir = ResourcePaths._resource_dir()
        return os.path.join(resource_dir, "clang")

    def memristor_runtime():
        resource_dir = ResourcePaths._resource_dir()
        return os.path.join(resource_dir, "libMemristorDialectRuntime.so")

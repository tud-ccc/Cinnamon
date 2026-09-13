from ._utility.signature_extractor import FunctionSignature
from zipfile import ZipFile as zipfile
import pickle


class CompiledModel:
    def __init__(self, signatures: dict[str, FunctionSignature], shared_object: bytes):
        self._shared_object = shared_object
        self._function_signatures = signatures

    def save_to_file(self, filename: str) -> None:
        with zipfile(filename, "w") as f:
            f.writestr("shared_object.so", self._shared_object)
            f.writestr("signatures.json", pickle.dumps(self._function_signatures))

    @staticmethod
    def load_from_file(filename: str) -> "CompiledModel":
        with zipfile(filename, "r") as f:
            shared_object = f.read("shared_object.so")
            signatures = pickle.loads(f.read("signatures.json"))

        return CompiledModel(signatures, shared_object)

    def data(self) -> bytes:
        return self._shared_object

    def signatures(self) -> dict[str, FunctionSignature]:
        return self._function_signatures

import dataclasses


@dataclasses.dataclass
class FunctionSignature:
    name: str
    argument_types: list[str]
    return_type: str


class SignatureExtractor:
    def extract(mlir: bytes) -> dict:
        signatures = {}

        # FIXME: This is a very naive way to extract function signatures
        for line in mlir.decode("utf-8").split("\n"):
            line = line.strip()
            if not line.startswith("func.func"):
                continue

            name = line.split("@")[1].split("(")[0].strip()

            arg_types = []
            args_string = line.split("(")[1].split(")")[0].strip()
            for arg in args_string.split("%"):
                if len(arg) == 0:
                    continue
                arg_type = arg.split(" ")[1].strip(", ")
                arg_types.append(arg_type)

            return_type = line.split("->")[1].split("{")[0].strip()

            signatures[name] = FunctionSignature(name, arg_types, return_type)
        return signatures

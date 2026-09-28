"""A bench driver for one compute block on its own.

The suites' drivers are written by hand, one per benchmark, because each
knows its operands' meaning and checks a golden result. A block class of an
end-to-end model is benched with neither: its reference module (the
allocation report's `classes[i].reference`) holds the class's one compute
block in a function of its own, and what is timed is that function on
operands of the right shapes. So the driver is generated from the
function's signature:

  - tensors are passed as bare pointers (the host lowering's
    use-bare-ptr-memref-call-conv), filled with small random values the way
    the suites fill theirs, but never zero: a class's kernel may divide by
    one of its operands (a softmax's sum, a norm's sigma), which in a model
    is never zero and on a DPU faults when it is;
  - integer scalars are passed by value, `index` ones as 0 -- in a class they
    are offsets into a stacked weight, a layer index -- and the others as 1,
    which is a valid shift amount and divisor;
  - tensor results become trailing out-parameters
    (buffer-results-to-out-params), zeroed; a scalar result is still
    returned, and kept in a member so the call is not dead;
  - there is no golden reference, so the driver refuses a checked run:
    bench it with BENCH_CHECK=0, as the end-to-end models are.

    write_driver(reference_mlir, function, out_cpp)
"""

from __future__ import annotations

import math
import pathlib
import re

from . import paths

_ELEMENT = {
    "i1": "bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "index": "int64_t",
    "f16": "uint16_t",
    "bf16": "uint16_t",
    "f32": "float",
    "f64": "double",
}


def _split_top(text: str) -> list[str]:
    """`text` split at the commas outside any <...>, (...), [...] or {...}."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def _parse_type(text: str) -> tuple[str, tuple[int, ...] | None]:
    """(element type, shape) of `tensor<AxBxT>` / `memref<...>`, or of a
    scalar type with shape None."""
    m = re.fullmatch(r"(?:tensor|memref)<(.*)>", text.strip())
    if not m:
        return text.strip(), None
    dims = m.group(1).split(",")[0].split("x")
    *shape, element = dims
    if any(not d.isdigit() for d in shape):
        raise ValueError(f"a dynamic shape in {text}")
    return element.strip(), tuple(int(d) for d in shape)


def signature(reference_mlir: pathlib.Path, function: str):
    """The (argument, result) types of `function` in `reference_mlir`, each a
    (element type, shape) with shape None for a scalar."""
    text = reference_mlir.read_text()
    m = re.search(
        rf"func\.func @{re.escape(function)}\((.*?)\)\s*(->\s*(.*?))?\s*(attributes|\{{)",
        text,
        re.S,
    )
    if not m:
        raise ValueError(f"no function @{function} in {reference_mlir}")
    args = []
    for arg in _split_top(m.group(1)):
        # "%arg0: tensor<...> {cinm.static}"
        typ = arg.split(":", 1)[1].strip()
        typ = re.sub(r"\s*\{.*\}\s*$", "", typ)
        args.append(_parse_type(typ))
    results_text = (m.group(3) or "").strip()
    if results_text.startswith("(") and results_text.endswith(")"):
        results_text = results_text[1:-1]
    results = [_parse_type(t) for t in _split_top(results_text)] if results_text else []
    return args, results


def _c_type(element: str) -> str:
    if element not in _ELEMENT:
        raise ValueError(f"no C type for element type {element}")
    return _ELEMENT[element]


def render(reference_mlir: pathlib.Path, function: str) -> str:
    args, results = signature(reference_mlir, function)
    params, members, fills, passes = [], [], [], []
    call = ""
    for i, (element, shape) in enumerate(args):
        ctype = _c_type(element)
        name = f"a{i}"
        if shape is None:
            params.append(ctype)
            value = "0" if element == "index" else "1"
            members.append(f"  {ctype} {name} = {value};")
            passes.append(name)
        else:
            params.append(f"{ctype} *")
            members.append(f"  std::vector<{ctype}> {name};")
            fills.append(f"    {name} = rnd<{ctype}>({math.prod(shape) or 1});")
            passes.append(f"{name}.data()")
    returned = "void"
    for i, (element, shape) in enumerate(results):
        ctype = _c_type(element)
        name = f"r{i}"
        if shape is None:
            if returned != "void":
                raise ValueError(
                    f"@{function} returns several scalars, which the host"
                    " lowering returns as a struct"
                )
            returned = ctype
            members.append(f"  {ctype} {name} = {ctype}();")
            call = f"{name} = "
            continue
        params.append(f"{ctype} *")
        members.append(f"  std::vector<{ctype}> {name};")
        fills.append(f"    {name}.assign({math.prod(shape) or 1}, {ctype}());")
        passes.append(f"{name}.data()")

    common = paths.benchmarks_dir() / "common.hpp"
    return f"""// GENERATED by cinm_experiments.kernel_driver from
// {reference_mlir}
// -- one compute block on its own, timed on random operands. No golden
// reference: bench with BENCH_CHECK=0.

#include "{common}"

#include <cstdint>

namespace {{

// A xorshift fill, as the model drivers use: some of these operands are a
// model's whole weight stack. Never zero, since a kernel may divide by one.
template <class T> std::vector<T> rnd(size_t n) {{
  return bench::interleaved_pages([n] {{
    std::vector<T> v(n);
    uint32_t s = 0x9e3779b9u;
    for (auto &x : v) {{
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      x = (T)(1 + s % (bench::kOperandRange - 1));
    }}
    return v;
  }});
}}

}} // namespace

extern "C" {returned} BENCH_FN({", ".join(params)});

struct Kernel {{
  static constexpr bench::Size kSizes[] = {{
      {{TOSTR(BENCH_FN), {{1, 0, 0}}}},
  }};

{chr(10).join(members)}
  std::vector<DTY> out;

  void setup(const size_t *) {{
{chr(10).join(fills)}
    printf("%s  (one block, {len(args)} operand(s), {len(results)} result(s))",
           TOSTR(BENCH_FN));
  }}

  void run() {{ {call}BENCH_FN({", ".join(passes)}); }}

  const std::vector<DTY> &output() const {{ return out; }}

  std::vector<double> reference() const {{
    fprintf(stderr,
            "%s: a block benched on its own has no golden reference; run "
            "with BENCH_CHECK=0\\n",
            TOSTR(BENCH_FN));
    exit(1);
  }}
}};

int main(int argc, char **argv) {{ return bench::run<Kernel>(argc, argv); }}
"""


def write_driver(
    reference_mlir: pathlib.Path, function: str, out_cpp: pathlib.Path
) -> pathlib.Path:
    """Write the driver of `function` to `out_cpp`, unless it already holds
    exactly that (so that its mtime only moves when the driver does)."""
    text = render(pathlib.Path(reference_mlir), function)
    out_cpp = pathlib.Path(out_cpp)
    if not out_cpp.exists() or out_cpp.read_text() != text:
        out_cpp.parent.mkdir(parents=True, exist_ok=True)
        out_cpp.write_text(text)
    return out_cpp

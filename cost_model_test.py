import random

from xdsl.context import Context
from xdsl.parser import Parser

name = "cost_model_test"
operations = {
  "cinm.compute"
}
dse_parameters = {
  "unroll-factor": (1, 16, True)
}
dse_max_iterations = 10

ctx = Context(allow_unregistered=True)

def get_passes_for_next_run(dse_parameters: dict[str, float]):
  return f"affine-loop-unroll{{unroll-factor={int(dse_parameters["unroll-factor"])} }}"

def run(ir: str, location: str) -> float :
  parser = Parser(ctx, ir)
  compute_op = parser.parse_operation()

  for op in compute_op.walk():
    op.name = op.get_attr_or_prop("op_name__").data

  cinm_ops = [op for op in compute_op.walk() if op.name.startswith("cinm.")]
  return abs(len(cinm_ops) - 10)

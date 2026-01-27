import random

from xdsl.context import Context
from xdsl.parser import Parser

name = "cost_model_test"
passes = "cinm-tiling"
operations = {
    "cinm.compute"
}

ctx = Context(allow_unregistered=True)

# def run(op: str, elementType: str, operand_dimensions: list[list[int]], location: str) -> float :
#   print(op, elementType, operand_dimensions, location)
#   return random.uniform(0.0, 10.0)

def run(ir: str, location: str) -> float :
  parser = Parser(ctx, ir)
  compute_op = parser.parse_operation()
  print(ir)

  print(parser.forward_ssa_references)

  for op in compute_op.walk():
    op.name = op.get_attr_or_prop("op_name__").data

  cinm_ops = [op for op in compute_op.walk()]
  return len(cinm_ops) + random.uniform(0.0, 10.0)

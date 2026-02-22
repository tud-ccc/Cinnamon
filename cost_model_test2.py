import random

from xdsl.context import Context
from xdsl.parser import Parser

name = "cost_model_test_2"
operations = {
    "cinm.compute"
}

ctx = Context(allow_unregistered=True)

def get_passes_for_next_run():
  yield ""

def run(ir: str, location: str) -> float :
  parser = Parser(ctx, ir)
  compute_op = parser.parse_operation()

  for op in compute_op.walk():
    op.name = op.get_attr_or_prop("op_name__").data

  cinm_ops = [op for op in compute_op.walk()]
  return len(cinm_ops)

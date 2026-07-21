import torch
import sys

sys.path.append('./llama2.c')

from model import ModelArgs, Transformer  # from llama2.c

# Match these to the model's README / config
args = ModelArgs(
    dim=288,
    n_layers=6,
    n_heads=6,
    vocab_size=32000,
    max_seq_len=256,
)

model = Transformer(args)

# Load the .pt weights
checkpoint = torch.load("stories15M.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()

# Trace and export to ONNX
tokens = torch.zeros((1, 10), dtype=torch.long)  # dummy input

torch.onnx.export(
    model,
    tokens,
    "llama2.onnx",
    input_names=["tokens"],
    output_names=["logits"],
    opset_version=14,
)

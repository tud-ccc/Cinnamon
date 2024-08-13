import os
import tempfile
import random
from unittest import TestCase
import torch
import numpy as np
from multi_head_attention import MultiHeadAttention

from torch_mlir import torchscript
from torch_mlir import fx
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

class TestMultiHeadAttention(TestCase):

    def test_divisible(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(in_features=73, head_num=5)

    @staticmethod
    def get_torch_layer_with_weights(feature_dim, head_num, weights, bias):
        layer = MultiHeadAttention(feature_dim, head_num, weights, bias)
        layer.linear_q.weight = torch.nn.Parameter(
            torch.from_numpy(weights[:, :feature_dim]).transpose(1, 0)
        )
        layer.linear_q.bias = torch.nn.Parameter(
            torch.from_numpy(bias[:feature_dim])
        )
        layer.linear_k.weight = torch.nn.Parameter(
            torch.from_numpy(weights[:, feature_dim:feature_dim * 2]).transpose(1, 0)
        )
        layer.linear_k.bias = torch.nn.Parameter(
            torch.from_numpy(bias[feature_dim:feature_dim * 2])
        )
        layer.linear_v.weight = torch.nn.Parameter(
            torch.from_numpy(weights[:, feature_dim * 2:feature_dim * 3]).transpose(1, 0)
        )
        layer.linear_v.bias = torch.nn.Parameter(
            torch.from_numpy(bias[feature_dim * 2:feature_dim * 3])
        )
        layer.linear_o.weight = torch.nn.Parameter(
            torch.from_numpy(weights[:, -feature_dim:]).transpose(1, 0)
        )
        layer.linear_o.bias = torch.nn.Parameter(
            torch.from_numpy(bias[-feature_dim:])
        )
        return layer


    def test_same_output_without_mask(self):
        batch_size, seq_len, feature_dim, head_num = 7, 12, 16, 4
        # batch_size, seq_len, feature_dim, head_num = 1, 4, 8, 2
        weights = np.random.standard_normal((feature_dim, feature_dim * 4))
        bias = np.random.standard_normal((feature_dim * 4,))
        torch_net = MultiHeadAttention(feature_dim, head_num, weights, bias)
        x = np.random.standard_normal((batch_size, seq_len, feature_dim))
        x = torch.from_numpy(x)
        # y_hat = torch_net(x, x, x)
        # print(y_hat)

        print(torchscript.compile(
            MultiHeadAttention(feature_dim, head_num, weights, bias),
            [x, x, x],
            output_type="linalg-on-tensors",
            enable_ir_printing=True,
        ))


t = TestMultiHeadAttention()
t.test_same_output_without_mask()
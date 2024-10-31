# RUN: python %s | FileCheck %s

import torch
from cinnamon.torch_backend.cinm import CinmBackend


def predictable_tensor(shape, dtype):
    count = torch.tensor(shape).prod()
    values = range(1, 1 + count)
    return torch.tensor(values, dtype=dtype).reshape(shape)


def predictable_param(param):
    return torch.nn.Parameter(
        predictable_tensor(param.shape, param.dtype), requires_grad=param.requires_grad
    )


class IntLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, dtype):
        super(IntLinear, self).__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.zeros((out_features, in_features)), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.zeros((out_features,)), requires_grad=False
        )

    def forward(self, x):
        x_int = x.to(self.dtype)
        weight_int = self.weight.to(self.dtype)
        bias_int = self.bias.to(self.dtype)
        return (torch.matmul(x_int, weight_int.t()) + bias_int).to(torch.float32)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = IntLinear(5, 3, torch.uint16)
        self.fc1.weight = predictable_param(self.fc1.weight)
        self.fc1.bias = predictable_param(self.fc1.bias)

        self.fc2 = IntLinear(3, 2, torch.uint16)
        self.fc2.weight = predictable_param(self.fc2.weight)
        self.fc2.bias = predictable_param(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = Model()

sample_input = predictable_tensor((1, 5), torch.float32)

backend = CinmBackend()
compiled_model = backend.compile(model, sample_input)

model_invoker = backend.load(compiled_model)

# CHECK: model via cinm [ 945 2134]
print("model via cinm", model_invoker.forward(sample_input).to(torch.int16).numpy()[0])

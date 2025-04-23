# RUN: python %s | FileCheck %s

import torch
from cinnamon.torch_backend.linalg_on_tensor import LinalgOnTensorBackend


def predictable_tensor(shape, dtype):
    count = torch.tensor(shape).prod()
    values = range(1, 1 + count)
    return torch.tensor(values, dtype=dtype).reshape(shape)


def predictable_param(param):
    return torch.nn.Parameter(predictable_tensor(param.shape, param.dtype))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = torch.nn.Linear(5, 5)
        self.fc1.weight = predictable_param(self.fc1.weight)
        self.fc1.bias = predictable_param(self.fc1.bias)

        self.fc2 = torch.nn.Linear(5, 10)
        self.fc2.weight = predictable_param(self.fc2.weight)
        self.fc2.bias = predictable_param(self.fc2.bias)

        self.fc3 = torch.nn.Linear(10, 2)
        self.fc3.weight = predictable_param(self.fc3.weight)
        self.fc3.bias = predictable_param(self.fc3.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = Model()

sample_input = predictable_tensor((1, 5), torch.float32)

backend = LinalgOnTensorBackend()
compiled_model = backend.compile(model, sample_input)

model_invoker = backend.load(compiled_model)

# CHECK: model via torch [1929786. 4658337.]
print("model via torch", model(sample_input).detach().numpy()[0])
# CHECK: model via cinm [1929786. 4658337.]
print("model via cinm", model_invoker.main(sample_input).numpy()[0])

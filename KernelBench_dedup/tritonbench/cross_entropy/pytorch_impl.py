import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(input, target)


# Default shapes from operator.py: B=8, T=2048, V=2^12=4096
B = 8
T = 2048
V = 4096


def get_inputs():
    input_tensor = torch.randn(B * T, V, dtype=torch.float32)
    target = torch.randint(V, (B * T,))
    return [input_tensor, target]


def get_init_inputs():
    return []


def get_test_inputs():
    return [x.cuda() for x in get_inputs()]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model().cuda().eval()
    with torch.no_grad():
        return model(*inputs)

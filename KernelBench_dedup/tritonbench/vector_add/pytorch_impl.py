import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


# Default shapes
SIZE = 2**18  # 262144


def get_inputs():
    return [torch.rand(SIZE, dtype=torch.float32), torch.rand(SIZE, dtype=torch.float32)]


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

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, reduce_dim=None):
        super().__init__()
        self.reduce_dim = reduce_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=self.reduce_dim)


# Default shapes: 2D input, reduce along dim 1
M = 512
N = 512
REDUCE_DIM = 1


def get_inputs():
    return [torch.randn(M, N, dtype=torch.float32)]


def get_init_inputs():
    return [REDUCE_DIM]


def get_test_inputs():
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)

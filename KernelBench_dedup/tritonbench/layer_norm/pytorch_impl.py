import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(normalized_shape))
        self.bias = nn.Parameter(torch.rand(normalized_shape))
        self.eps = 1e-5
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


# Default shapes from operator.py: M=4096, first N=1024 (512*2)
M = 4096
N = 1024


def get_inputs():
    return [torch.randn(M, N, dtype=torch.float32)]


def get_init_inputs():
    return [N]


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

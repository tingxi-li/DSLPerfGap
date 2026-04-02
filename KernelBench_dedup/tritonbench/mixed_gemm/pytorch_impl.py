import torch
import torch.nn as nn


class Model(nn.Module):
    """Baseline mixed GEMM: torch.matmul(a, w) in bf16."""

    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        return torch.matmul(a, w)


# Default shapes from _generate_default_shapes()
M = 4096
N = 8192
K = 4096
DTYPE = torch.bfloat16


def get_inputs():
    a = torch.randn(M, K, dtype=DTYPE)
    w = torch.randn(K, N, dtype=DTYPE)
    return [a, w]


def get_init_inputs():
    return []


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
